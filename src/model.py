from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json

import torch
from torch import Tensor, nn
from transformers import BertModel


class DiseaseAttentionHead(nn.Module):
    """Single disease-specific attention head over BERT sequence output."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.logit_proj = nn.Linear(hidden_size, 1)

    def _shape(self, x: Tensor, bsz: int, seq_len: int) -> Tensor:
        # (batch, seq_len, hidden) -> (batch, heads, seq_len, head_dim)
        return (
            x.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, sequence_output: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            sequence_output: (batch, seq_len, hidden_size)
            attention_mask:  (batch, seq_len) with 1 for tokens to keep, 0 for padding

        Returns:
            logit: (batch, 1)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = sequence_output.size()

        query = self._shape(self.q_proj(sequence_output), batch_size, seq_len)
        key = self._shape(self.k_proj(sequence_output), batch_size, seq_len)
        value = self._shape(self.v_proj(sequence_output), batch_size, seq_len)

        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, heads, seq_len, seq_len)
        scores = scores / (self.head_dim ** 0.5)

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, value)  # (batch, heads, seq_len, head_dim)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        context = self.out_proj(context)
        context = self.layer_norm(context + sequence_output)

        pooled = context.mean(dim=1)  # (batch, hidden_size)
        logit = self.logit_proj(pooled)  # (batch, 1)

        # Aggregate attention across heads and query positions for interpretability
        attn_per_token = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)

        return logit, attn_per_token


class BertMultiDiseaseClassifier(nn.Module):
    """BERT-based multi-label classifier for 4 mental health diseases."""

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_diseases: int = 4,
        disease_names: Optional[Iterable[str]] = None,
        dropout_rate: float = 0.3,
        freeze_bert_layers: int = 8,
        use_disease_attention: bool = True,
        hidden_size: int = 768,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()

        if disease_names is None:
            disease_names = ["depression", "anxiety", "ocd", "adhd"]
        disease_names = list(disease_names)
        if len(disease_names) != num_diseases:
            raise ValueError("num_diseases must match length of disease_names")

        self.bert_model_name = bert_model_name
        self.num_diseases = num_diseases
        self.disease_names = disease_names
        self.dropout_rate = dropout_rate
        self.freeze_bert_layers = freeze_bert_layers
        self.use_disease_attention = use_disease_attention
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        if hidden_size != bert_hidden:
            raise ValueError(f"hidden_size ({hidden_size}) must match BERT hidden size ({bert_hidden})")

        self._freeze_bert()

        self.shared_extractor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
        )

        heads: Dict[str, nn.Module] = {}
        for name in disease_names:
            if use_disease_attention:
                heads[name] = DiseaseAttentionHead(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout_rate,
                )
            else:
                heads[name] = nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, 1),
                )
        self.disease_heads = nn.ModuleDict(heads)

        self.sigmoid = nn.Sigmoid()

    def _freeze_bert(self) -> None:
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        encoder_layers = list(self.bert.encoder.layer)
        for i, layer in enumerate(encoder_layers):
            requires_grad = i >= self.freeze_bert_layers
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output: Tensor = outputs.last_hidden_state
        pooled_output: Tensor = outputs.pooler_output

        shared_features = self.shared_extractor(pooled_output)

        logits_per_disease: List[Tensor] = []
        attention_weights: Dict[str, Tensor] = {}

        for name in self.disease_names:
            head = self.disease_heads[name]
            if self.use_disease_attention:
                logit, attn = head(sequence_output, attention_mask)
                shared_logit = shared_features.mean(dim=-1, keepdim=True)
                logit = logit + shared_logit
                logits_per_disease.append(logit)
                attention_weights[name] = attn
            else:
                logit = head(shared_features)
                logits_per_disease.append(logit)

        logits = torch.cat(logits_per_disease, dim=1)
        probabilities = self.sigmoid(logits)

        return {
            "probabilities": probabilities,
            "logits": logits,
            "attention_weights": attention_weights,
            "shared_features": shared_features,
        }

    @torch.no_grad()
    def predict(
        self,
        probabilities: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, List[Dict[str, float]]]:
        binary = (probabilities >= threshold).long()
        batch_size = probabilities.size(0)

        results: List[Dict[str, float]] = []
        for i in range(batch_size):
            probs_i = probabilities[i].tolist()
            preds_i = binary[i].tolist()
            sample: Dict[str, Any] = {}
            present: List[str] = []
            for name, p, pred in zip(self.disease_names, probs_i, preds_i):
                sample[name] = float(p)
                if pred == 1:
                    present.append(name)
            sample["predictions"] = present
            results.append(sample)

        return binary, results

    @torch.no_grad()
    def get_disease_probabilities(self, probabilities: Tensor, threshold: float = 0.5) -> List[Dict[str, Any]]:
        _, result = self.predict(probabilities, threshold=threshold)
        return result

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> None:
        total = self.get_total_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable
        print(f"Total parameters    : {total}")
        print(f"Trainable parameters: {trainable}")
        print(f"Frozen parameters   : {frozen}")

        print("\nBERT encoder layer trainability:")
        for i, layer in enumerate(self.bert.encoder.layer):
            any_trainable = any(p.requires_grad for p in layer.parameters())
            status = "trainable" if any_trainable else "frozen"
            print(f"  Layer {i:02d}: {status}")


@dataclass
class ModelConfig:
    bert_model_name: str = "bert-base-uncased"
    num_diseases: int = 4
    disease_names: Tuple[str, str, str, str] = ("depression", "anxiety", "ocd", "adhd")
    dropout_rate: float = 0.3
    freeze_bert_layers: int = 8
    use_disease_attention: bool = True
    hidden_size: int = 768
    num_attention_heads: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 5
    batch_size: int = 16
    max_length: int = 256
    threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["disease_names"] = list(self.disease_names)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in fields}
        if "disease_names" in filtered and not isinstance(filtered["disease_names"], tuple):
            filtered["disease_names"] = tuple(filtered["disease_names"])
        return cls(**filtered)  # type: ignore[arg-type]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ModelConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MultiDiseaseLoss(nn.Module):
    """Multi-label BCE loss with disease-specific weights and pos_weight."""

    def __init__(
        self,
        disease_weights: Optional[Dict[str, float]] = None,
        pos_weight: Optional[Tensor] = None,
        disease_names: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()

        if disease_names is None:
            disease_names = ["depression", "anxiety", "ocd", "adhd"]
        self.disease_names = list(disease_names)

        if disease_weights is None:
            disease_weights = {name: 1.0 for name in self.disease_names}
        self.disease_weights = disease_weights

        if pos_weight is not None:
            if pos_weight.numel() != len(self.disease_names):
                raise ValueError("pos_weight must have shape (num_diseases,)")
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None  # type: ignore[assignment]

        self._criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, targets: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if self.pos_weight is not None:
            loss_per_element = nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
                reduction="none",
            )
        else:
            loss_per_element = self._criterion(logits, targets)

        per_disease_losses: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for idx, name in enumerate(self.disease_names):
            loss_d = loss_per_element[:, idx].mean()
            weight = float(self.disease_weights.get(name, 1.0))
            per_disease_losses[name] = float(loss_d.item())
            weighted_sum += weight * loss_d
            total_weight += weight

        total_loss = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        return total_loss, per_disease_losses


def build_model(config: ModelConfig) -> BertMultiDiseaseClassifier:
    model = BertMultiDiseaseClassifier(
        bert_model_name=config.bert_model_name,
        num_diseases=config.num_diseases,
        disease_names=config.disease_names,
        dropout_rate=config.dropout_rate,
        freeze_bert_layers=config.freeze_bert_layers,
        use_disease_attention=config.use_disease_attention,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
    )
    print(model)
    model.get_layer_info()
    return model


def compute_pos_weights(dataset_stats_path: Path | str) -> Tensor:
    """Compute pos_weight = num_negative / num_positive per disease from dataset_stats.json."""
    path = Path(dataset_stats_path)
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    label_counts = stats.get("label_counts_per_disease", {})
    disease_order = ["depression", "anxiety", "ocd", "adhd"]
    weights: List[float] = []

    for name in disease_order:
        counts = label_counts.get(name, {})
        n_pos = float(counts.get("1", 1.0))
        n_neg = float(counts.get("0", 1.0))
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        weights.append(pos_weight)

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    cfg = ModelConfig()
    model = build_model(cfg)

    print("\nParameter counts:")
    print(f"  Total params    : {model.get_total_params()}")
    print(f"  Trainable params: {model.get_trainable_params()}")

    batch_size = 2
    seq_len = cfg.max_length
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        token_type_ids=dummy_token_type_ids,
    )

    probs: Tensor = outputs["probabilities"]
    logits: Tensor = outputs["logits"]
    shared_features: Tensor = outputs["shared_features"]
    attn = outputs["attention_weights"]

    print("\nOutput shapes:")
    print(f"  probabilities   : {tuple(probs.shape)}")
    print(f"  logits          : {tuple(logits.shape)}")
    print(f"  shared_features : {tuple(shared_features.shape)}")
    for name, w in attn.items():
        print(f"  attention[{name}]: {tuple(w.shape)}")

    human_readable = model.get_disease_probabilities(probs, threshold=cfg.threshold)
    print("\nDisease probabilities (demo):")
    for i, sample in enumerate(human_readable):
        print(f"Sample {i}: {sample}")

    checkpoints_dir = Path("checkpoints")
    cfg_path = checkpoints_dir / "model_config.json"
    cfg.save(cfg_path)
    print(f"\nSaved ModelConfig to: {cfg_path.resolve()}")

