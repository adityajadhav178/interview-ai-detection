from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.model import (
    BertMultiDiseaseClassifier,
    ModelConfig,
    MultiDiseaseLoss,
    build_model,
    compute_pos_weights,
)


@dataclass(frozen=True)
class TextTrainConfig:
    processed_dir: Path
    checkpoints_dir: Path
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5


class MentalHealthDataset(Dataset):
    """Dataset that wraps pre-tokenized encodings and per-disease labels."""

    def __init__(self, df: pd.DataFrame, encodings: Dict[str, Tensor], disease_to_id: Dict[str, int]) -> None:
        self.df = df.reset_index(drop=True)
        self.encodings = encodings
        self.disease_to_id = disease_to_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        row = self.df.iloc[idx]
        disease = str(row["disease"])
        label_scalar = int(row["label"])

        # Multi-label target: only the disease for this sample has a label; others are treated as 0.
        num_diseases = len(self.disease_to_id)
        target = torch.zeros(num_diseases, dtype=torch.float32)
        disease_idx = self.disease_to_id.get(disease, -1)
        if disease_idx >= 0:
            target[disease_idx] = float(label_scalar)

        item: Dict[str, Tensor] = {
            "input_ids": self.encodings["input_ids"][idx].long(),
            "attention_mask": self.encodings["attention_mask"][idx].long(),
            "labels": target,
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx].long()
        else:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])
        return item


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_processed_artifacts(
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, int], Path]:
    train_pkl = processed_dir / "train.pkl"
    val_pkl = processed_dir / "val.pkl"
    test_pkl = processed_dir / "test.pkl"

    enc_train_path = processed_dir / "train_encodings.pt"
    enc_val_path = processed_dir / "val_encodings.pt"
    enc_test_path = processed_dir / "test_encodings.pt"

    label_mappings_path = processed_dir / "label_mappings.json"

    if not train_pkl.exists():
        raise FileNotFoundError(f"Expected train.pkl at {train_pkl}")

    df_train = pd.read_pickle(train_pkl)
    df_val = pd.read_pickle(val_pkl)
    df_test = pd.read_pickle(test_pkl)

    enc_train: Dict[str, Tensor] = torch.load(enc_train_path)
    enc_val: Dict[str, Tensor] = torch.load(enc_val_path)
    enc_test: Dict[str, Tensor] = torch.load(enc_test_path)

    with label_mappings_path.open("r", encoding="utf-8") as f:
        label_mappings = json.load(f)
    disease_to_id = {k: int(v) for k, v in label_mappings["disease_to_id"].items()}

    return df_train, df_val, df_test, enc_train, enc_val, enc_test, disease_to_id, label_mappings_path


def train_text(cfg: TextTrainConfig) -> Path:
    """Train BERT-based multi-disease classifier on preprocessed text data."""
    _set_seed(cfg.seed)

    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        df_train,
        df_val,
        _,
        enc_train,
        enc_val,
        _,
        disease_to_id,
        label_mappings_path,
    ) = _load_processed_artifacts(cfg.processed_dir)

    train_dataset = MentalHealthDataset(df_train, enc_train, disease_to_id)
    val_dataset = MentalHealthDataset(df_val, enc_val, disease_to_id)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Build model configuration, using defaults; max_length is determined by preprocessing.
    model_cfg = ModelConfig()
    model: BertMultiDiseaseClassifier = build_model(model_cfg).to(device)

    # Loss and optimizer
    stats_path = label_mappings_path.with_name("dataset_stats.json")
    pos_weight = compute_pos_weights(stats_path).to(device)
    criterion = MultiDiseaseLoss(pos_weight=pos_weight, disease_names=model.disease_names)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_model_path = cfg.checkpoints_dir / "text_model.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs["logits"]

            loss, _ = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs["logits"]
                loss, _ = criterion(logits, labels)
                val_loss += float(loss.item())
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(f"Epoch {epoch}/{cfg.num_epochs} - train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")

    return best_model_path


