from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class McqTrainConfig:
    processed_dir: Path
    checkpoints_dir: Path
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3


def train_mcq(cfg: McqTrainConfig) -> Path:
    """Placeholder for MCQ model training."""
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.checkpoints_dir / "mcq_model.pt"
    model_path.write_bytes(b"")
    return model_path

