from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextTrainConfig:
    processed_dir: Path
    checkpoints_dir: Path
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5


def train_text(cfg: TextTrainConfig) -> Path:
    """Placeholder for text model training."""
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.checkpoints_dir / "text_model.pt"
    model_path.write_bytes(b"")
    return model_path

