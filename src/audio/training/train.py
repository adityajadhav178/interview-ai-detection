from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioTrainConfig:
    processed_dir: Path
    checkpoints_dir: Path
    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 3e-4


def train_audio(cfg: AudioTrainConfig) -> Path:
    """Placeholder for audio model training.

    Returns the path to the saved model artifact (or best checkpoint).
    """
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.checkpoints_dir / "audio_model.pt"
    model_path.write_bytes(b"")  # placeholder artifact
    return model_path

