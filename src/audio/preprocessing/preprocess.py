from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioPreprocessConfig:
    raw_dir: Path
    processed_dir: Path


def preprocess_audio(cfg: AudioPreprocessConfig) -> None:
    """Placeholder for audio preprocessing.

    Typical steps:
    - resampling / channel mixing
    - VAD / segmentation
    - loudness normalization
    - saving aligned segments to cfg.processed_dir
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally left as a stub for modality owner to implement.

