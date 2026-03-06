from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextPreprocessConfig:
    raw_dir: Path
    processed_dir: Path


def preprocess_text(cfg: TextPreprocessConfig) -> None:
    """Placeholder for transcript preprocessing.

    Typical steps:
    - normalization (punctuation/casing)
    - diarization-aware formatting
    - segment alignment with audio/video timestamps
    - saving cleaned text + metadata to cfg.processed_dir
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

