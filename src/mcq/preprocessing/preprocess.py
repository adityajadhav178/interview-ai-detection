from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class McqPreprocessConfig:
    raw_dir: Path
    processed_dir: Path


def preprocess_mcq(cfg: McqPreprocessConfig) -> None:
    """Placeholder for MCQ preprocessing.

    Typical steps:
    - schema validation (question ids, allowable answers)
    - missing-value handling
    - subject/session joins
    - exporting clean tabular data to cfg.processed_dir
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

