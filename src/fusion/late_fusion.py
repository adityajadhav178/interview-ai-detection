from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LateFusionOutput:
    y_true: np.ndarray
    per_modality_proba: dict[str, np.ndarray]
    fused_proba: np.ndarray


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def run_late_fusion(configs: dict[str, dict[str, Any]]) -> LateFusionOutput:
    """Placeholder late fusion runner.

    For scaffolding, we generate deterministic dummy probabilities so the pipeline runs.
    Replace with:
    - load each modality's trained model
    - compute per-sample probabilities
    - combine (weighted avg / stacking / calibrated ensembling)
    """
    _ = configs

    rng = np.random.default_rng(0)
    n = 32
    num_classes = 2

    per_modality_logits = {
        "audio": rng.normal(size=(n, num_classes)),
        "video": rng.normal(size=(n, num_classes)),
        "text": rng.normal(size=(n, num_classes)),
        "mcq": rng.normal(size=(n, num_classes)),
    }
    per_modality_proba = {k: _softmax(v) for k, v in per_modality_logits.items()}

    fused_proba = sum(per_modality_proba.values()) / len(per_modality_proba)
    y_true = rng.integers(0, num_classes, size=(n,))

    return LateFusionOutput(
        y_true=y_true,
        per_modality_proba=per_modality_proba,
        fused_proba=fused_proba,
    )

