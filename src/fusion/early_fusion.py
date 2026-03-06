from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EarlyFusionOutput:
    y_true: np.ndarray
    fused_proba: np.ndarray


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def run_early_fusion(configs: dict[str, dict[str, Any]]) -> EarlyFusionOutput:
    """Placeholder early fusion runner.

    Replace with:
    - extract per-modality embeddings/features for aligned samples
    - concatenate / cross-attend / multi-branch network
    - train/infer a single fusion classifier
    """
    _ = configs

    rng = np.random.default_rng(1)
    n = 32
    num_classes = 2

    fused_logits = rng.normal(size=(n, num_classes))
    fused_proba = _softmax(fused_logits)
    y_true = rng.integers(0, num_classes, size=(n,))
    return EarlyFusionOutput(y_true=y_true, fused_proba=fused_proba)

