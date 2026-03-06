from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from evaluation.metrics import accuracy_from_proba, macro_f1_from_proba


def _get(output: Any, key: str) -> Any:
    if is_dataclass(output):
        return getattr(output, key)
    if isinstance(output, dict):
        return output[key]
    raise TypeError(f"Unsupported fusion output type: {type(output)!r}")


def evaluate_fusion(fusion_output: Any) -> dict[str, float]:
    """Compute a small set of standard metrics for fused outputs."""
    y_true = np.asarray(_get(fusion_output, "y_true"))
    fused_proba = np.asarray(_get(fusion_output, "fused_proba"))

    return {
        "accuracy": float(accuracy_from_proba(y_true, fused_proba)),
        "macro_f1": float(macro_f1_from_proba(y_true, fused_proba)),
    }


def to_dict(obj: Any) -> dict[str, Any]:
    """Helper for logging/debugging fusion outputs."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}

