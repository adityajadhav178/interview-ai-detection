from __future__ import annotations

import numpy as np


def accuracy_from_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(proba).argmax(axis=-1).astype(int)
    return float((y_pred == y_true).mean())


def macro_f1_from_proba(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    """Simple macro-F1 implementation without external deps."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(proba).argmax(axis=-1).astype(int)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s: list[float] = []
    for c in classes:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0

