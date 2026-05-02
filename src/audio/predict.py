"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Inference / Prediction
  Load trained model → process new audio → classify
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from . import config as C
from .preprocess import clean_audio
from .features import extract_features


class AudioPredictor:
    """
    End-to-end predictor for mental health classification from audio.

    Usage
    -----
        predictor = AudioPredictor()
        result    = predictor.predict("path/to/audio.wav")
        print(result)
        # {"label": 1, "condition": "Depression", "confidence": 0.87,
        #  "probabilities": {"Normal": 0.05, "Depression": 0.87, ...}}
    """

    def __init__(self, model_dir: str | Path = C.MODEL_DIR):
        md = Path(model_dir)

        scaler_path = md / "scaler.joblib"
        model_path  = md / "best_audio_model.joblib"

        if not scaler_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Missing model artifacts in {md}. Run training first."
            )

        self.scaler   = joblib.load(scaler_path)
        bundle        = joblib.load(model_path)
        if isinstance(bundle, dict):
            self.model    = bundle.get("model", bundle)
            self.label_map = bundle.get("label_map", {})
            self.feat_cols = bundle.get("features", [])
        else:
            self.model    = bundle
            self.label_map = {}
            self.feat_cols = []

    # ─────────────────────────────────────────────────────────

    def predict(self, audio_path: str | Path) -> dict | None:
        """
        Classify a single audio file.

        Returns dict with keys:
          label, condition, confidence, probabilities
        """
        try:
            audio, sr = clean_audio(audio_path)
            if audio is None or len(audio) == 0:
                print("  ✗ Could not clean audio")
                return None

            feats = extract_features(audio, sr)
            if feats is None:
                print("  ✗ Feature extraction failed")
                return None

            df = pd.DataFrame([feats])
            if self.feat_cols:
                df = df.reindex(columns=self.feat_cols, fill_value=0.0)

            X = self.scaler.transform(df)
            pred = int(self.model.predict(X)[0])

            # map back to original label
            orig = self.label_map.get(pred, pred)

            # probabilities
            probs = {}
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(X)[0]
                for idx, prob in enumerate(p):
                    lbl = self.label_map.get(idx, idx)
                    name = C.LABEL_NAMES.get(lbl, str(lbl))
                    probs[name] = round(float(prob), 4)

            cond = C.LABEL_NAMES.get(orig, str(orig))
            conf = max(probs.values()) if probs else 0.0

            return {
                "label":         orig,
                "condition":     cond,
                "confidence":    conf,
                "probabilities": probs,
            }

        except Exception as e:
            print(f"  ✗ Prediction error: {e}")
            return None

    # ─────────────────────────────────────────────────────────

    def predict_folder(self, folder: str | Path) -> dict | None:
        """
        Classify from a folder of WAV files (multi-answer interview).
        Uses majority-voting across individual file predictions.
        """
        folder = Path(folder)
        wavs = sorted(folder.glob("*.wav"))

        if not wavs:
            print(f"  ✗ No WAV files in {folder}")
            return None

        preds = []
        for wav in wavs[:10]:         # cap at 10 files
            r = self.predict(wav)
            if r:
                preds.append(r)

        if not preds:
            return None

        # Average probabilities across files
        all_probs: dict[str, list] = {}
        for r in preds:
            for cond, prob in r["probabilities"].items():
                all_probs.setdefault(cond, []).append(prob)

        avg_probs = {c: round(np.mean(ps), 4) for c, ps in all_probs.items()}
        best_cond = max(avg_probs, key=avg_probs.get)
        best_label = next(
            (lbl for lbl, name in C.LABEL_NAMES.items() if name == best_cond),
            -1
        )

        return {
            "label":         best_label,
            "condition":     best_cond,
            "confidence":    avg_probs[best_cond],
            "probabilities": avg_probs,
            "files_used":    len(preds),
        }


# ═════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict mental health from audio")
    parser.add_argument("input", help="Path to .wav file or folder of .wav files")
    parser.add_argument("--model-dir", default=str(C.MODEL_DIR))
    args = parser.parse_args()

    predictor = AudioPredictor(args.model_dir)
    path = Path(args.input)

    if path.is_dir():
        result = predictor.predict_folder(path)
    else:
        result = predictor.predict(path)

    if result:
        print(f"\n  ★ Prediction: {result['condition']}  "
              f"(confidence {result['confidence']:.1%})")
        for cond, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"    {cond:>12}  {prob:>6.1%}  {bar}")
    else:
        print("\n  ✗ Prediction failed")
