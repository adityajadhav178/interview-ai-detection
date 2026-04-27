"""
Audio data audit utility.

Reports class balance across:
1) Raw audio files
2) Segmented audio files
3) Feature CSV rows

Usage:
  python scripts/audio_data_audit.py
"""

from __future__ import annotations

import json
import contextlib
import wave
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
SEG_DIR = PROJECT_ROOT / "data" / "raw" / "audio_segmented"
FEATURES_CSV = PROJECT_ROOT / "data" / "final_features" / "features.csv"
BALANCED_CSV = PROJECT_ROOT / "data" / "final_features" / "features_balanced.csv"
OUT_DIR = PROJECT_ROOT / "evaluation" / "audio"

CLASSES = ["normal", "depression", "anxiety", "adhd", "ocd"]


def wav_duration_sec(path: Path) -> float | None:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return None


def collect_audio_stats(base_dir: Path, split_name: str) -> pd.DataFrame:
    rows: list[dict] = []

    for cls in CLASSES:
        cls_dir = base_dir / cls
        wavs = list(cls_dir.rglob("*.wav")) if cls_dir.exists() else []

        durs: list[float] = []
        for wav_path in wavs:
            dur = wav_duration_sec(wav_path)
            if dur is not None:
                durs.append(dur)

        total = float(sum(durs))
        series = pd.Series(durs, dtype=float)
        rows.append(
            {
                "split": split_name,
                "class": cls,
                "files": int(len(wavs)),
                "readable_files": int(len(durs)),
                "total_sec": round(total, 2),
                "total_hr": round(total / 3600.0, 2),
                "mean_sec": round(float(series.mean()) if not series.empty else 0.0, 2),
                "median_sec": round(float(series.median()) if not series.empty else 0.0, 2),
                "min_sec": round(float(series.min()) if not series.empty else 0.0, 2),
                "max_sec": round(float(series.max()) if not series.empty else 0.0, 2),
            }
        )

    return pd.DataFrame(rows)


def collect_feature_counts(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["source", "class", "count", "ratio_to_max"])

    df = pd.read_csv(path)

    if "condition" in df.columns:
        counts = df["condition"].astype(str).str.lower().value_counts().sort_index()
    elif "label" in df.columns:
        counts = df["label"].astype(str).value_counts().sort_index()
    else:
        return pd.DataFrame(columns=["source", "class", "count", "ratio_to_max"])

    max_count = int(counts.max()) if len(counts) else 1
    rows = []
    for cls, cnt in counts.items():
        rows.append(
            {
                "source": name,
                "class": str(cls),
                "count": int(cnt),
                "ratio_to_max": round(float(cnt) / max_count if max_count else 0.0, 4),
            }
        )

    return pd.DataFrame(rows)


def print_imbalance_warnings(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        print(f"\n{title}: no data")
        return

    max_files = int(df["files"].max()) if "files" in df.columns else 0
    min_files = int(df["files"].min()) if "files" in df.columns else 0
    ratio = (max_files / min_files) if min_files > 0 else float("inf")

    print(f"\n{title}")
    print(df.to_string(index=False))
    if ratio >= 2.0:
        print(f"WARNING: file-count imbalance ratio is {ratio:.2f}x")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_stats = collect_audio_stats(RAW_DIR, "raw")
    seg_stats = collect_audio_stats(SEG_DIR, "segmented")

    feat_counts = collect_feature_counts(FEATURES_CSV, "features.csv")
    bal_counts = collect_feature_counts(BALANCED_CSV, "features_balanced.csv")

    print_imbalance_warnings(raw_stats, "RAW AUDIO STATS")
    print_imbalance_warnings(seg_stats, "SEGMENTED AUDIO STATS")

    print("\nFEATURE ROW COUNTS")
    if not feat_counts.empty:
        print(feat_counts.to_string(index=False))
    else:
        print("features.csv not found or unsupported schema")

    print("\nBALANCED FEATURE ROW COUNTS")
    if not bal_counts.empty:
        print(bal_counts.to_string(index=False))
    else:
        print("features_balanced.csv not found or unsupported schema")

    audio_summary = pd.concat([raw_stats, seg_stats], ignore_index=True)
    audio_summary.to_csv(OUT_DIR / "audio_data_audit_summary.csv", index=False)

    summary = {
        "raw": raw_stats.to_dict(orient="records"),
        "segmented": seg_stats.to_dict(orient="records"),
        "features": feat_counts.to_dict(orient="records"),
        "features_balanced": bal_counts.to_dict(orient="records"),
    }
    (OUT_DIR / "audio_data_audit_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(f"- {OUT_DIR / 'audio_data_audit_summary.csv'}")
    print(f"- {OUT_DIR / 'audio_data_audit_summary.json'}")


if __name__ == "__main__":
    main()
