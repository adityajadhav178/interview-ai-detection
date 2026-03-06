from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.audio.training.train import AudioTrainConfig, train_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train audio modality model (skeleton).")
    p.add_argument("--config", type=Path, required=True, help="Path to audio_config.yaml")
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/audio"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    data_cfg = cfg_dict.get("data", {})
    train_cfg = cfg_dict.get("training", {})

    cfg = AudioTrainConfig(
        processed_dir=Path(data_cfg.get("processed_dir", "data/processed/audio")),
        checkpoints_dir=args.checkpoints_dir,
        seed=int(train_cfg.get("seed", 42)),
        batch_size=int(train_cfg.get("batch_size", 16)),
        num_epochs=int(train_cfg.get("num_epochs", 10)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
    )

    out = train_audio(cfg)
    print(f"Saved placeholder audio model to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

