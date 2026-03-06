from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.video.training.train import VideoTrainConfig, train_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train video modality model (skeleton).")
    p.add_argument("--config", type=Path, required=True, help="Path to video_config.yaml")
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/video"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    data_cfg = cfg_dict.get("data", {})
    train_cfg = cfg_dict.get("training", {})

    cfg = VideoTrainConfig(
        processed_dir=Path(data_cfg.get("processed_dir", "data/processed/video")),
        checkpoints_dir=args.checkpoints_dir,
        seed=int(train_cfg.get("seed", 42)),
        batch_size=int(train_cfg.get("batch_size", 8)),
        num_epochs=int(train_cfg.get("num_epochs", 10)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
    )

    out = train_video(cfg)
    print(f"Saved placeholder video model to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

