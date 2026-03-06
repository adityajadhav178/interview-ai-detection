from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.mcq.training.train import McqTrainConfig, train_mcq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MCQ modality model (skeleton).")
    p.add_argument("--config", type=Path, required=True, help="Path to mcq_config.yaml")
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/mcq"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    data_cfg = cfg_dict.get("data", {})
    train_cfg = cfg_dict.get("training", {})

    cfg = McqTrainConfig(
        processed_dir=Path(data_cfg.get("processed_dir", "data/processed/mcq")),
        checkpoints_dir=args.checkpoints_dir,
        seed=int(train_cfg.get("seed", 42)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_epochs=int(train_cfg.get("num_epochs", 50)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
    )

    out = train_mcq(cfg)
    print(f"Saved placeholder MCQ model to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

