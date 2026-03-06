from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.text.training.train import TextTrainConfig, train_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train text modality model (skeleton).")
    p.add_argument("--config", type=Path, required=True, help="Path to text_config.yaml")
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/text"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    data_cfg = cfg_dict.get("data", {})
    train_cfg = cfg_dict.get("training", {})

    cfg = TextTrainConfig(
        processed_dir=Path(data_cfg.get("processed_dir", "data/processed/text")),
        checkpoints_dir=args.checkpoints_dir,
        seed=int(train_cfg.get("seed", 42)),
        batch_size=int(train_cfg.get("batch_size", 16)),
        num_epochs=int(train_cfg.get("num_epochs", 3)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
    )

    out = train_text(cfg)
    print(f"Saved placeholder text model to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

