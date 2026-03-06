from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.fusion.evaluation import evaluate_fusion
from src.fusion.early_fusion import run_early_fusion
from src.fusion.late_fusion import run_late_fusion


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal interview AI detection pipeline (skeleton).")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing modality config YAMLs.",
    )
    p.add_argument(
        "--fusion",
        choices=["early", "late"],
        default="late",
        help="Fusion strategy to run.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_dir = args.config_dir

    configs = {
        "audio": load_yaml(config_dir / "audio_config.yaml"),
        "video": load_yaml(config_dir / "video_config.yaml"),
        "text": load_yaml(config_dir / "text_config.yaml"),
        "mcq": load_yaml(config_dir / "mcq_config.yaml"),
    }

    if args.fusion == "early":
        outputs = run_early_fusion(configs)
    else:
        outputs = run_late_fusion(configs)

    metrics = evaluate_fusion(outputs)
    print("Fusion metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

