"""
학습 진입점.

Usage:
  # Phase 1 (unbraid 사전학습):
  python scripts/train.py --config configs/phase1.yaml

  # Phase 2 (braid 파인튜닝):
  python scripts/train.py --config configs/phase2.yaml
"""

import argparse
import logging
import sys
import os

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def deep_merge(base: dict, override: dict) -> dict:
    """override를 base에 재귀적으로 병합. override가 우선."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    """YAML 로드. base: 키가 있으면 base config와 병합."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.pop("base", None)
    if base_path:
        with open(base_path, encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.getLogger(__name__).info(
        f"Config 로드: dataset={cfg['training']['dataset']}, "
        f"output={cfg['checkpointing']['output_dir']}"
    )

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
