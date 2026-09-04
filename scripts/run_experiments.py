#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adresso_pos.config import load_config
from adresso_pos.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="ADReSSo21 controlled POS experiments")
    parser.add_argument("--config", default="configs/adresso_pos.yaml")
    parser.add_argument("--variant", help="Run one variant (for example ONLY_VERB_PRON)")
    parser.add_argument("--seed", type=int, help="Run one training seed")
    parser.add_argument("--fold", type=int, help="Run one zero-based fold")
    parser.add_argument("--random-seed", type=int, help="Random matched construction seed")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    run(cfg, args.variant, args.seed, args.fold, args.random_seed, args.prepare_only, args.force_prepare)


if __name__ == "__main__": main()
