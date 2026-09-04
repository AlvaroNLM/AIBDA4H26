from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent.parent)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    return Path(cfg["_project_root"], cfg["paths"][key]).resolve()


def public_config(cfg: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}


def experiment_id(cfg: dict, variant: str, training_seed: int, fold: int,
                  random_variant_seed: int | None = None, model_name: str | None = None) -> str:
    fields = {
        "dataset": cfg["dataset"]["name"], "variant": variant,
        "random_variant_seed": random_variant_seed, "training_seed": training_seed,
        "fold": fold, "model_name": model_name or cfg["model"]["name"],
    }
    readable = "__".join(str(fields[k]).replace("/", "-") for k in fields)
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:10]
    return f"{readable}__{digest}"
