from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .baselines import train_pos_counts, train_pos_sequence
from .config import experiment_id, public_config, resolve_path
from .data import discover_data
from .metrics import save_run_outputs
from .pos import prepare_pos_cache
from .reporting import generate_reports
from .splits import make_folds
from .statistics import descriptive_statistics
from .train_bert import train_bert
from .variants import build_variant, random_baseline_for, variant_specs

LOG = logging.getLogger(__name__)


def prepare(cfg: dict, force: bool = False):
    cache = resolve_path(cfg, "cache_dir"); cache.mkdir(parents=True, exist_ok=True)
    train, test = discover_data(cfg); documents = pd.concat([train, test] if test is not None else [train], ignore_index=True)
    documents.to_parquet(cache / "documents.parquet", index=False)
    tokens = prepare_pos_cache(documents, cache / "pos_tokens.parquet", cfg["pos"], force=force)
    folds_path = cache / "fold_assignments.csv"
    folds = make_folds(documents, cfg["n_splits"], cfg["training_seeds"][0])
    folds.to_csv(folds_path, index=False)
    variants_dir = cache / "variants"; variants_dir.mkdir(exist_ok=True)
    stats_rows = []
    for variant in variant_specs(cfg["hypotheses"]):
        frame = build_variant(tokens, variant, cfg["hypotheses"])
        frame.to_parquet(variants_dir / f"{variant}.parquet", index=False)
        descriptive_statistics(frame, tokens, resolve_path(cfg, "report_dir") / "descriptive" / variant)
        stats_rows.append({"variant": variant, "tokens_mean": frame.n_tokens.mean(), "empty_documents": int(frame.was_empty.sum())})
    for real in [v for v in variant_specs(cfg["hypotheses"]) if v.startswith(("ONLY_", "WITHOUT_"))]:
        random_name = random_baseline_for(real)
        for seed in cfg["random_seeds"]:
            frame = build_variant(tokens, random_name, cfg["hypotheses"], seed)
            frame.to_parquet(variants_dir / f"{random_name}__rseed-{seed}.parquet", index=False)
            descriptive_statistics(frame, tokens, resolve_path(cfg, "report_dir") / "descriptive" / random_name / f"seed-{seed}")
        stats_rows.append({"variant": random_name, "tokens_mean": frame.n_tokens.mean(), "empty_documents": int(frame.was_empty.sum())})
    stats = pd.DataFrame(stats_rows); stats.to_csv(cache / "variant_statistics.csv", index=False)
    _plot_pos_distribution(tokens, resolve_path(cfg, "report_dir") / "pos_distribution_by_class.png")
    LOG.info("Prepared %d documents and %d tokens", len(documents), len(tokens))
    return documents, tokens, folds


def available_variants(cfg):
    real = list(variant_specs(cfg["hypotheses"])) + ["POS_COUNTS"]
    randoms = [random_baseline_for(v) for v in real if v.startswith(("ONLY_", "WITHOUT_"))]
    return real + randoms


def run(cfg: dict, selected_variant: str | None = None, selected_seed: int | None = None,
        selected_fold: int | None = None, selected_random_seed: int | None = None,
        prepare_only: bool = False, force_prepare: bool = False):
    cache = resolve_path(cfg, "cache_dir")
    required = [cache / "documents.parquet", cache / "pos_tokens.parquet", cache / "fold_assignments.csv"]
    if force_prepare or not all(p.exists() for p in required): prepare(cfg, force=force_prepare)
    if prepare_only: return
    documents = pd.read_parquet(cache / "documents.parquet"); tokens = pd.read_parquet(cache / "pos_tokens.parquet")
    folds = pd.read_csv(cache / "fold_assignments.csv"); variants = [selected_variant] if selected_variant else available_variants(cfg)
    unknown = set(variants) - set(available_variants(cfg))
    if unknown: raise ValueError(f"Unknown variants {unknown}; choose from {available_variants(cfg)}")
    seeds = [selected_seed] if selected_seed is not None else cfg["training_seeds"]
    fold_values = [selected_fold] if selected_fold is not None else range(cfg["n_splits"])
    for variant in variants:
        random_seeds = ([selected_random_seed] if selected_random_seed is not None else cfg["random_seeds"]) if variant.startswith("RANDOM_") else [None]
        for random_seed in random_seeds:
            frame = None if variant == "POS_COUNTS" else _load_variant(cache, variant, random_seed)
            for seed in seeds:
                for fold in fold_values:
                    _run_one(cfg, documents, tokens, folds, frame, variant, seed, fold, random_seed)
    stats_path = cache / "variant_statistics.csv"
    generate_reports(resolve_path(cfg, "run_dir"), resolve_path(cfg, "report_dir"), pd.read_csv(stats_path) if stats_path.exists() else None)


def _load_variant(cache, variant, random_seed):
    suffix = f"__rseed-{random_seed}" if variant.startswith("RANDOM_") else ""
    path = cache / "variants" / f"{variant}{suffix}.parquet"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}; run --prepare-only first")
    return pd.read_parquet(path)


def _run_one(cfg, documents, tokens, folds, frame, variant, seed, fold, random_seed):
    exp_id = experiment_id(cfg, variant, seed, fold, random_seed,
                           "logistic-regression" if variant == "POS_COUNTS" else ("pos-bilstm" if variant == "POS_SEQUENCE" else None))
    output = resolve_path(cfg, "run_dir") / exp_id
    if (output / "COMPLETED").exists(): LOG.info("Skipping completed %s", exp_id); return
    assignment = folds[folds.fold == fold]
    train_doc_ids = set(assignment.loc[assignment.role == "train", "document_id"])
    val_doc_ids = set(assignment.loc[assignment.role == "validation", "document_id"])
    train_participants = set(documents.loc[documents.document_id.isin(train_doc_ids), "participant_id"])
    val_participants = set(documents.loc[documents.document_id.isin(val_doc_ids), "participant_id"])
    if train_participants & val_participants: raise AssertionError("Participant leakage")
    if variant == "POS_COUNTS":
        metrics, predictions, features = train_pos_counts(tokens[tokens.split == "train"], train_participants, val_participants, seed)
        metrics["features"] = features; model_name = "logistic-regression"
    else:
        train_df = frame[frame.document_id.isin(train_doc_ids)]; val_df = frame[frame.document_id.isin(val_doc_ids)]
        test_df = frame[frame.split == "test"]
        if variant == "POS_SEQUENCE":
            metrics, predictions = train_pos_sequence(train_df, val_df, cfg, seed); model_name = "pos-bilstm"
        else:
            metrics, predictions = train_bert(train_df, val_df, test_df, cfg, seed, output); model_name = cfg["model"]["name"]
    predictions["seed"] = seed; predictions["fold"] = fold; predictions["variant"] = variant; predictions["random_variant_seed"] = random_seed
    predictions["best_epoch"] = metrics.get("best_epoch")
    run_cfg = {"pipeline": public_config(cfg), "experiment": {"experiment_id": exp_id, "dataset": cfg["dataset"]["name"], "variant": variant,
               "random_variant_seed": random_seed, "training_seed": seed, "fold": fold, "model_name": model_name}}
    save_run_outputs(output, metrics, predictions, run_cfg); LOG.info("Completed %s", exp_id)


def _plot_pos_distribution(tokens, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    output.parent.mkdir(parents=True, exist_ok=True)
    data = tokens.loc[~tokens.is_space].groupby(["label", "pos"]).size().rename("count").reset_index()
    data["percentage"] = 100 * data["count"] / data.groupby("label")["count"].transform("sum")
    fig, ax = plt.subplots(figsize=(11, 6)); sns.barplot(data=data, x="pos", y="percentage", hue="label", ax=ax)
    ax.set_ylabel("Tokens (%)"); ax.set_xlabel("Universal POS"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
