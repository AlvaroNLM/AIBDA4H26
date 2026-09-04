from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, probabilities) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    pred = (prob >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return {k: float(v) for k, v in result.items()}


def save_run_outputs(output_dir: Path, metrics: dict, predictions: pd.DataFrame, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")


def paired_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame, metric: str = "macro_f1",
                     iterations: int = 2000, seed: int = 2026) -> dict:
    keys = ["participant_id", "document_id"]
    merged = pred_a.merge(pred_b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.array_equal(merged.label_a, merged.label_b):
        raise ValueError("Paired predictions have inconsistent labels")
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(merged)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        a = classification_metrics(merged.label_a.iloc[idx], merged.probability_a.iloc[idx])[metric]
        b = classification_metrics(merged.label_b.iloc[idx], merged.probability_b.iloc[idx])[metric]
        diffs.append(a - b)
    nonpos = sum(x <= 0 for x in diffs); nonneg = sum(x >= 0 for x in diffs)
    return {"metric": metric, "observed_difference": classification_metrics(merged.label_a, merged.probability_a)[metric] - classification_metrics(merged.label_b, merged.probability_b)[metric],
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "two_sided_p": float(min(1.0, 2 * (1 + min(nonpos, nonneg)) / (iterations + 1))),
            "iterations": iterations}
