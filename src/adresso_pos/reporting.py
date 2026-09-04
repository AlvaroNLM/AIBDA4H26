from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .statistics import empirical_comparison


def collect_runs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in run_dir.glob("*/metrics.json"):
        if not (path.parent / "COMPLETED").exists(): continue
        metrics = json.loads(path.read_text())
        cfg = json.loads((path.parent / "config.json").read_text())
        rows.append({**cfg["experiment"], **metrics, "run_path": str(path.parent)})
    return pd.DataFrame(rows)


def generate_reports(run_dir: Path, report_dir: Path, variant_stats: pd.DataFrame | None = None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True); runs = collect_runs(run_dir)
    if runs.empty: return
    runs.to_csv(report_dir / "all_runs.csv", index=False); runs.to_json(report_dir / "all_runs.json", orient="records", indent=2)
    real = runs[~runs.variant.str.startswith("RANDOM_")]
    summary = real.groupby("variant").agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"), roc_auc_mean=("roc_auc", "mean")).reset_index()
    all_score = summary.loc[summary.variant == "ALL", "macro_f1_mean"]
    baseline = float(all_score.iloc[0]) if len(all_score) else float("nan")
    summary["relative_macro_f1"] = summary.macro_f1_mean / baseline
    summary["performance_drop"] = baseline - summary.macro_f1_mean
    if variant_stats is not None:
        summary = summary.merge(variant_stats[["variant", "tokens_mean"]].rename(columns={"tokens_mean": "n_tokens_mean"}), on="variant", how="left")
    summary.to_csv(report_dir / "final_condition_table.csv", index=False); summary.to_json(report_dir / "final_condition_table.json", orient="records", indent=2)
    comparisons = []
    for variant in real.variant.unique():
        if not (variant.startswith("ONLY_") or variant.startswith("WITHOUT_")): continue
        random_name = ("RANDOM_ONLY_MATCHED_" + variant[5:]) if variant.startswith("ONLY_") else ("RANDOM_REMOVE_MATCHED_" + variant[8:])
        random_scores = (runs.loc[runs.variant == random_name]
                         .groupby("random_variant_seed")["macro_f1"].mean().tolist())
        if random_scores:
            comparisons.append({"linguistic_condition": variant, "random_baseline": random_name,
                                **empirical_comparison(real.loc[real.variant == variant, "macro_f1"].mean(), random_scores)})
    pd.DataFrame(comparisons).to_csv(report_dir / "linguistic_vs_random.csv", index=False)
    sns.set_theme(style="whitegrid")
    _bar(summary, "macro_f1_mean", "macro_f1_std", "Macro-F1", report_dir / "macro_f1_by_condition.png")
    _bar(summary, "balanced_accuracy_mean", "balanced_accuracy_std", "Balanced accuracy", report_dir / "balanced_accuracy_by_condition.png")
    _bar(summary, "relative_macro_f1", None, "Relative macro-F1 vs ALL", report_dir / "relative_performance.png")
    without = summary[summary.variant.str.startswith("WITHOUT_")]
    _bar(without, "performance_drop", None, "Macro-F1 drop vs ALL", report_dir / "without_performance_drop.png")
    if "n_tokens_mean" in summary:
        fig, ax = plt.subplots(figsize=(8, 6)); sns.scatterplot(data=summary, x="n_tokens_mean", y="macro_f1_mean", hue="variant", ax=ax)
        fig.tight_layout(); fig.savefig(report_dir / "token_length_vs_performance.png", dpi=180); plt.close(fig)
    if comparisons:
        comp = pd.DataFrame(comparisons); fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(comp.linguistic_condition, comp.random_mean, yerr=[comp.random_mean-comp.random_95_ci_low, comp.random_95_ci_high-comp.random_mean], fmt="o", label="random matched 95% interval")
        ax.scatter(comp.linguistic_condition, comp.observed_score, marker="x", label="linguistic"); ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout()
        fig.savefig(report_dir / "linguistic_vs_random.png", dpi=180); plt.close(fig)


def _bar(data, value, error, ylabel, output):
    if data.empty: return
    fig, ax = plt.subplots(figsize=(max(8, .55 * len(data)), 6)); x = range(len(data))
    ax.bar(x, data[value], yerr=data[error] if error else None, capsize=3); ax.set_xticks(list(x), data.variant, rotation=55, ha="right"); ax.set_ylabel(ylabel)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
