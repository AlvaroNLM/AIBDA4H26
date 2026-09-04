from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FOCUS_POS = ["VERB", "PRON", "NOUN", "ADJ", "ADV"]


def descriptive_statistics(variant_df: pd.DataFrame, tokens: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = variant_df.n_tokens
    summary = pd.DataFrame([{
        "variant": variant_df.variant.iloc[0], "documents": len(variant_df),
        "class_0_documents": int((variant_df.label == 0).sum()), "class_1_documents": int((variant_df.label == 1).sum()),
        "total_tokens": int(lengths.sum()), "tokens_mean": lengths.mean(), "tokens_median": lengths.median(),
        "tokens_std": lengths.std(ddof=1), "tokens_min": lengths.min(), "tokens_max": lengths.max(),
        "empty_documents": int(variant_df.was_empty.sum()), "empty_percent": 100 * variant_df.was_empty.mean(),
    }])
    summary.to_csv(output_dir / "document_statistics.csv", index=False)
    selected = variant_df[["participant_id", "label", "split", "selected_pos"]].copy()
    selected["pos"] = selected.selected_pos.str.split()
    relevant = selected.explode("pos").dropna(subset=["pos"])
    pos_class = relevant.groupby(["split", "label", "pos"]).size().rename("count").reset_index()
    totals = pos_class.groupby(["split", "label"])["count"].transform("sum")
    pos_class["percentage"] = 100 * pos_class["count"] / totals
    pos_class.to_csv(output_dir / "pos_distribution_by_class.csv", index=False)
    per_participant = (relevant[relevant.pos.isin(FOCUS_POS)].groupby(["participant_id", "label", "pos"])
                       .size().unstack(fill_value=0).reindex(columns=FOCUS_POS, fill_value=0).reset_index())
    per_participant.groupby("label")[FOCUS_POS].mean().reset_index().to_csv(output_dir / "mean_focus_pos_by_participant_class.csv", index=False)


def empirical_comparison(observed: float, random_scores: list[float]) -> dict:
    scores = np.asarray(random_scores, dtype=float)
    if not len(scores):
        raise ValueError("Random score distribution is empty")
    # One-sided randomization p: probability that a matched random control is at least as good.
    return {"observed_score": observed, "random_mean": float(scores.mean()),
            "random_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "random_95_ci_low": float(np.percentile(scores, 2.5)),
            "random_95_ci_high": float(np.percentile(scores, 97.5)),
            "empirical_p_value": float((1 + np.sum(scores >= observed)) / (len(scores) + 1)),
            "observed_percentile_in_random": float(100 * np.mean(scores <= observed)),
            "n_random": int(len(scores))}
