from __future__ import annotations

import hashlib
import random

import pandas as pd


def variant_specs(hypotheses: list[dict]) -> dict[str, set[str] | None]:
    specs: dict[str, set[str] | None] = {"ALL": None, "POS_SEQUENCE": None}
    for hyp in hypotheses:
        tags = set(hyp["include_pos"])
        name = hyp["name"].upper()
        specs[f"ONLY_{name}"] = tags
        specs[f"WITHOUT_{name}"] = tags
    return specs


def random_baseline_for(variant: str) -> str:
    if variant.startswith("ONLY_"):
        return "RANDOM_ONLY_MATCHED_" + variant[5:]
    if variant.startswith("WITHOUT_"):
        return "RANDOM_REMOVE_MATCHED_" + variant[8:]
    raise ValueError(f"No matched baseline for {variant}")


def _document_rng(seed: int, document_id: str) -> random.Random:
    stable = int(hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()[:16], 16)
    return random.Random(stable)


def build_variant(tokens: pd.DataFrame, variant: str, hypotheses: list[dict],
                  random_seed: int | None = None) -> pd.DataFrame:
    specs = variant_specs(hypotheses)
    random_only = variant.startswith("RANDOM_ONLY_MATCHED_")
    random_remove = variant.startswith("RANDOM_REMOVE_MATCHED_")
    if random_only:
        source = "ONLY_" + variant[len("RANDOM_ONLY_MATCHED_"):]
    elif random_remove:
        source = "WITHOUT_" + variant[len("RANDOM_REMOVE_MATCHED_"):]
    else:
        source = variant
    if source not in specs:
        raise KeyError(f"Unknown variant {variant}; known linguistic variants: {sorted(specs)}")
    tags = specs[source]
    records = []
    for doc_id, group in tokens.sort_values("position").groupby("document_id", sort=False):
        group = group.loc[~group.is_space].copy()
        if variant == "ALL":
            selected = group
        elif variant == "POS_SEQUENCE":
            selected = group
        elif random_only or random_remove:
            if random_seed is None:
                raise ValueError("A random_seed is required for random matched variants")
            removed_or_kept = int(group.pos.isin(tags).sum())
            rng = _document_rng(random_seed, str(doc_id))
            chosen = set(rng.sample(range(len(group)), k=removed_or_kept))
            mask = [i in chosen for i in range(len(group))]
            selected = group.loc[mask] if random_only else group.loc[[not x for x in mask]]
        elif source.startswith("ONLY_"):
            selected = group.loc[group.pos.isin(tags)]
        else:
            selected = group.loc[~group.pos.isin(tags)]
        text_tokens = selected.pos.tolist() if variant == "POS_SEQUENCE" else selected.token.tolist()
        if variant == "ALL" and "whitespace" in selected:
            rendered = "".join(t + w for t, w in zip(selected.token, selected.whitespace)).strip()
        else:
            rendered = " ".join(text_tokens)
        first = group.iloc[0]
        records.append({"document_id": doc_id, "participant_id": first.participant_id,
                        "label": int(first.label), "split": first.split,
                        "text": rendered if text_tokens else "[UNK]", "selected_pos": " ".join(selected.pos.tolist()),
                        "n_tokens": len(text_tokens), "was_empty": len(text_tokens) == 0,
                        "variant": variant, "random_variant_seed": random_seed})
    return pd.DataFrame(records)
