from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(documents: pd.DataFrame, n_splits: int, seed: int = 13) -> pd.DataFrame:
    train = documents.loc[documents.split == "train"].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(train, train.label, train.participant_id)):
        train_groups = set(train.iloc[train_idx].participant_id)
        val_groups = set(train.iloc[val_idx].participant_id)
        if train_groups & val_groups:
            raise AssertionError("Participant leakage in generated folds")
        rows.extend({"document_id": x, "fold": fold, "role": "train"} for x in train.iloc[train_idx].document_id)
        rows.extend({"document_id": x, "fold": fold, "role": "validation"} for x in train.iloc[val_idx].document_id)
    return pd.DataFrame(rows)
