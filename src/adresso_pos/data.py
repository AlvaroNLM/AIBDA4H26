from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TEXT_ALIASES = ("transcription", "transcript", "text", "utterance", "content", "sentence")
ID_ALIASES = ("participant_id", "participant", "subject_id", "subject", "speaker", "file", "filename", "id")
LABEL_ALIASES = ("label", "diagnosis", "class", "target", "group", "dx")


def _detect_column(df: pd.DataFrame, explicit: str | None, aliases: tuple[str, ...], kind: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Configured {kind} column {explicit!r} is absent: {list(df.columns)}")
        return explicit
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias)
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not detect {kind} column among {list(df.columns)}; configure it explicitly")


def _participant(value: object) -> str:
    # A filename is a document/participant ID in the released ADReSSo CSVs.
    return Path(str(value).strip()).stem


def _labels(series: pd.Series, positive_label: object) -> pd.Series:
    values = series.dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(f"Binary classification requires exactly two labels, found {values}")
    positive_text = str(positive_label).strip().lower()
    return series.map(lambda x: int(str(x).strip().lower() == positive_text)).astype(int)


def load_split(path: str | Path, cfg: dict, split: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        raw = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    dcfg = cfg["dataset"]
    text_col = _detect_column(raw, dcfg.get("text_column"), TEXT_ALIASES, "text")
    id_col = _detect_column(raw, dcfg.get("participant_column"), ID_ALIASES, "participant")
    label_col = _detect_column(raw, dcfg.get("label_column"), LABEL_ALIASES, "label")
    out = pd.DataFrame({
        "document_id": [f"{split}:{i}" for i in range(len(raw))],
        "participant_id": raw[id_col].map(_participant),
        "text": raw[text_col].fillna("").astype(str),
        "label": _labels(raw[label_col], dcfg.get("positive_label", 1)),
        "split": split,
        "source_file": str(path),
    })
    conflicting = out.groupby("participant_id")["label"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Some participants have conflicting labels")
    return out


def discover_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    root = Path(cfg["_project_root"], cfg["dataset"]["root"])
    train_name = cfg["dataset"].get("train_file")
    test_name = cfg["dataset"].get("test_file")
    candidates = [p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}]
    train_path = root / train_name if train_name else next((p for p in candidates if "train" in p.name.lower()), None)
    test_path = root / test_name if test_name else next((p for p in candidates if "test" in p.name.lower()), None)
    if not train_path or not train_path.exists():
        raise FileNotFoundError(f"No training table found below {root}")
    train = load_split(train_path, cfg, "train")
    test = load_split(test_path, cfg, "test") if test_path and test_path.exists() else None
    if test is not None and set(train.participant_id) & set(test.participant_id):
        raise ValueError("Participant leakage between official train and test")
    return train, test
