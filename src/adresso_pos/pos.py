from __future__ import annotations

from pathlib import Path

import pandas as pd
import spacy

UNIVERSAL_POS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
                 "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")


def tag_documents(documents: pd.DataFrame, model: str, batch_size: int = 32,
                  n_process: int = 1) -> pd.DataFrame:
    try:
        nlp = spacy.load(model, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(f"spaCy model {model!r} is required. Run: python -m spacy download {model}") from exc
    rows = []
    docs = nlp.pipe(documents.text.tolist(), batch_size=batch_size, n_process=n_process)
    for meta, doc in zip(documents.itertuples(index=False), docs):
        for token in doc:
            pos = token.pos_ if token.pos_ in UNIVERSAL_POS else "X"
            rows.append({"document_id": meta.document_id, "participant_id": meta.participant_id,
                         "label": int(meta.label), "split": meta.split, "token": token.text,
                         "whitespace": token.whitespace_, "pos": pos, "position": token.i, "is_space": token.is_space})
    return pd.DataFrame(rows)


def prepare_pos_cache(documents: pd.DataFrame, output: Path, pos_cfg: dict, force: bool = False) -> pd.DataFrame:
    if output.exists() and not force:
        return pd.read_parquet(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tagged = tag_documents(documents, pos_cfg["spacy_model"], pos_cfg.get("batch_size", 32), pos_cfg.get("n_process", 1))
    tagged.to_parquet(output, index=False)
    return tagged
