import json

import pandas as pd

from adresso_pos.metrics import save_run_outputs
from adresso_pos.splits import make_folds
from adresso_pos.variants import build_variant


HYPOTHESES = [{"name": "verb", "include_pos": ["VERB"]}, {"name": "verb_pron", "include_pos": ["VERB", "PRON"]}]


def tokens():
    rows = []
    specs = [
        ("d1", "p1", 0, [("he", "PRON"), ("runs", "VERB"), ("fast", "ADV")]),
        ("d2", "p2", 1, [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB"), (".", "PUNCT")]),
        ("d3", "p3", 0, [("I", "PRON"), ("see", "VERB")]),
        ("d4", "p4", 1, [("nice", "ADJ"), ("cat", "NOUN")]),
    ]
    for doc, participant, label, values in specs:
        for position, (token, pos) in enumerate(values):
            rows.append({"document_id": doc, "participant_id": participant, "label": label, "split": "train",
                         "token": token, "pos": pos, "position": position, "is_space": False})
    return pd.DataFrame(rows)


def test_only_and_without_are_correct():
    source = tokens()
    only = build_variant(source, "ONLY_VERB_PRON", HYPOTHESES)
    without = build_variant(source, "WITHOUT_VERB_PRON", HYPOTHESES)
    assert only.set_index("document_id").loc["d1", "text"] == "he runs"
    assert without.set_index("document_id").loc["d1", "text"] == "fast"
    assert only.set_index("document_id").loc["d4", "text"] == "[UNK]"
    assert only.set_index("document_id").loc["d4", "was_empty"]


def test_random_only_matches_token_count_and_is_reproducible():
    source = tokens()
    real = build_variant(source, "ONLY_VERB", HYPOTHESES).set_index("document_id")
    a = build_variant(source, "RANDOM_ONLY_MATCHED_VERB", HYPOTHESES, 42).set_index("document_id")
    b = build_variant(source, "RANDOM_ONLY_MATCHED_VERB", HYPOTHESES, 42).set_index("document_id")
    assert a.n_tokens.to_dict() == real.n_tokens.to_dict()
    pd.testing.assert_frame_equal(a, b)


def test_random_remove_matches_removed_count():
    source = tokens(); totals = source.groupby("document_id").size()
    real = build_variant(source, "WITHOUT_VERB", HYPOTHESES).set_index("document_id")
    random = build_variant(source, "RANDOM_REMOVE_MATCHED_VERB", HYPOTHESES, 7).set_index("document_id")
    assert (totals - real.n_tokens).to_dict() == (totals - random.n_tokens).to_dict()


def test_group_folds_have_no_leakage():
    docs = pd.DataFrame({"document_id": [f"d{i}" for i in range(12)],
                         "participant_id": [f"p{i // 2}" for i in range(12)],
                         "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], "split": "train"})
    folds = make_folds(docs, 3)
    joined = folds.merge(docs, on="document_id")
    for _, group in joined.groupby("fold"):
        train = set(group.loc[group.role == "train", "participant_id"])
        validation = set(group.loc[group.role == "validation", "participant_id"])
        assert not train & validation


def test_metrics_and_predictions_are_persisted(tmp_path):
    predictions = pd.DataFrame({"participant_id": ["p1"], "document_id": ["d1"], "label": [0], "probability": [.2], "predicted_class": [0]})
    save_run_outputs(tmp_path, {"macro_f1": 1.0}, predictions, {"experiment": {"variant": "ALL"}})
    assert json.loads((tmp_path / "metrics.json").read_text())["macro_f1"] == 1.0
    assert (tmp_path / "predictions.csv").exists() and (tmp_path / "predictions.parquet").exists() and (tmp_path / "COMPLETED").exists()
