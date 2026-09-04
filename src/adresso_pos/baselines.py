from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics
from .pos import UNIVERSAL_POS
from .train_bert import seed_everything

COUNT_POS = ["VERB", "PRON", "NOUN", "ADJ", "ADV", "AUX", "DET", "ADP"]


def pos_count_features(tokens: pd.DataFrame) -> pd.DataFrame:
    clean = tokens.loc[~tokens.is_space]
    counts = clean.groupby(["participant_id", "label", "split", "pos"]).size().unstack(fill_value=0)
    for pos in UNIVERSAL_POS:
        if pos not in counts: counts[pos] = 0
    total = counts[list(UNIVERSAL_POS)].sum(axis=1).clip(lower=1)
    out = pd.DataFrame(index=counts.index)
    for pos in COUNT_POS: out[f"freq_{pos}"] = counts[pos] / total
    out["verb_pron_ratio"] = counts.VERB / counts.PRON.clip(lower=1)
    out["pron_noun_ratio"] = counts.PRON / counts.NOUN.clip(lower=1)
    return out.reset_index()


def train_pos_counts(tokens: pd.DataFrame, train_ids: set[str], val_ids: set[str], seed: int):
    data = pos_count_features(tokens)
    features = [c for c in data if c.startswith("freq_") or c.endswith("_ratio")]
    tr = data[data.participant_id.isin(train_ids)]; va = data[data.participant_id.isin(val_ids)]
    model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed, class_weight=None))])
    model.fit(tr[features], tr.label)
    prob = model.predict_proba(va[features])[:, 1]
    pred = va[["participant_id", "label"]].copy(); pred["document_id"] = pred.participant_id
    pred["probability"] = prob; pred["predicted_class"] = (prob >= .5).astype(int); pred["evaluation_split"] = "validation"
    return classification_metrics(pred.label, prob), pred, features


class SequenceDataset(Dataset):
    def __init__(self, frame, vocab):
        self.rows = [(torch.tensor([vocab.get(x, vocab["X"]) for x in text.split()], dtype=torch.long), int(label))
                     for text, label in zip(frame.text, frame.label)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def _collate(batch):
    seqs, labels = zip(*batch); lengths = torch.tensor([len(x) for x in seqs])
    return nn.utils.rnn.pad_sequence(seqs, batch_first=True), lengths, torch.tensor(labels)


class PosLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden_dim, 2)
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(torch.cat([hidden[-2], hidden[-1]], dim=1))


def train_pos_sequence(train_df, val_df, cfg, seed):
    seed_everything(seed, cfg["model"].get("deterministic", True)); pc = cfg["pos_sequence"]
    vocab = {"<PAD>": 0, **{p: i + 1 for i, p in enumerate(UNIVERSAL_POS)}}
    train_loader = DataLoader(SequenceDataset(train_df, vocab), batch_size=pc["batch_size"], shuffle=True,
                              collate_fn=_collate, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(SequenceDataset(val_df, vocab), batch_size=pc["batch_size"], shuffle=False, collate_fn=_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosLSTM(len(vocab), pc["embedding_dim"], pc["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(pc["learning_rate"]), weight_decay=cfg["model"]["weight_decay"])
    best, best_state, bad, best_epoch = -1., None, 0, 0
    for epoch in range(1, pc["epochs"] + 1):
        model.train()
        for x, lengths, labels in train_loader:
            optimizer.zero_grad(); loss = nn.CrossEntropyLoss()(model(x.to(device), lengths), labels.to(device)); loss.backward(); optimizer.step()
        metrics, _ = _sequence_predict(model, val_loader, val_df, device)
        if metrics["macro_f1"] > best:
            best, best_state, bad, best_epoch = metrics["macro_f1"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0, epoch
        else:
            bad += 1
            if bad >= pc["patience"]: break
    model.load_state_dict(best_state)
    metrics, predictions = _sequence_predict(model, val_loader, val_df, device); metrics["best_epoch"] = best_epoch
    return metrics, predictions


def _sequence_predict(model, loader, frame, device):
    model.eval(); probs = []
    with torch.no_grad():
        for x, lengths, _ in loader: probs.extend(torch.softmax(model(x.to(device), lengths), 1)[:, 1].cpu().tolist())
    pred = frame[["document_id", "participant_id", "label"]].copy(); pred["probability"] = probs
    pred = pred.groupby(["participant_id", "label"], as_index=False).agg(document_id=("document_id", lambda x: "|".join(x)), probability=("probability", "mean"))
    pred["predicted_class"] = (pred.probability >= .5).astype(int); pred["evaluation_split"] = "validation"
    return classification_metrics(pred.label, pred.probability), pred
