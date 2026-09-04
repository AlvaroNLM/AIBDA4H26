from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from .metrics import classification_metrics


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.enc = tokenizer(self.frame.text.tolist(), truncation=True, max_length=max_length)
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = int(self.frame.iloc[idx].label)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs); self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_bert(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame | None,
               cfg: dict, seed: int, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    mc = cfg["model"]; seed_everything(seed, mc.get("deterministic", True))
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], local_files_only=mc.get("local_files_only", False))
    model = BertForSequenceClassification.from_pretrained(mc["name"], num_labels=2,
                                                          local_files_only=mc.get("local_files_only", False))
    train_ds = TextDataset(train_df, tokenizer, mc["max_length"])
    val_ds = TextDataset(val_df, tokenizer, mc["max_length"])
    counts = train_df.label.value_counts().reindex([0, 1], fill_value=0).to_numpy()
    ratio = counts.max() / counts.min()
    class_weights = None
    if ratio >= mc.get("class_weighting_threshold", float("inf")):
        class_weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"), learning_rate=float(mc["learning_rate"]),
        per_device_train_batch_size=mc["batch_size"], per_device_eval_batch_size=mc["batch_size"],
        num_train_epochs=mc["epochs"], weight_decay=mc["weight_decay"], warmup_ratio=mc["warmup_ratio"],
        max_steps=mc.get("max_steps", -1),
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        report_to="none", seed=seed, data_seed=seed)
    def compute(eval_pred):
        logits, labels = eval_pred
        prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        return classification_metrics(labels, prob)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                              processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                              compute_metrics=compute, class_weights=class_weights,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=mc["early_stopping_patience"])])
    trainer.train()
    best_epoch = int(round(float(trainer.state.epoch or 0)))
    all_predictions = []
    validation_metrics = {}
    for split, frame in (("validation", val_df), ("test", test_df)):
        if frame is None or frame.empty: continue
        pred = trainer.predict(TextDataset(frame, tokenizer, mc["max_length"]))
        prob = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:, 1]
        result = frame[["document_id", "participant_id", "label"]].copy()
        result["probability"] = prob; result["predicted_class"] = (prob >= .5).astype(int); result["evaluation_split"] = split
        result = (result.groupby(["participant_id", "label", "evaluation_split"], as_index=False)
                  .agg(document_id=("document_id", lambda x: "|".join(map(str, x))), probability=("probability", "mean")))
        result["predicted_class"] = (result.probability >= .5).astype(int)
        all_predictions.append(result)
        if split == "validation": validation_metrics = classification_metrics(result.label, prob)
    validation_metrics.update({"best_epoch": best_epoch, "class_weighting_used": class_weights is not None,
                               "class_weights": class_weights.tolist() if class_weights is not None else None})
    return validation_metrics, pd.concat(all_predictions, ignore_index=True)
