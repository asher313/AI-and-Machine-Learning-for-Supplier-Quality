# src/sqm_ai/dl/ncr_finetune.py
"""Train an NCR classifier with separate development and final-test data."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
CATEGORIES = ["cosmetic", "dimensional", "material", "functional"]


class NCRTextDataset(torch.utils.data.Dataset):
    """Tokenized descriptions plus an integer label."""
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = list(labels)
        if not self.labels or any(len(v) != len(self.labels) for v in enc.values()):
            raise ValueError("nonempty aligned token rows and labels required")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def run(
    train_texts, train_y, val_texts, val_y, test_texts, test_y,
    *, task="category", output="artifacts/ncr_category",
    model_name=MODEL_NAME, revision=None, epochs=3, use_cpu=False,
):
    """Caller supplies chronologically disjoint fit/validation/test sets.

    Category IDs are 0..3 in CATEGORIES order. Severity IDs are 0..4
    (original severity minus one); they remain human-review suggestions.
    """
    if task not in {"category", "severity"}:
        raise ValueError("task must be category or severity")
    names = CATEGORIES if task == "category" else [str(i) for i in range(1, 6)]
    groups = [(train_texts, train_y), (val_texts, val_y), (test_texts, test_y)]
    for texts, labels in groups:
        if not len(texts) or len(texts) != len(labels):
            raise ValueError("nonempty aligned texts and labels required")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("descriptions must be nonempty strings")
        if any(not isinstance(y, (int, np.integer)) or not 0 <= y < len(names) for y in labels):
            raise ValueError("label ID outside selected task")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory")
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={v: i for i, v in enumerate(names)},
    )
    datasets = [
        NCRTextDataset(tok(list(texts), padding="max_length", truncation=True, max_length=256), labels)
        for texts, labels in groups
    ]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"f1": f1_score(labels, np.argmax(logits, axis=-1),
                               labels=list(range(len(names))), average="macro", zero_division=0)}

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        greater_is_better=True, save_total_limit=2,
        fp16=torch.cuda.is_available() and not use_cpu, use_cpu=use_cpu,
        report_to="none", seed=42, data_seed=42,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1],
        processing_class=tok, compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(output / "final"))
    tok.save_pretrained(output / "final")
    metrics = trainer.evaluate(datasets[2], metric_key_prefix="test")
    record = {"task": task, "label_names": names, "model_name": str(model_name),
              "requested_revision": revision, "resolved_revision": getattr(model.config, "_commit_hash", None),
              "best_validation_checkpoint": trainer.state.best_model_checkpoint,
              "train_rows": len(datasets[0]), "validation_rows": len(datasets[1]),
              "test_rows": len(datasets[2]), "metrics": metrics}
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return metrics
