# sqm_ai/dl/ncr_finetune.py
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
CATEGORIES = [
    "cosmetic", "dimensional", "material", "functional",
]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="macro"),
    }


def build(num_labels: int):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.\
        from_pretrained(MODEL_NAME, num_labels=num_labels)
    return tok, model


def tokenize(tok, texts: list[str]):
    return tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
    )


args = TrainingArguments(
    output_dir="artifacts/ncr_category",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none",
)


class NCRTextDataset(torch.utils.data.Dataset):
    """Tokenized descriptions plus an integer label."""

    def __init__(self, enc, labels: list[int]):
        self.enc = enc
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {
            k: torch.tensor(v[i])
            for k, v in self.enc.items()
        }
        item["labels"] = torch.tensor(self.labels[i])
        return item


def run(train_texts, train_y, val_texts, val_y) -> dict:
    """Time-ordered split in, fine-tuned model out."""
    tok, model = build(len(CATEGORIES))
    train_ds = NCRTextDataset(
        tokenize(tok, train_texts), train_y
    )
    val_ds = NCRTextDataset(
        tokenize(tok, val_texts), val_y
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("artifacts/ncr_category")
    return trainer.evaluate()
