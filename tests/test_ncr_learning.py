"""Offline checks use tiny random models, not downloaded or paid models."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from transformers import (
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)

from sqm_ai.dl.ncr_finetune import run
from sqm_ai.dl.ncr_zero_shot import classify


@pytest.mark.parametrize("task,count", [("category", 4), ("severity", 5)])
def test_trainer_has_separate_final_test(tmp_path, task, count):
    base = tmp_path / "base"
    base.mkdir()
    (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhole\nscratch\n")
    tok = BertTokenizerFast(vocab_file=str(base / "vocab.txt"), model_input_names=["input_ids", "attention_mask"])
    tok.save_pretrained(base)
    model = DistilBertForSequenceClassification(DistilBertConfig(vocab_size=len(tok), dim=16, hidden_dim=32, n_heads=2, n_layers=1, num_labels=count))
    model.save_pretrained(base)
    labels = list(range(count))
    metrics = run(["hole"] * (count * 2), labels * 2,
                  ["scratch"] * count, labels,
                  ["hole scratch"] * count, labels,
                  task=task, model_name=str(base), output=tmp_path / "run",
                  epochs=1, use_cpu=True)
    assert "test_f1" in metrics and "eval_f1" not in metrics
    assert (tmp_path / "run/final/tokenizer_config.json").exists()
    assert (tmp_path / "run/run.json").exists()


def test_prompted_baseline_rejects_invalid_or_incomplete_response():
    def api(text, reason="end_turn"):
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            stop_reason=reason, content=[SimpleNamespace(type="text", text=text)])))
    got = classify("hole offset", api=api('{"category":"dimensional","severity":3}'), model="test")
    assert got.severity == 3
    with pytest.raises(ValidationError):
        classify("hole offset", api=api('{"category":"dimensional","severity":8}'), model="test")
    with pytest.raises(ValueError, match="incomplete"):
        classify("hole offset", api=api('{}', "max_tokens"), model="test")
