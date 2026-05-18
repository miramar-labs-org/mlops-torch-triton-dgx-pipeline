import os
import tempfile

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_model():
    from transformers import DistilBertForSequenceClassification, DistilBertConfig
    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        num_labels=2,
    )
    return DistilBertForSequenceClassification(config)


def _make_loader(batch_size=2, seq_len=8, num_batches=2):
    batches = []
    for _ in range(num_batches):
        batches.append({
            "input_ids":      torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "label":          torch.randint(0, 2, (batch_size,)),
        })
    return batches


# ---------------------------------------------------------------------------
# tokenize_batch
# ---------------------------------------------------------------------------

def test_tokenize_batch_output_shape():
    from transformers import DistilBertTokenizerFast
    from train import tokenize_batch, MAX_LEN

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    batch = {"text": ["I loved this film!", "Terrible movie."]}
    out = tokenize_batch(batch, tokenizer)

    assert "input_ids" in out
    assert "attention_mask" in out
    assert len(out["input_ids"]) == 2
    assert len(out["input_ids"][0]) == MAX_LEN


def test_tokenize_batch_truncates_long_input():
    from transformers import DistilBertTokenizerFast
    from train import tokenize_batch, MAX_LEN

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    long_text = "word " * 300
    out = tokenize_batch({"text": [long_text]}, tokenizer)
    assert len(out["input_ids"][0]) == MAX_LEN


def test_tokenize_batch_pads_short_input():
    from transformers import DistilBertTokenizerFast
    from train import tokenize_batch, MAX_LEN

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    out = tokenize_batch({"text": ["Hi"]}, tokenizer)
    assert len(out["input_ids"][0]) == MAX_LEN
    # padding token (0) fills the tail
    assert out["input_ids"][0][-1] == tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_float_between_0_and_1():
    from train import evaluate

    model = _make_model()
    device = torch.device("cpu")
    model.to(device)

    acc = evaluate(model, _make_loader(), device)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_evaluate_perfect_accuracy():
    from train import evaluate

    model = _make_model()
    device = torch.device("cpu")
    model.to(device)

    # Freeze all params and bias logit[1] strongly so argmax always == 1
    with torch.no_grad():
        model.classifier.bias.fill_(0)
        model.classifier.bias[1] = 100.0

    loader = [
        {
            "input_ids":      torch.zeros(4, 8, dtype=torch.long),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
            "label":          torch.ones(4, dtype=torch.long),
        }
    ]
    acc = evaluate(model, loader, device)
    assert acc == 1.0


def test_evaluate_model_set_to_eval_mode():
    from train import evaluate

    model = _make_model()
    device = torch.device("cpu")
    model.train()
    evaluate(model, _make_loader(), device)
    assert not model.training


# ---------------------------------------------------------------------------
# ONNX export (no GPU, no MLflow)
# ---------------------------------------------------------------------------

def test_onnx_export_creates_file():
    from train import MAX_LEN

    model = _make_model()
    device = torch.device("cpu")
    model.to(device).eval()

    dummy_ids  = torch.ones(1, MAX_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = os.path.join(tmp, "model.onnx")
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits":         {0: "batch_size"},
            },
            opset_version=14,
        )
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
