"""Inference pipeline for the Streamlit dashboard.

This version rebuilds the transformer architectures and prediction head directly
from the saved checkpoints, without requiring the missing training or
preprocessing packages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

APP_DIR = Path(__file__).resolve().parents[1]
CKPT_DIR = APP_DIR / "models" / "undersampling_no_environment"
MAX_LENGTH = 512
DEVICE = torch.device("cpu")

# Human-friendly names + test-set metrics for the undersampling_no_environment corpus.
MODEL_INFO: dict[str, dict] = {
    "roberta": {"display": "RoBERTa", "accuracy": 0.9175, "macro_f1": 0.9177},
    "distilbert": {"display": "DistilBERT", "accuracy": 0.9107, "macro_f1": 0.9111},
    "electra": {"display": "ELECTRA", "accuracy": 0.8746, "macro_f1": 0.8761},
    "bert": {"display": "BERT", "accuracy": 0.8418, "macro_f1": 0.8429},
}

DEFAULT_MODEL = "roberta"

LABELS: list[str] = ["economics", "health", "politics", "sports", "technology"]

_MODEL_BACKBONES: dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "roberta": "roberta-base",
}


def preprocess(text: str) -> str:
    text = text.strip().lower()
    text = " ".join(text.split())
    return text


class TransformerClassifier(nn.Module):
    def __init__(self, hf_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.head(pooled)
        return F.log_softmax(logits, dim=-1)


def get_labels() -> list[str]:
    return LABELS


def available_models() -> list[str]:
    return [key for key in MODEL_INFO if (CKPT_DIR / f"{key}_best.pt").is_file()]


@lru_cache(maxsize=4)
def load_model(model_key: str):
    if model_key not in _MODEL_BACKBONES:
        raise ValueError(f"Unknown model '{model_key}'.")

    hf_name = _MODEL_BACKBONES[model_key]
    labels = get_labels()

    model = TransformerClassifier(hf_name, num_classes=len(labels)).to(DEVICE)
    ckpt = CKPT_DIR / f"{model_key}_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer, model


def classify(text: str, model_key: str = DEFAULT_MODEL) -> dict[str, float]:
    labels = get_labels()
    tokenizer, model = load_model(model_key)

    clean = preprocess(text)
    enc = tokenizer(
        clean,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        log_probs = model(input_ids, attention_mask)
    probs = log_probs.exp().squeeze(0).tolist()
    return dict(zip(labels, probs))
