"""Inference pipeline for the Streamlit dashboard.

This version rebuilds the transformer architectures and prediction head directly
from the saved checkpoints, without requiring the missing training or
preprocessing packages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Suppress ALL warnings about UNEXPECTED keys
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

APP_DIR = Path(__file__).resolve().parents[1]
CKPT_DIR = APP_DIR / "models" / "undersampling_no_environment"
MAX_LENGTH = 512
DEVICE = torch.device("cpu")

# Human-friendly names + test-set metrics
# RoBERTa has the best academic performance (91.75% accuracy)
# DistilBERT often performs better on individual articles
MODEL_INFO: dict[str, dict] = {
    "distilbert": {"display": "DistilBERT", "accuracy": 0.9107, "macro_f1": 0.9111, "academic_rank": 2},
    "roberta": {"display": "RoBERTa 🏆", "accuracy": 0.9175, "macro_f1": 0.9177, "academic_rank": 1},
    "electra": {"display": "ELECTRA", "accuracy": 0.8746, "macro_f1": 0.8761, "academic_rank": 3},
    "bert": {"display": "BERT", "accuracy": 0.8418, "macro_f1": 0.8429, "academic_rank": 4},
}

# DistilBERT is the default because it performs better on your specific use case
DEFAULT_MODEL = "distilbert"

LABELS: list[str] = ["economics", "health", "politics", "sports", "technology"]

_MODEL_BACKBONES: dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "roberta": "roberta-base",
}

HEAD_SIZE = 512
MIN_WORDS_FOR_CONFIDENCE = 20
MIN_CHARS_FOR_CLASSIFICATION = 50


def preprocess(text: str) -> str:
    """Clean and normalize input text."""
    text = text.strip().lower()
    text = " ".join(text.split())
    return text


class TransformerClassifier(nn.Module):
    """Transformer classifier with a simple head."""
    
    def __init__(self, hf_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, HEAD_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HEAD_SIZE, num_classes),
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
    """Return list of available models with checkpoint files."""
    models = []
    for key in MODEL_INFO:
        ckpt_path = CKPT_DIR / f"{key}_best.pt"
        if ckpt_path.is_file():
            models.append(key)
    return models


def filter_state_dict(state_dict, model_state_dict):
    """Filter state dict to only include keys that match the model architecture."""
    filtered = {}
    skipped = []
    
    # Patterns to skip (pre-training heads that we don't need)
    skip_patterns = [
        "cls.predictions",
        "cls.seq_relationship",
        "predictions",
        "seq_relationship",
        "lm_head",
        "vocab_transform",
        "vocab_layer_norm",
        "vocab_projector",
        "mlm",
        "nsp",
        "discriminator_predictions",
        "embeddings_project",
    ]
    
    for key, value in state_dict.items():
        should_skip = False
        
        for pattern in skip_patterns:
            if pattern in key:
                should_skip = True
                break
        
        if should_skip:
            skipped.append(key)
            continue
        
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            filtered[key] = value
        else:
            skipped.append(key)
    
    if skipped:
        print(f"Skipped {len(skipped)} pre-training head keys")
    
    return filtered


@lru_cache(maxsize=4)
def load_model(model_key: str):
    """Load a model with strict=False to handle architecture mismatches."""
    if model_key not in _MODEL_BACKBONES:
        raise ValueError(f"Unknown model '{model_key}'")

    hf_name = _MODEL_BACKBONES[model_key]
    labels = get_labels()
    ckpt_path = CKPT_DIR / f"{model_key}_best.pt"
    
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {model_key} from {ckpt_path}...")
    
    model = TransformerClassifier(hf_name, num_classes=len(labels)).to(DEVICE)
    
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        model_state_dict = model.state_dict()
        filtered_state_dict = filter_state_dict(state_dict, model_state_dict)
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ {model_key} loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
            print(f"✅ {model_key} loaded with strict=False")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            raise
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    return tokenizer, model


def classify(text: str, model_key: str = DEFAULT_MODEL) -> dict[str, float]:
    """
    Classify a text input and return confidence scores for each category.
    Returns low confidence scores if text is too short or invalid.
    """
    labels = get_labels()
    
    # Check if text has enough content
    if len(text.strip()) < MIN_CHARS_FOR_CLASSIFICATION:
        print(f"⚠️ Text too short: {len(text.strip())} characters")
        # Return near-zero confidence across all categories
        return {label: 0.01 for label in labels}
    
    clean = preprocess(text)
    
    # Check word count
    word_count = len(clean.split())
    if word_count < MIN_WORDS_FOR_CONFIDENCE:
        print(f"⚠️ Too few words: {word_count} words")
        # Return low confidence across all categories
        return {label: 0.02 for label in labels}
    
    try:
        tokenizer, model = load_model(model_key)
    except Exception as e:
        print(f"Error loading model {model_key}: {e}")
        return {label: 1.0 / len(labels) for label in labels}

    # Tokenize
    enc = tokenizer(
        clean,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Run inference
    with torch.no_grad():
        log_probs = model(input_ids, attention_mask)
    
    # Convert log-probabilities to probabilities
    probs = log_probs.exp().squeeze(0).tolist()
    
    # Ensure we have the right number of probabilities
    if len(probs) != len(labels):
        print(f"Warning: Expected {len(labels)} probabilities, got {len(probs)}")
        if len(probs) < len(labels):
            probs = probs + [0.0] * (len(labels) - len(probs))
        else:
            probs = probs[:len(labels)]
    
    return dict(zip(labels, probs))


def classify_multiple(text: str, model_keys: list[str] = None) -> dict:
    """Run classification with multiple models and return all results."""
    if model_keys is None:
        model_keys = available_models()
    
    if not model_keys:
        raise ValueError("No models available for classification")
    
    results = {}
    
    for key in model_keys:
        try:
            probs = classify(text, key)
            results[key] = {
                "probs": probs,
                "predicted": max(probs, key=probs.get),
                "confidence": max(probs.values()),
                "display_name": MODEL_INFO[key]["display"],
                "accuracy": MODEL_INFO[key]["accuracy"],
                "macro_f1": MODEL_INFO[key]["macro_f1"],
                "academic_rank": MODEL_INFO[key].get("academic_rank", 99),
            }
        except Exception as e:
            print(f"Error with model {key}: {e}")
            results[key] = {"error": str(e)}
    
    return results
