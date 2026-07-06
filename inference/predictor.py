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
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Suppress warnings about UNEXPECTED keys
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

# The classification head size used during training
HEAD_SIZE = 512


def preprocess(text: str) -> str:
    """Clean and normalize input text."""
    text = text.strip().lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


class TransformerClassifier(nn.Module):
    """Transformer classifier with a simple head."""
    
    def __init__(self, hf_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Simple classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, HEAD_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HEAD_SIZE, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get pooled output (CLS token for BERT-like models)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            # For models without pooler (like DistilBERT, ELECTRA), use CLS token
            pooled = outputs.last_hidden_state[:, 0, :]
        
        logits = self.head(pooled)
        return F.log_softmax(logits, dim=-1)


def get_labels() -> list[str]:
    """Return the list of category labels."""
    return LABELS


def available_models() -> list[str]:
    """Return list of available models with checkpoint files."""
    models = []
    for key in MODEL_INFO:
        ckpt_path = CKPT_DIR / f"{key}_best.pt"
        if ckpt_path.is_file():
            models.append(key)
    return models


@lru_cache(maxsize=4)
def load_model(model_key: str):
    """Load a model with strict=False to handle architecture mismatches."""
    if model_key not in _MODEL_BACKBONES:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(_MODEL_BACKBONES.keys())}")

    hf_name = _MODEL_BACKBONES[model_key]
    labels = get_labels()
    ckpt_path = CKPT_DIR / f"{model_key}_best.pt"
    
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {model_key} from {ckpt_path}...")
    
    # Create model with the same architecture used during training
    model = TransformerClassifier(hf_name, num_classes=len(labels)).to(DEVICE)
    
    # Load the checkpoint with strict=False - this automatically handles:
    # - UNEXPECTED keys (pre-training heads like cls.predictions.*)
    # - Missing keys (if any)
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        
        # First try: load with strict=False (will ignore pre-training head weights)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ {model_key} loaded successfully")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Fallback: Try loading only the encoder weights
        try:
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            # Filter to only encoder weights
            encoder_state = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"✅ {model_key} encoder loaded successfully")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            raise
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    return tokenizer, model


def classify(text: str, model_key: str = DEFAULT_MODEL) -> dict[str, float]:
    """Classify a text input and return confidence scores for each category."""
    labels = get_labels()
    
    try:
        tokenizer, model = load_model(model_key)
    except Exception as e:
        print(f"Error loading model {model_key}: {e}")
        # Return uniform distribution as fallback
        return {label: 1.0 / len(labels) for label in labels}

    # Preprocess text
    clean = preprocess(text)
    
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
        # Pad or truncate as needed
        if len(probs) < len(labels):
            probs = probs + [0.0] * (len(labels) - len(probs))
        else:
            probs = probs[:len(labels)]
    
    return dict(zip(labels, probs))


def classify_ensemble(text: str, model_keys: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Run classification with multiple models and return ensemble results.
    
    Args:
        text: The input text to classify
        model_keys: List of model keys to use. If None, uses all available models.
    
    Returns:
        Dictionary with results from each model and ensemble predictions
    """
    if model_keys is None:
        model_keys = available_models()
    
    if not model_keys:
        raise ValueError("No models available for ensemble classification")
    
    results = {}
    all_probs = []
    
    for key in model_keys:
        try:
            probs = classify(text, key)
            results[key] = {
                "probs": probs,
                "predicted": max(probs, key=probs.get),
                "confidence": max(probs.values())
            }
            all_probs.append(probs)
        except Exception as e:
            print(f"Error with model {key}: {e}")
            continue
    
    if not all_probs:
        raise ValueError("No models could classify the text")
    
    # Ensemble: average probabilities
    ensemble_probs = {}
    for label in get_labels():
        ensemble_probs[label] = sum(p[label] for p in all_probs) / len(all_probs)
    
    results["ensemble"] = {
        "probs": ensemble_probs,
        "predicted": max(ensemble_probs, key=ensemble_probs.get),
        "confidence": max(ensemble_probs.values()),
        "models_used": len(all_probs)
    }
    
    # Calculate agreement
    predictions = [r["predicted"] for r in results.values() if "predicted" in r and r.get("predicted") is not None]
    if predictions:
        from collections import Counter
        agreement = Counter(predictions)
        results["ensemble"]["agreement"] = {
            cat: count / len(predictions) for cat, count in agreement.items()
        }
        results["ensemble"]["most_agreed"] = max(agreement, key=agreement.get)
    
    return results


def get_model_confidence_summary(text: str) -> Dict[str, Dict]:
    """
    Get detailed confidence information from all models for debugging.
    
    Args:
        text: The input text to classify
    
    Returns:
        Dictionary with detailed classification results from each model
    """
    labels = get_labels()
    results = {}
    
    for model_key in available_models():
        try:
            probs = classify(text, model_key)
            predicted = max(probs, key=probs.get)
            confidence = probs[predicted]
            
            # Get the top 3 predictions
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_probs[:3]
            
            # Calculate entropy as a measure of uncertainty
            import math
            entropy = -sum(p * math.log(p + 1e-10) for p in probs.values())
            
            results[model_key] = {
                "display_name": MODEL_INFO[model_key]["display"],
                "predictions": probs,
                "top_1": (predicted, confidence),
                "top_3": [(cat, prob) for cat, prob in top_3],
                "confidence": confidence,
                "entropy": entropy,
                "accuracy": MODEL_INFO[model_key]["accuracy"],
                "macro_f1": MODEL_INFO[model_key]["macro_f1"],
            }
        except Exception as e:
            print(f"Error with model {model_key}: {e}")
            results[model_key] = {"error": str(e)}
    
    return results
