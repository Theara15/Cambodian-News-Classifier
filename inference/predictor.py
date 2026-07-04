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


def filter_state_dict(state_dict, model_state_dict, model_key: str):
    """Filter state dict to only include keys that match the model architecture."""
    filtered = {}
    skipped = []
    shape_mismatch = []
    
    # Keys that should be skipped for specific models
    skip_patterns = {
        "electra": [
            "discriminator_predictions",
            "electra.embeddings_project",
            "embeddings_project",
            "discriminator",
        ],
        "bert": [
            "cls.seq_relationship",
            "cls.predictions",
            "seq_relationship",
            "predictions",
            "cls.",
            "bert.embeddings.token_type_embeddings",  # BERT specific
        ],
        "roberta": [
            "lm_head",  # RoBERTa LM head
        ],
        "distilbert": [
            "vocab_transform",  # DistilBERT LM head
            "vocab_layer_norm",
            "vocab_projector",
        ],
    }
    
    # Also skip any keys that look like pre-training heads
    pretraining_patterns = [
        "predictions",
        "seq_relationship",
        "lm_head",
        "vocab_transform",
        "vocab_layer_norm",
        "vocab_projector",
        "cls.",
        "mlm",
        "nsp",
    ]
    
    for key, value in state_dict.items():
        # Check if this key should be skipped
        should_skip = False
        
        # Check model-specific patterns
        if model_key in skip_patterns:
            for pattern in skip_patterns[model_key]:
                if pattern in key:
                    should_skip = True
                    break
        
        # Check general pre-training patterns
        if not should_skip:
            for pattern in pretraining_patterns:
                if pattern in key:
                    should_skip = True
                    break
        
        if should_skip:
            skipped.append(key)
            continue
        
        # Check if key exists in model
        if key in model_state_dict:
            # Check shape match
            if value.shape == model_state_dict[key].shape:
                filtered[key] = value
            else:
                shape_mismatch.append(f"{key}: {value.shape} vs {model_state_dict[key].shape}")
                skipped.append(key)
        else:
            skipped.append(key)
    
    if skipped:
        print(f"Skipped {len(skipped)} keys for {model_key}")
        if len(skipped) <= 10:
            print(f"Skipped keys: {skipped}")
        else:
            print(f"Skipped keys: {skipped[:10]}... (and {len(skipped) - 10} more)")
    
    if shape_mismatch:
        print(f"Shape mismatches: {shape_mismatch}")
    
    return filtered


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
    
    # Load the checkpoint
    try:
        state_dict = torch.load(ckpt_path, map_location=DEVICE)
        
        # Get the model's state dict keys
        model_state_dict = model.state_dict()
        
        # Filter the state dict to only include matching keys
        filtered_state_dict = filter_state_dict(state_dict, model_state_dict, model_key)
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"✅ {model_key} loaded successfully")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try loading with strict=False as fallback
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
            print(f"✅ {model_key} loaded with strict=False")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            # Try loading just the encoder weights
            try:
                # Load only the encoder part
                state_dict = torch.load(ckpt_path, map_location=DEVICE)
                encoder_keys = [k for k in state_dict.keys() if not k.startswith("head.")]
                encoder_state = {k: state_dict[k] for k in encoder_keys}
                model.encoder.load_state_dict(encoder_state, strict=False)
                print(f"✅ {model_key} encoder loaded successfully")
            except Exception as e3:
                print(f"Encoder-only loading also failed: {e3}")
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
