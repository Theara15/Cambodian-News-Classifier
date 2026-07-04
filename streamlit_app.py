"""Cambodian News Classifier - Streamlit dashboard.

Run from the project root:

    streamlit run streamlit_app.py

Serves the fine-tuned encoders (BERT, DistilBERT, RoBERTa, ELECTRA) behind a 
single classifier UI with session history and an about/model-card page.
"""

from __future__ import annotations

import io
import json
import gc
from datetime import datetime

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Lazy imports with error handling
# --------------------------------------------------------------------------- #
def get_predictor():
    """Lazy load the predictor module with error handling."""
    try:
        from inference.predictor import (
            DEFAULT_MODEL,
            MODEL_INFO,
            available_models,
            classify,
            get_labels,
        )
        return DEFAULT_MODEL, MODEL_INFO, available_models, classify, get_labels
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        # Return fallback values
        return "roberta", {}, lambda: [], lambda x, y: {}, lambda: []


# --------------------------------------------------------------------------- #
# Page config + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cambodian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Per-category accent colours
CATEGORY_COLORS = {
    "politics": "#7c3aed",
    "technology": "#10b981",
    "economics": "#3b82f6",
    "health": "#0ea5e9",
    "sports": "#f59e0b",
}
DEFAULT_COLOR = "#64748b"

MIN_WORDS = 50

CSS = """
<style>
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    body, .stApp, .main {background: #f0f2f6 !important; color: #111827 !important;}
    .block-container {
        background: #ffffff !important;
        padding: 1.5rem 2rem !important;
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
        max-width: 1200px;
        margin: 1rem auto !important;
    }
    
    .app-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d4a7a 100%);
        border-radius: 18px;
        padding: 20px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 16px 40px rgba(26, 54, 93, 0.16);
    }
    .brand {display: flex; align-items: center; gap: 14px;}
    .brand-logo {
        width: 46px; height: 46px; border-radius: 14px;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        display:flex; align-items:center; justify-content:center;
        font-size: 22px; color:white; font-weight:800;
    }
    .brand-title {color:white; font-size:22px; font-weight:800; line-height:1.05;}
    .brand-sub {color:rgba(255,255,255,0.8); font-size:11px; letter-spacing:1.5px; font-weight:600;}

    .card {
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:24px 28px;
        box-shadow:0 18px 40px rgba(15, 23, 42, 0.06);
        transition: all 0.2s ease;
    }
    .card:hover {
        border-color: #d1d5db;
    }
    .card-title {font-size:22px; font-weight:800; color:#111827; margin:0;}
    .card-sub {font-size:14px; color:#64748b; margin-top:8px; line-height:1.5;}

    .result-head {
        background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:24px 28px;
        margin-bottom:18px;
        position: relative;
        overflow: hidden;
    }
    .result-head::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
    }
    .result-kicker {font-size:11px; letter-spacing:1.5px; color:#64748b; font-weight:700;}
    .result-cat {font-size:32px; font-weight:800; margin:10px 0 0 0; text-transform:uppercase; color:#111827;}
    .conf-pill {
        float:right; background:#2563eb; color:white; font-weight:700;
        font-size:12px; padding:7px 14px; border-radius:999px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    .stat-box {
        background:#f8fafc;
        border:1px solid #e5e7eb;
        border-radius:16px;
        padding:18px;
        text-align:center;
        transition: all 0.2s ease;
    }
    .stat-box:hover {
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    .stat-num {font-size:28px; font-weight:800; color:#111827;}
    .stat-lab {font-size:12px; color:#64748b; margin-top:4px;}

    .bar-row {display:flex; align-items:center; margin:10px 0; font-size:13px;}
    .bar-name {width:110px; color:#111827; text-transform:capitalize; font-weight:500;}
    .bar-track {flex:1; background:#f1f5f9; border-radius:6px; height:10px; overflow:hidden; margin:0 12px;}
    .bar-fill {height:100%; border-radius:6px; transition: width 0.6s ease;}
    .bar-val {width:54px; text-align:right; font-weight:700; color:#111827;}

    .ok-note {color:#16a34a; font-size:13px; font-weight:600;}
    .warn-note {color:#d97706; font-size:13px; font-weight:600;}

    .badge {
        display:inline-block; padding:4px 12px; border-radius:999px;
        font-size:11px; font-weight:700; color:white; text-transform:capitalize;
    }

    .stTextArea textarea {
        background: #f9fafb !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        color: #111827 !important;
        min-height: 260px;
        font-size: 14px !important;
        line-height: 1.6 !important;
        transition: all 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1) !important;
    }
    .stTextArea textarea::placeholder {color:#94a3b8 !important;}

    .input-hint {color: #64748b; font-size:12px; margin-top:6px;}

    div.stButton > button {
        border-radius:12px; 
        font-weight:700; 
        background: #2563eb; 
        color:white; 
        border:none;
        padding: 10px 24px !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #1e40af;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    div.stButton > button[disabled] {
        background: #93c5fd; 
        color: #ffffff;
        box-shadow: none;
        transform: none;
    }

    .stFileUploader label {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .stFileUploader > div {
        color: #111827 !important;
    }
    .stFileUploader button {
        color: white !important;
        background: #2563eb !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }
    .stFileUploader button:hover {
        background: #1e40af !important;
    }

    .stSelectbox label {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .stSelectbox > div {
        background: white !important;
        border-radius: 10px !important;
        border: 1px solid #d1d5db !important;
    }

    [data-testid="metric-container"] {
        background: #f8fafc;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.06);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] div {
        color: #111827 !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f4f9;
        border-radius: 12px;
        padding: 4px;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
        color: #64748b;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1a365d !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .history-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 16px 0;
    }
    .history-stat-item {
        background: #f8fafc;
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    .history-stat-item:hover {
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .history-stat-item .number {
        font-size: 28px;
        font-weight: 800;
        color: #111827;
    }
    .history-stat-item .label {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    .history-item {
        background: white;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 12px;
        border: 1px solid #e5e7eb;
        transition: all 0.25s ease;
        position: relative;
    }
    .history-item:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
        transform: translateX(4px);
    }
    .history-item .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .history-item .badge-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .history-item .meta-info {
        color: #64748b;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    }
    .history-item .meta-info span {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .history-item .preview {
        margin-top: 12px;
        color: #374151;
        font-size: 14px;
        line-height: 1.7;
        padding: 12px 16px;
        background: #f8fafc;
        border-radius: 10px;
        border-left: 4px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    .history-item:hover .preview {
        border-left-color: #2563eb;
    }
    .history-item .confidence-bar-mini {
        display: inline-block;
        height: 4px;
        width: 60px;
        background: #f1f5f9;
        border-radius: 4px;
        overflow: hidden;
        vertical-align: middle;
        margin-left: 6px;
    }
    .history-item .confidence-bar-mini .fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    .filter-section {
        display: flex;
        gap: 12px;
        margin: 16px 0 20px 0;
        flex-wrap: wrap;
        align-items: center;
        background: #f8fafc;
        padding: 12px 16px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
    }
    .filter-section .search-box {
        flex: 1;
        min-width: 200px;
    }
    .filter-section .category-filter {
        min-width: 150px;
    }
    .filter-section .sort-filter {
        min-width: 150px;
    }

    .about-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 16px 0;
    }
    .about-grid-item {
        background: #f8fafc;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: all 0.2s ease;
    }
    .about-grid-item:hover {
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        transform: translateY(-2px);
    }
    .about-grid-item .icon {
        font-size: 32px;
        margin-bottom: 6px;
    }
    .about-grid-item .label {
        font-weight: 700;
        color: #111827;
        font-size: 14px;
    }
    .about-grid-item .desc {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    .about-section {
        margin: 20px 0 12px 0;
        padding: 16px 20px;
        background: #f8fafc;
        border-radius: 14px;
        border-left: 4px solid #2563eb;
        transition: all 0.2s ease;
    }
    .about-section:hover {
        border-left-color: #7c3aed;
        background: #f1f4f9;
    }
    .about-section .title {
        font-weight: 700;
        color: #111827;
        font-size: 15px;
    }
    .about-section .content {
        color: #475569;
        font-size: 14px;
        margin-top: 6px;
        line-height: 1.7;
    }
    .about-section .content code {
        background: #e5e7eb;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        color: #1e3a8a;
    }

    .category-chip {
        border-radius: 14px;
        padding: 14px 12px;
        text-align: center;
        transition: all 0.2s ease;
        cursor: default;
    }
    .category-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .category-chip .letter {
        font-size: 24px;
        font-weight: 800;
        text-transform: uppercase;
    }
    .category-chip .name {
        font-size: 12px;
        font-weight: 600;
        color: #111827;
        margin-top: 4px;
    }

    .placeholder-card {
        text-align:center; 
        color:#111827;
        padding: 30px 20px;
    }
    .placeholder-card .icon {
        font-size: 56px;
        margin-bottom: 12px;
    }
    .placeholder-card .headline {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
    }
    .placeholder-card .desc {
        color: #64748b;
        font-size: 14px;
        max-width: 440px;
        margin: 8px auto 0 auto;
        line-height: 1.6;
    }

    .feature-list {list-style:none; padding-left:0; margin:18px 0 0 0; color:#475569;}
    .feature-list li {margin:10px 0; display:flex; align-items:flex-start; gap:10px; color:#475569;}
    .feature-list li::before {content:'✓'; color:#16a34a; font-weight:700;}

    @media (max-width: 768px) {
        .history-stats {
            grid-template-columns: 1fr 1fr;
        }
        .about-grid {
            grid-template-columns: 1fr 1fr;
        }
        .block-container {
            padding: 1rem !important;
        }
        .result-cat {
            font-size: 24px;
        }
        .conf-pill {
            float: none;
            display: inline-block;
            margin-top: 8px;
        }
        .history-item .header {
            flex-direction: column;
            align-items: flex-start;
        }
        .history-item .meta-info {
            flex-wrap: wrap;
        }
        .filter-section {
            flex-direction: column;
        }
        .filter-section .search-box,
        .filter-section .category-filter,
        .filter-section .sort-filter {
            width: 100%;
        }
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("page", "Classifier")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("model_key", None)  # Will be set after loading
    st.session_state.setdefault("models_loaded", False)
    st.session_state.setdefault("available_models_list", [])
    st.session_state.setdefault("model_info_dict", {})


_init_state()


def _color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


def html_block(html: str) -> None:
    cleaned = "".join(line.strip() for line in html.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model loader with error handling and caching
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    """Load models with error handling and caching."""
    try:
        DEFAULT_MODEL, MODEL_INFO, available_models_fn, classify_fn, get_labels_fn = get_predictor()
        
        # Get available models
        try:
            models = available_models_fn()
        except Exception as e:
            print(f"Error loading available models: {e}")
            models = []
        
        # Filter out models that fail to load
        valid_models = []
        for model in models:
            try:
                # Test if model can load
                classify_fn("test", model)
                valid_models.append(model)
            except Exception as e:
                print(f"Model {model} failed to load: {e}")
                continue
        
        # If no models load, try default
        if not valid_models and DEFAULT_MODEL:
            try:
                classify_fn("test", DEFAULT_MODEL)
                valid_models = [DEFAULT_MODEL]
            except Exception as e:
                print(f"Default model {DEFAULT_MODEL} failed to load: {e}")
                # Try BERT as last resort
                try:
                    classify_fn("test", "bert")
                    valid_models = ["bert"]
                except:
                    valid_models = []
        
        return {
            "models": valid_models,
            "model_info": MODEL_INFO,
            "default_model": valid_models[0] if valid_models else None,
            "classify_fn": classify_fn,
            "get_labels_fn": get_labels_fn,
            "loaded": True
        }
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return {
            "models": [],
            "model_info": {},
            "default_model": None,
            "classify_fn": lambda x, y: {},
            "get_labels_fn": lambda: [],
            "loaded": False
        }


def get_classify_function():
    """Get the classify function."""
    if not st.session_state.models_loaded:
        result = load_models()
        st.session_state.models_loaded = result["loaded"]
        st.session_state.available_models_list = result["models"]
        st.session_state.model_info_dict = result["model_info"]
        st.session_state.model_key = result["default_model"]
        st.session_state._classify_fn = result["classify_fn"]
        st.session_state._get_labels_fn = result["get_labels_fn"]
    return st.session_state._classify_fn


def get_labels_fn():
    """Get the labels function."""
    if not st.session_state.models_loaded:
        get_classify_function()
    return st.session_state._get_labels_fn


def get_available_models():
    """Get available models."""
    if not st.session_state.models_loaded:
        get_classify_function()
    return st.session_state.available_models_list


def get_model_info():
    """Get model info."""
    if not st.session_state.models_loaded:
        get_classify_function()
    return st.session_state.model_info_dict


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    html_block(
        """
        <div class="app-header">
          <div class="brand">
            <div class="brand-logo">📰</div>
            <div>
              <div class="brand-title">Cambodian News Classifier</div>
              <div class="brand-sub">Multi-class news article categorization</div>
            </div>
          </div>
        </div>
        """
    )


# --------------------------------------------------------------------------- #
# Confidence bars
# --------------------------------------------------------------------------- #
def render_scores(scores: dict[str, float]) -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = [
        '<div style="font-weight:700;color:#111827;margin:8px 0 6px;font-size:15px;">'
        "📊 Confidence Scores</div>"
    ]
    for cat, prob in ordered:
        pct = prob * 100
        color = _color(cat)
        parts.append(
            '<div class="bar-row">'
            f'<div class="bar-name">{cat}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
            "</div>"
            f'<div class="bar-val" style="color:{color};">{pct:.1f}%</div>'
            "</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    # Load models info
    MODEL_INFO = get_model_info()
    models = get_available_models()
    labels = get_labels_fn()()
    
    if not models:
        st.error(
            "⚠️ No models available. Please check that the fine-tuned weights "
            "are in the `models/undersampling_no_environment/` directory."
        )
        st.stop()
    
    left, right = st.columns([1, 1], gap="large")

    with left:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())
        
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin-bottom:4px;">
              <div>
                <div style="font-size:20px;font-weight:800;color:#111827;">📝 Input Section</div>
                <div style="font-size:14px;color:#64748b;margin-top:4px;">Paste or upload news text for category classification.</div>
              </div>
              <div style="color:#64748b;font-size:13px;background:#f8fafc;padding:6px 14px;border-radius:8px;border:1px solid #e5e7eb;">
                {chars:,} chars · {words:,} words
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_selector(MODEL_INFO, models)

        tab_text, tab_pdf = st.tabs(["📄 Direct Text Entry", "⬆ PDF Upload"])
        
        with tab_text:
            text = st.text_area(
                "Text input",
                value=st.session_state.input_text,
                height=320,
                label_visibility="collapsed",
                placeholder="Paste news text here (English). Perfect for copied articles or short texts.\n\nThe government announced new economic policies today...",
                key="text_area_input",
            )
            st.session_state.input_text = text
            st.markdown(
                '<div class="input-hint">💡 Perfect for copied articles or short texts.</div>',
                unsafe_allow_html=True,
            )

        with tab_pdf:
            st.markdown(
                '<div style="color:#111827;font-weight:600;margin-bottom:8px;">📄 Upload a PDF article</div>',
                unsafe_allow_html=True,
            )
            pdf = st.file_uploader("Upload a PDF article", type=["pdf"], label_visibility="collapsed")
            if pdf is not None:
                extracted = _read_pdf(pdf)
                if extracted:
                    st.session_state.input_text = extracted
                    text = extracted
                    st.success(f"✅ Extracted {len(extracted.split()):,} words from PDF.")
                    st.text_area("Extracted text", value=extracted, height=180)

        words = len(text.split())
        classify_fn = get_classify_function()
        analyze = st.button(
            "🔍 Analyze Text",
            use_container_width=True,
            disabled=words == 0,
        )
        if analyze:
            if words < MIN_WORDS:
                st.warning(
                    f"⚠️ Only {words} words detected. {MIN_WORDS}+ words give more "
                    "reliable results, but classifying anyway."
                )
            with st.spinner("🧠 Running inference…"):
                try:
                    scores = classify_fn(text, st.session_state.model_key)
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    scores = {cat: 0.0 for cat in labels}
                gc.collect()  # Force garbage collection
            
            top_cat = max(scores, key=scores.get)
            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_INFO.get(st.session_state.model_key, {}).get("display", st.session_state.model_key),
                "category": top_cat,
                "confidence": scores[top_cat],
                "scores": scores,
                "chars": len(text),
                "words": words,
                "preview": text.strip().replace("\n", " ")[:160],
            }
            st.session_state.last_result = result
            st.session_state.history.insert(0, result)

    with right:
        result = st.session_state.last_result
        if result is None:
            st.markdown(
                """
                <div class="card placeholder-card">
                    <div class="icon">🔎</div>
                    <div class="headline">Ready to classify your article</div>
                    <div class="desc">
                        Paste text or upload a PDF, then use the button below to see the predicted category and confidence scores.
                    </div>
                    <ul class="feature-list">
                        <li>Supports news text input</li>
                        <li>5-category classification model</li>
                        <li>Confidence scores for all categories</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        cat = result["category"]
        conf = result["confidence"] * 100
        html_block(
            f"""
            <div class="result-head">
              <span class="conf-pill">{conf:.1f}% confidence</span>
              <div class="result-kicker">🏆 TOP CLASSIFICATION</div>
              <div class="result-cat" style="color:{_color(cat)};">{cat}</div>
            </div>
            """
        )

        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="stat-box"><div class="stat-num">{result["chars"]:,}</div>'
            '<div class="stat-lab">Characters</div></div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-box"><div class="stat-num">{result["words"]:,}</div>'
            '<div class="stat-lab">Words</div></div>',
            unsafe_allow_html=True,
        )

        if result["words"] >= MIN_WORDS:
            st.markdown(
                '<p class="ok-note">✅ Text length is optimal for classification</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="warn-note">⚠️ Short text — prediction may be less reliable</p>',
                unsafe_allow_html=True,
            )

        render_scores(result["scores"])
        st.markdown(
            f'<div style="font-size:12px;color:#64748b;margin-top:8px;">🤖 Model: {result["model"]}</div>',
            unsafe_allow_html=True,
        )

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "📥 Export JSON",
                data=json.dumps(result, indent=2),
                file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with e2:
            if st.button("🗑 Clear Results", use_container_width=True):
                st.session_state.last_result = None
                st.session_state.input_text = ""
                st.rerun()


def _read_pdf(uploaded) -> str:
    try:
        import pdfplumber
    except ImportError:
        st.error("PDF support needs `pdfplumber`. Install with: pip install pdfplumber")
        return ""
    try:
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")
        return ""


# --------------------------------------------------------------------------- #
# Session history page
# --------------------------------------------------------------------------- #
def page_history() -> None:
    history = st.session_state.history
    
    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:4px;">
            <div>
                <div style="font-size:22px;font-weight:800;color:#111827;">📋 Session History</div>
                <div style="font-size:14px;color:#64748b;margin-top:4px;">
                    All classification results from this browser session
                </div>
            </div>
            <div style="font-size:12px;color:#94a3b8;background:#f8fafc;padding:6px 14px;border-radius:8px;border:1px solid #e5e7eb;">
                🔄 Resets on refresh
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.info("📭 No classifications yet. Analyze an article to populate the history.")
        return

    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)
    
    st.markdown(
        f"""
        <div class="history-stats">
            <div class="history-stat-item">
                <div class="number">{len(history)}</div>
                <div class="label">Total Articles</div>
            </div>
            <div class="history-stat-item">
                <div class="number">{len(set(cats))}/5</div>
                <div class="label">Categories Used</div>
            </div>
            <div class="history-stat-item">
                <div class="number">{(sum(confidences) / len(confidences)) * 100:.1f}%</div>
                <div class="label">Avg Confidence</div>
            </div>
            <div class="history-stat-item">
                <div class="number" style="color:{_color(top_cat)};">{top_cat.capitalize()}</div>
                <div class="label">Top Category</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    
    with filter_col1:
        query = st.text_input(
            "🔍 Search articles",
            placeholder="Filter by preview text…",
            label_visibility="collapsed",
        )
    
    with filter_col2:
        cat_filter = st.selectbox(
            "📂 Category",
            ["All"] + sorted(set(cats)),
            label_visibility="collapsed",
        )
    
    with filter_col3:
        sort_options = ["📅 Most Recent", "📅 Oldest First", "📈 Highest Confidence", "📉 Lowest Confidence"]
        sort_by = st.selectbox(
            "🔀 Sort",
            sort_options,
            index=0,
            label_visibility="collapsed",
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    rows = []
    for h in history:
        if query and query.lower() not in h["preview"].lower():
            continue
        if cat_filter != "All" and h["category"] != cat_filter:
            continue
        rows.append(h)
    
    if sort_by == "📅 Most Recent":
        rows = rows
    elif sort_by == "📅 Oldest First":
        rows = list(reversed(rows))
    elif sort_by == "📈 Highest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"], reverse=True)
    elif sort_by == "📉 Lowest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"])

    st.markdown(
        f"""
        <div style="margin:12px 0 16px 0;color:#64748b;font-size:13px;display:flex;align-items:center;gap:8px;">
            <span style="background:#2563eb;color:white;border-radius:999px;padding:0px 10px;font-weight:700;font-size:12px;">{len(rows)}</span>
            <span>article{'s' if len(rows) != 1 else ''} found</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for h in rows:
        color = _color(h["category"])
        conf_pct = h["confidence"] * 100
        
        if conf_pct >= 80:
            conf_color = "#16a34a"
            conf_emoji = "🟢"
            conf_label = "High"
        elif conf_pct >= 60:
            conf_color = "#f59e0b"
            conf_emoji = "🟡"
            conf_label = "Medium"
        else:
            conf_color = "#ef4444"
            conf_emoji = "🔴"
            conf_label = "Low"
        
        st.markdown(
            f"""
            <div class="history-item">
                <div class="header">
                    <div class="badge-group">
                        <span class="badge" style="background:{color};">{h['category']}</span>
                        <span style="font-size:14px;font-weight:700;color:#111827;">
                            <span class="confidence-emoji">{conf_emoji}</span> {conf_pct:.1f}%
                            <span style="font-weight:400;color:#64748b;font-size:11px;">({conf_label})</span>
                        </span>
                        <span class="confidence-bar-mini">
                            <span class="fill" style="width:{conf_pct:.1f}%;background:{conf_color};"></span>
                        </span>
                    </div>
                    <div class="meta-info">
                        <span>🤖 {h['model']}</span>
                        <span>📝 {h['words']:,} words</span>
                        <span class="timestamp">🕐 {h['timestamp']}</span>
                    </div>
                </div>
                <div class="preview">{h['preview']}…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    df = pd.DataFrame(
        [
            {
                "timestamp": h["timestamp"],
                "model": h["model"],
                "category": h["category"],
                "confidence": round(h["confidence"], 4),
                "words": h["words"],
                "chars": h["chars"],
                "preview": h["preview"],
            }
            for h in history
        ]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Export All (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #
def page_about() -> None:
    labels = get_labels_fn()()
    MODEL_INFO = get_model_info()
    models = get_available_models()
    
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:24px;">
            <div style="font-size:56px;line-height:1.2;">📰</div>
            <div style="font-size:26px;font-weight:800;color:#111827;margin-top:8px;">
                Cambodian News Classifier
            </div>
            <div style="font-size:14px;color:#64748b;margin-top:4px;">
                Thesis Project — Deployment Deliverable (Part 3)
            </div>
            <div style="width:60px;height:4px;background:linear-gradient(90deg,#2563eb,#7c3aed);margin:12px auto 0 auto;border-radius:4px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        f"""
        <div class="about-grid">
            <div class="about-grid-item">
                <div class="icon">📂</div>
                <div class="label">5 Categories</div>
                <div class="desc">Politics, Technology, Economics, Health, Sports</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">🤖</div>
                <div class="label">{len(models)} Models</div>
                <div class="desc">{', '.join([MODEL_INFO.get(m, {}).get('display', m) for m in models])}</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">🏆</div>
                <div class="label">Best Accuracy</div>
                <div class="desc">RoBERTa — 91.75%</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">📊</div>
                <div class="label">Balanced Dataset</div>
                <div class="desc">Undersampling · No Environment Class</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="background:linear-gradient(135deg, #f8fafc 0%, #f1f4f9 100%);border-radius:14px;padding:18px 22px;border:1px solid #e5e7eb;margin:12px 0;">
            <div style="color:#475569;font-size:14px;line-height:1.8;">
                This dashboard classifies English-language Cambodian news articles into one of
                <strong>five categories</strong> using transformer encoders fine-tuned on a custom corpus
                scraped from Cambodian news outlets. The <strong>Environment</strong> class was excluded
                from the corpus, leaving five balanced categories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:19px;font-weight:800;color:#111827;margin-top:28px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
            📊 Model Performance
            <span style="font-size:12px;font-weight:400;color:#64748b;">— test-set metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    model_data = []
    for key, info in MODEL_INFO.items():
        if key not in models:
            continue
        is_best = key == "roberta"
        model_data.append({
            "Model": info.get("display", key),
            "Accuracy": f"{info.get('accuracy', 0)*100:.2f}%",
            "Macro F1": f"{info.get('macro_f1', 0)*100:.2f}%",
            "Status": "✅ Available",
            "🏆": "⭐ Best" if is_best else ""
        })
    
    if model_data:
        df_models = pd.DataFrame(model_data)
        st.dataframe(
            df_models,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
                "Macro F1": st.column_config.TextColumn("Macro F1", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "🏆": st.column_config.TextColumn("", width="small"),
            }
        )
    
    st.markdown(
        """
        <div style="background:linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);border-radius:14px;padding:18px 22px;border:1px solid #93c5fd;margin:16px 0;">
            <div style="font-weight:700;color:#1e40af;font-size:16px;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                🏆 Model Recommendation
            </div>
            <div style="color:#1e3a8a;font-size:14px;line-height:1.8;">
                <strong>RoBERTa</strong> is best overall by report/test metrics — 
                highest accuracy (91.75%) and macro-F1 (91.77%) on the balanced dataset.
            </div>
            <div style="color:#1e3a8a;font-size:13px;line-height:1.7;margin-top:10px;padding-top:10px;border-top:1px solid #93c5fd;">
                💡 <strong>Note:</strong> <strong>DistilBERT</strong> can still look better on one input because 
                confidence varies sample by sample. The model with the best test-set metrics 
                may not always produce the highest confidence for every individual article.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:19px;font-weight:800;color:#111827;margin-top:28px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
            🔧 Pipeline
            <span style="font-size:12px;font-weight:400;color:#64748b;">— classification workflow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="about-section">
            <div class="title">1. Preprocess</div>
            <div class="content">
                Lowercase, strip HTML / URLs / emails / digits, drop stop-words 
                (identical to training, via <code>preprocessing.clean.preprocess</code>)
            </div>
        </div>
        <div class="about-section">
            <div class="title">2. Tokenize</div>
            <div class="content">
                Model-specific HuggingFace tokenizer, <code>max_length=512</code>, body text only
            </div>
        </div>
        <div class="about-section">
            <div class="title">3. Classify</div>
            <div class="content">
                <code>TransformerClassifier</code> ([CLS] → 512 → LogSoftmax over 5 classes)
            </div>
        </div>
        <div class="about-section">
            <div class="title">4. Report</div>
            <div class="content">
                <code>exp()</code> of log-probabilities gives the confidence scores shown in the dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:19px;font-weight:800;color:#111827;margin-top:28px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
            📂 Categories
            <span style="font-size:12px;font-weight:400;color:#64748b;">— 5 classes</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    cat_cols = st.columns(5)
    for idx, cat in enumerate(labels):
        color = _color(cat)
        with cat_cols[idx]:
            st.markdown(
                f"""
                <div class="category-chip" style="background:{color}15;border:2px solid {color};">
                    <div class="letter" style="color:{color};">{cat[0]}</div>
                    <div class="name">{cat.capitalize()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.caption(
        "The *no-environment* variant — Environment class was excluded from the corpus, leaving five balanced categories."
    )
    
    st.markdown(
        """
        <div style="font-size:19px;font-weight:800;color:#111827;margin-top:28px;margin-bottom:4px;display:flex;align-items:center;gap:10px;">
            ⚠️ Known Limitations
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="background:linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);border-radius:14px;padding:18px 22px;border:1px solid #fecaca;margin:12px 0;">
            <div style="color:#475569;font-size:14px;line-height:1.9;">
                <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:6px;">
                    <span style="color:#dc2626;font-weight:700;">•</span>
                    <span>Trained only on a handful of Cambodian English-language outlets; may underperform on other regions or styles.</span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:6px;">
                    <span style="color:#dc2626;font-weight:700;">•</span>
                    <span>English only — Khmer-language or heavily code-switched text is out of scope.</span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px;">
                    <span style="color:#dc2626;font-weight:700;">•</span>
                    <span>History is per-session and clears on browser refresh.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="text-align:center;color:#94a3b8;font-size:12px;margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;">
            Cambodian News Classifier v1.0 · Built with Streamlit · Thesis Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model selector
# --------------------------------------------------------------------------- #
def model_selector(MODEL_INFO, models) -> None:
    if not models:
        st.warning("No models available.")
        return
    
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    
    default_idx = models.index(st.session_state.model_key) if st.session_state.model_key in models else 0
    choice = st.selectbox(
        "🤖 Classification model",
        models,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_INFO.get(k, {}).get('display', k)}  ·  "
            f"Acc {MODEL_INFO.get(k, {}).get('accuracy', 0)*100:.1f}% / F1 {MODEL_INFO.get(k, {}).get('macro_f1', 0)*100:.1f}%"
        ),
        key="model_select",
        help="Pick which fine-tuned encoder runs the classification. "
        "RoBERTa is the best performer.",
    )
    st.session_state.model_key = choice


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        current = st.session_state.model_key
        MODEL_INFO = get_model_info()
        if current and current in MODEL_INFO:
            st.markdown(f"**Active model:** {MODEL_INFO[current].get('display', current)}")
            st.caption(
                f"Accuracy {MODEL_INFO[current].get('accuracy', 0)*100:.2f}% · "
                f"Macro-F1 {MODEL_INFO[current].get('macro_f1', 0)*100:.2f}% (test set)"
            )
        st.caption("Switch models from the dropdown on the Classifier page.")
        st.divider()
        st.caption("Corpus: undersampling_no_environment · 5 classes · max_length 512")


def main() -> None:
    render_header()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button(
            "📊 Classifier", 
            use_container_width=True,
            type="primary" if st.session_state.page == "Classifier" else "secondary",
        ):
            st.session_state.page = "Classifier"
            st.rerun()
    
    with col2:
        if st.button(
            "📋 Session History",
            use_container_width=True,
            type="primary" if st.session_state.page == "Session History" else "secondary",
        ):
            st.session_state.page = "Session History"
            st.rerun()
    
    with col3:
        if st.button(
            "ℹ️ About",
            use_container_width=True,
            type="primary" if st.session_state.page == "About" else "secondary",
        ):
            st.session_state.page = "About"
            st.rerun()
    
    st.markdown("---")
    
    render_sidebar()
    page = st.session_state.page
    if page == "Classifier":
        page_classifier()
    elif page == "Session History":
        page_history()
    else:
        page_about()


if __name__ == "__main__":
    main()
