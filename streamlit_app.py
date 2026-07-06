"""Cambodian News Classifier - Streamlit dashboard.

Run from the project root:

    streamlit run streamlit_app.py

Serves the four fine-tuned ``undersampling_no_environment`` encoders (BERT,
DistilBERT, RoBERTa, ELECTRA) behind a single classifier UI with session
history and an about/model-card page.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from collections import Counter

import pandas as pd
import streamlit as st

from inference.predictor import (
    DEFAULT_MODEL,
    MODEL_INFO,
    available_models,
    classify,
    classify_multiple,
    get_labels,
)

# --------------------------------------------------------------------------- #
# Page config + theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cambodian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Per-category accent colours (consistent across the whole app).
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
    /* hide default chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
    body, .stApp, .main {background: #f8f9fb !important; color: #111827 !important;}
    .block-container {
        background: #ffffff !important;
        padding-top: 1rem;
        padding-bottom: 2rem;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        max-width: 1200px;
    }
    .app-header {
        background: #1e3a8a;
        border-radius: 18px;
        padding: 20px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 16px 40px rgba(30, 58, 138, 0.16);
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
    }
    .card-title {font-size:22px; font-weight:800; color:#111827; margin:0;}
    .card-sub {font-size:14px; color:#111827; margin-top:8px; line-height:1.5;}

    .result-head {
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:24px 28px;
        margin-bottom:18px;
    }
    .result-kicker {font-size:11px; letter-spacing:1.5px; color:#111827; font-weight:700;}
    .result-cat {font-size:32px; font-weight:800; margin:10px 0 0 0; text-transform:uppercase; color:#111827;}
    .conf-pill {
        float:right; background:#2563eb; color:white; font-weight:700;
        font-size:12px; padding:7px 14px; border-radius:999px;
    }
    .recommended-badge {
        background: #16a34a;
        color: white;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 8px;
    }

    .stat-box {
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:16px;
        padding:18px;
        text-align:center;
    }
    .stat-num {font-size:28px; font-weight:800; color:#111827;}
    .stat-lab {font-size:12px; color:#111827; margin-top:4px;}

    .bar-row {display:flex; align-items:center; margin:10px 0; font-size:13px;}
    .bar-name {width:110px; color:#111827; text-transform:capitalize;}
    .bar-track {flex:1; background:#f1f5f9; border-radius:6px; height:10px; overflow:hidden; margin:0 12px;}
    .bar-fill {height:100%; border-radius:6px;}
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
        color: #111827 !important;
        min-height: 260px;
    }
    .stTextArea textarea::placeholder {color:#6b7280 !important;}

    .input-hint {color: #111827; font-size:12px; margin-top:6px;}
    .feature-list {list-style:none; padding-left:0; margin:18px 0 0 0; color:#111827;}
    .feature-list li {margin:10px 0; display:flex; align-items:flex-start; gap:10px; color:#111827;}
    .feature-list li::before {content:'✓'; color:#16a34a; font-weight:700;}

    div.stButton > button {border-radius:12px; font-weight:700; background: #2563eb; color:white; border:none;}
    div.stButton > button:hover {background: #1e40af;}
    div.stButton > button[disabled] {background: #93c5fd; color: #ffffff;}

    .placeholder-card {text-align:center; color:#111827;}
    
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
    }
    .stFileUploader button:hover {
        background: #1e40af !important;
    }
    
    .stAlert {
        color: #111827 !important;
    }
    .stAlert p {
        color: #111827 !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        color: #111827 !important;
    }
    
    .stSelectbox label {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] label {
        color: #111827 !important;
    }
    [data-testid="metric-container"] div {
        color: #111827 !important;
    }
    
    .stSubheader {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
    }
    
    .stCaption {
        color: #111827 !important;
    }
    
    .stSuccess {
        color: #111827 !important;
        background: #f0fdf4 !important;
        border-color: #bbf7d0 !important;
    }
    .stSuccess p {
        color: #111827 !important;
    }
    
    .stTextInput label {
        color: #111827 !important;
    }
    
    .stTextArea label {
        color: #111827 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important;
    }
    
    .stDownloadButton button {
        background: #2563eb !important;
        color: white !important;
    }
    .stDownloadButton button:hover {
        background: #1e40af !important;
    }
    
    .stTable {
        color: #111827 !important;
    }
    .stTable td, .stTable th {
        color: #111827 !important;
    }
    .stTable thead tr th {
        color: #111827 !important;
        background: #f8fafc !important;
        font-weight: 700 !important;
    }
    .stTable tbody tr td {
        color: #111827 !important;
    }
    
    .stDataFrame {
        color: #111827 !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #111827 !important;
    }
    
    .about-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin: 16px 0;
    }
    .about-grid-item {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .about-grid-item .icon {
        font-size: 28px;
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
        border-radius: 12px;
        border-left: 4px solid #2563eb;
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
    
    .history-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 16px 0;
    }
    .history-stat-item {
        background: #f8fafc;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .history-stat-item .number {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
    }
    .history-stat-item .label {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .history-item {
        background: white;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    .history-item:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.06);
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
        gap: 8px;
        flex-wrap: wrap;
    }
    .history-item .meta-info {
        color: #64748b;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .history-item .meta-info span {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .history-item .preview {
        margin-top: 10px;
        color: #374151;
        font-size: 14px;
        line-height: 1.6;
        padding: 10px 14px;
        background: #f8fafc;
        border-radius: 8px;
        border-left: 3px solid #e5e7eb;
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
        transition: width 0.3s ease;
    }
    
    .model-comparison-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 12px 0;
    }
    .model-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    .model-card:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.06);
    }
    .model-card .model-name {
        font-weight: 700;
        font-size: 13px;
        color: #111827;
        margin-bottom: 6px;
    }
    .model-card .model-prediction {
        font-size: 18px;
        font-weight: 800;
        margin: 4px 0;
    }
    .model-card .model-confidence {
        font-size: 12px;
        color: #64748b;
    }
    .model-card .model-metrics {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .model-card.recommended {
        border: 2px solid #16a34a;
        background: #f0fdf4;
    }
    
    .ensemble-banner {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .ensemble-banner .ensemble-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 4px;
        opacity: 0.9;
    }
    .ensemble-banner .ensemble-prediction {
        font-size: 28px;
        font-weight: 800;
        margin: 4px 0;
    }
    .ensemble-banner .ensemble-confidence {
        font-size: 14px;
        opacity: 0.9;
    }
    .ensemble-banner .ensemble-models {
        font-size: 12px;
        opacity: 0.7;
        margin-top: 4px;
    }
    
    .agreement-bar {
        display: flex;
        gap: 8px;
        margin: 8px 0;
        flex-wrap: wrap;
    }
    .agreement-item {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: #f1f5f9;
        color: #111827;
    }
    .agreement-item.high {
        background: #dbeafe;
        color: #1e40af;
    }
    .agreement-item.medium {
        background: #fef3c7;
        color: #92400e;
    }
    .agreement-item.low {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .distilbert-highlight {
        background: linear-gradient(135deg, #dcfce7, #86efac);
        border-radius: 12px;
        padding: 12px 16px;
        border: 2px solid #16a34a;
        margin: 8px 0;
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
    st.session_state.setdefault("last_multiple_results", None)
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("model_key", DEFAULT_MODEL)
    st.session_state.setdefault("use_ensemble", False)


_init_state()


def _color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


def html_block(html: str) -> None:
    """Render raw HTML, collapsing per-line indentation."""
    cleaned = "".join(line.strip() for line in html.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header / navigation
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
def render_scores(scores: dict[str, float], title: str = "📊 Confidence Scores") -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = [
        f'<div style="font-weight:700;color:#111827;margin:6px 0 4px;">{title}</div>'
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


def render_model_comparison(results: dict) -> None:
    """Render comparison of all models with their predictions."""
    st.markdown("### 🤖 Model Comparison")
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        st.warning("No models could classify the text.")
        return
    
    cols = st.columns(min(len(valid_results), 4))
    
    for idx, (key, result) in enumerate(valid_results.items()):
        if idx >= len(cols):
            break
        
        pred_cat = result["predicted"]
        conf = result["confidence"]
        color = _color(pred_cat)
        display_name = result.get("display_name", key)
        
        # Highlight DistilBERT as recommended
        is_recommended = key == "distilbert"
        
        with cols[idx]:
            card_class = "model-card recommended" if is_recommended else "model-card"
            st.markdown(
                f"""
                <div class="{card_class}" style="border-left:4px solid {color};">
                    <div class="model-name">
                        {display_name}
                        { '⭐' if is_recommended else '' }
                    </div>
                    <div class="model-prediction" style="color:{color};">
                        {pred_cat}
                    </div>
                    <div class="model-confidence">
                        Confidence: {conf*100:.1f}%
                    </div>
                    <div class="model-metrics">
                        Acc: {result.get('accuracy', 0)*100:.1f}% · 
                        F1: {result.get('macro_f1', 0)*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            with st.expander(f"Top 3 for {display_name}"):
                sorted_probs = sorted(result["probs"].items(), key=lambda x: x[1], reverse=True)
                for cat, prob in sorted_probs[:3]:
                    pct = prob * 100
                    c = _color(cat)
                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;font-size:13px;padding:2px 0;">
                            <span style="color:{c};font-weight:600;">{cat}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def render_ensemble_results(results: dict) -> dict:
    """Render ensemble prediction results."""
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if len(valid_results) < 2:
        st.info("Need at least 2 models for ensemble prediction.")
        return {}
    
    ensemble_probs = {}
    for label in get_labels():
        ensemble_probs[label] = sum(r["probs"][label] for r in valid_results.values()) / len(valid_results)
    
    pred = max(ensemble_probs, key=ensemble_probs.get)
    conf = ensemble_probs[pred]
    color = _color(pred)
    
    predictions = [r["predicted"] for r in valid_results.values()]
    agreement = Counter(predictions)
    most_agreed = max(agreement, key=agreement.get)
    agreement_pct = agreement[most_agreed] / len(predictions)
    
    st.markdown(
        f"""
        <div class="ensemble-banner">
            <div class="ensemble-title">🎯 Ensemble Prediction ({len(valid_results)} models)</div>
            <div class="ensemble-prediction" style="color:white;">{pred}</div>
            <div class="ensemble-confidence">Confidence: {conf*100:.1f}%</div>
            <div class="ensemble-models">
                🤝 Most agreed: {most_agreed} ({agreement_pct*100:.0f}% of models)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("**Model Agreement:**")
    agreement_html = '<div class="agreement-bar">'
    for cat, count in agreement.items():
        pct = count / len(predictions)
        if pct >= 0.5:
            cls = "high"
        elif pct >= 0.25:
            cls = "medium"
        else:
            cls = "low"
        agreement_html += f'<span class="agreement-item {cls}" style="border-left:3px solid {_color(cat)};">{cat}: {count}/{len(predictions)}</span>'
    agreement_html += '</div>'
    st.markdown(agreement_html, unsafe_allow_html=True)
    
    render_scores(ensemble_probs, "📊 Ensemble Confidence Scores")
    
    return {
        "predicted": pred,
        "confidence": conf,
        "probs": ensemble_probs,
        "models_used": len(valid_results),
        "agreement": dict(agreement)
    }


# --------------------------------------------------------------------------- #
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())
        html_block(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;">
              <div>
                <p class="card-title">Input Section</p>
                <p class="card-sub">Paste or upload news text for category classification.</p>
              </div>
              <div style="color:#111827;font-size:13px;">{chars:,} chars &nbsp;|&nbsp; {words:,} words</div>
            </div>
            """
        )

        col_mode, col_model = st.columns([1, 2])
        with col_mode:
            use_ensemble = st.checkbox(
                "🎯 Ensemble Mode",
                value=st.session_state.use_ensemble,
                help="Use all available models and combine their predictions"
            )
            st.session_state.use_ensemble = use_ensemble
        
        with col_model:
            if not use_ensemble:
                model_selector()

        # Show recommendation notice
        if not use_ensemble:
            st.info(
                "💡 **Tip:** DistilBERT is currently performing best for your use case. "
                "Try Ensemble Mode for even more robust predictions!"
            )

        tab_text, tab_pdf = st.tabs(["Direct Text Entry", "PDF Upload"])
        with tab_text:
            text = st.text_area(
                "Text input",
                value=st.session_state.input_text,
                height=320,
                label_visibility="collapsed",
                placeholder="Paste news text here (English). Perfect for copied articles or short texts.",
                key="text_area_input",
            )
            st.session_state.input_text = text
            st.markdown(
                '<div class="input-hint">Perfect for copied articles or short texts.</div>',
                unsafe_allow_html=True,
            )

        with tab_pdf:
            st.markdown(
                '<div style="color:#111827;font-weight:600;margin-bottom:8px;">Upload a PDF article</div>',
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
        analyze = st.button(
            "Analyze Text",
            use_container_width=True,
            disabled=words == 0,
        )
        if analyze:
            if words < MIN_WORDS:
                st.warning(
                    f"⚠️ Only {words} words detected. {MIN_WORDS}+ words give more "
                    "reliable results, but classifying anyway."
                )
            
            if st.session_state.use_ensemble:
                with st.spinner("Running ensemble inference on all models…"):
                    results = classify_multiple(text)
                    st.session_state.last_multiple_results = results
                    st.session_state.last_result = None
            else:
                with st.spinner("Running inference…"):
                    scores = classify(text, st.session_state.model_key)
                    model_display = MODEL_INFO[st.session_state.model_key]["display"]
                    # Clean up the display name (remove emoji for history)
                    model_display_clean = model_display.replace(" ⭐", "")
                    
                    st.session_state.last_result = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": model_display_clean,
                        "model_key": st.session_state.model_key,
                        "category": max(scores, key=scores.get),
                        "confidence": max(scores.values()),
                        "scores": scores,
                        "chars": len(text),
                        "words": words,
                        "preview": text.strip().replace("\n", " ")[:160],
                        "is_recommended": st.session_state.model_key == "distilbert",
                    }
                    st.session_state.last_multiple_results = None
                    st.session_state.history.insert(0, st.session_state.last_result)

    with right:
        if st.session_state.use_ensemble:
            results = st.session_state.last_multiple_results
            if results is None:
                st.markdown(
                    '<div class="card placeholder-card">'
                    "<div style=\"font-size:48px;line-height:1;\">🎯</div>"
                    "<div style=\"margin-top:18px;font-size:18px;font-weight:700;color:#111827;\">Ensemble Mode Active</div>"
                    "<div style=\"margin-top:12px;color:#111827;font-size:14px;max-width:440px;margin-left:auto;margin-right:auto;\">"
                    "All models will run simultaneously and their predictions will be combined."
                    "</div>"
                    "<ul class=\"feature-list\">"
                    "<li>Uses all available models</li>"
                    "<li>Shows model agreement</li>"
                    "<li>Ensemble prediction is more robust</li>"
                    "</ul>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                render_model_comparison(results)
                ensemble_result = render_ensemble_results(results)
                
                if ensemble_result and not any(
                    h.get("is_ensemble", False) for h in st.session_state.history[:1]
                ):
                    history_entry = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": "Ensemble",
                        "category": ensemble_result["predicted"],
                        "confidence": ensemble_result["confidence"],
                        "scores": ensemble_result["probs"],
                        "chars": len(text),
                        "words": words,
                        "preview": text.strip().replace("\n", " ")[:160],
                        "is_ensemble": True,
                        "models_used": ensemble_result["models_used"],
                        "agreement": ensemble_result["agreement"],
                    }
                    st.session_state.history.insert(0, history_entry)
                
                if st.button("📥 Export Results", use_container_width=True):
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "text_preview": text[:500],
                        "ensemble": ensemble_result,
                        "individual_models": {
                            k: {
                                "predicted": v["predicted"],
                                "confidence": v["confidence"],
                                "scores": v["probs"]
                            }
                            for k, v in results.items() if "error" not in v
                        }
                    }
                    st.download_button(
                        "⬇ Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"ensemble_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
        
        else:
            result = st.session_state.last_result
            if result is None:
                st.markdown(
                    '<div class="card placeholder-card">'
                    "<div style=\"font-size:48px;line-height:1;\">🔎</div>"
                    "<div style=\"margin-top:18px;font-size:18px;font-weight:700;color:#111827;\">Ready to classify your article</div>"
                    "<div style=\"margin-top:12px;color:#111827;font-size:14px;max-width:440px;margin-left:auto;margin-right:auto;\">"
                    f"<strong>DistilBERT</strong> is the default model (works best for your use case)."
                    "</div>"
                    "<ul class=\"feature-list\">"
                    "<li>Supports news text input</li>"
                    "<li>5-category classification model</li>"
                    "<li>Confidence scores for all categories</li>"
                    "</ul>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                return

            cat = result["category"]
            conf = result["confidence"] * 100
            
            # Show recommended badge if using DistilBERT
            recommended_badge = ' <span class="recommended-badge">⭐ Recommended</span>' if result.get("is_recommended", False) else ''
            
            html_block(
                f"""
                <div class="result-head">
                  <span class="conf-pill">{conf:.1f}% confidence</span>
                  <div class="result-kicker">🏆 TOP CLASSIFICATION</div>
                  <div class="result-cat" style="color:{_color(cat)};">{cat}{recommended_badge}</div>
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
            st.caption(f"Model: {result['model']}")

            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "⬇ Export",
                    data=json.dumps(result, indent=2),
                    file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with e2:
                if st.button("🗑 Clear", use_container_width=True):
                    st.session_state.last_result = None
                    st.session_state.input_text = ""
                    st.rerun()


def _read_pdf(uploaded) -> str:
    try:
        import pdfplumber
    except ImportError:
        st.error("PDF support needs `pdfplumber`. Install it with: pip install pdfplumber")
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
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px;">
            <div>
                <div style="font-size:22px;font-weight:800;color:#111827;">📋 Session History</div>
                <div style="font-size:14px;color:#64748b;margin-top:4px;">
                    All classification results from this session
                </div>
            </div>
            <div style="font-size:13px;color:#64748b;">
                🔄 Resets on browser refresh
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

    st.markdown("---")
    
    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])
    
    with filter_col1:
        query = st.text_input(
            "🔍 Search articles",
            placeholder="Filter by preview text…",
            label_visibility="collapsed",
        )
    
    with filter_col2:
        cat_filter = st.selectbox(
            "Category",
            ["All"] + sorted(set(cats)),
            label_visibility="collapsed",
        )
    
    with filter_col3:
        sort_options = ["Most Recent", "Oldest First", "Highest Confidence", "Lowest Confidence"]
        sort_by = st.selectbox(
            "Sort",
            sort_options,
            index=0,
            label_visibility="collapsed",
        )

    rows = []
    for h in history:
        if query and query.lower() not in h["preview"].lower():
            continue
        if cat_filter != "All" and h["category"] != cat_filter:
            continue
        rows.append(h)
    
    if sort_by == "Most Recent":
        rows = rows
    elif sort_by == "Oldest First":
        rows = list(reversed(rows))
    elif sort_by == "Highest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"], reverse=True)
    elif sort_by == "Lowest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"])

    st.markdown(
        f"""
        <div style="margin:12px 0 16px 0;color:#64748b;font-size:13px;">
            Showing <strong>{len(rows)}</strong> article{'s' if len(rows) != 1 else ''}
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
        elif conf_pct >= 60:
            conf_color = "#f59e0b"
            conf_emoji = "🟡"
        else:
            conf_color = "#ef4444"
            conf_emoji = "🔴"
        
        ensemble_badge = ' 🎯 Ensemble' if h.get('is_ensemble', False) else ''
        recommended_badge = ' ⭐' if h.get('is_recommended', False) else ''
        
        st.markdown(
            f"""
            <div class="history-item">
                <div class="header">
                    <div class="badge-group">
                        <span class="badge" style="background:{color};">{h['category']}</span>
                        <span style="font-size:13px;font-weight:600;color:#111827;">
                            {conf_emoji} {conf_pct:.1f}%
                        </span>
                        <span class="confidence-bar-mini">
                            <span class="fill" style="width:{conf_pct:.1f}%;background:{conf_color};"></span>
                        </span>
                        <span style="font-size:11px;color:#64748b;">{h['model']}{ensemble_badge}{recommended_badge}</span>
                    </div>
                    <div class="meta-info">
                        <span>📝 {h['words']:,} words</span>
                        <span>🕐 {h['timestamp']}</span>
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
                "ensemble": h.get("is_ensemble", False),
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
    labels = get_labels()
    
    st.markdown(
        """
        <div style="text-align:center;margin-bottom:20px;">
            <div style="font-size:48px;line-height:1.2;">📰</div>
            <div style="font-size:24px;font-weight:800;color:#111827;margin-top:8px;">
                Cambodian News Classifier
            </div>
            <div style="font-size:14px;color:#64748b;margin-top:4px;">
                Thesis Project — Deployment Deliverable (Part 3)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Show DistilBERT recommendation prominently
    st.markdown(
        """
        <div class="distilbert-highlight">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:28px;">⭐</span>
                <div>
                    <div style="font-weight:700;color:#16a34a;font-size:16px;">DistilBERT is the Recommended Model</div>
                    <div style="color:#166534;font-size:13px;">
                        Based on your testing, DistilBERT consistently outperforms other models 
                        on your specific use case. Use it as the default or try Ensemble Mode!
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div class="about-grid">
            <div class="about-grid-item">
                <div class="icon">📂</div>
                <div class="label">5 Categories</div>
                <div class="desc">Politics, Technology, Economics, Health, Sports</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">🤖</div>
                <div class="label">4 Models</div>
                <div class="desc">BERT, DistilBERT, RoBERTa, ELECTRA</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">⭐</div>
                <div class="label">Best for Your Use Case</div>
                <div class="desc">DistilBERT — Most reliable predictions</div>
            </div>
            <div class="about-grid-item">
                <div class="icon">🎯</div>
                <div class="label">Ensemble Mode</div>
                <div class="desc">Combine all models for robust predictions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="background:#f8fafc;border-radius:12px;padding:16px 20px;border:1px solid #e5e7eb;margin:12px 0;">
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
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            📊 Model Performance
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    model_data = []
    for key, info in MODEL_INFO.items():
        is_available = key in available_models()
        is_recommended = key == "distilbert"
        model_data.append({
            "Model": f"{info['display']} {'⭐' if is_recommended else ''}",
            "Accuracy": f"{info['accuracy']*100:.2f}%",
            "Macro F1": f"{info['macro_f1']*100:.2f}%",
            "Status": "✅ Available" if is_available else "❌ Unavailable",
            "Recommendation": "⭐ Recommended" if is_recommended else "",
        })
    
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
            "Recommendation": st.column_config.TextColumn("", width="small"),
        }
    )
    
    st.markdown(
        """
        <div style="background:#dbeafe;border-radius:12px;padding:16px 20px;border:1px solid #93c5fd;margin:12px 0;">
            <div style="font-weight:700;color:#1e40af;font-size:15px;margin-bottom:6px;">
                💡 Model Selection Guide
            </div>
            <div style="color:#1e3a8a;font-size:14px;line-height:1.7;">
                <strong>DistilBERT</strong> is the default and recommended model based on your testing.<br>
                <strong>RoBERTa</strong> has the highest test-set metrics but may underperform on specific articles.<br>
                Use <strong>Ensemble Mode</strong> to combine all models for the most robust predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            🔧 Pipeline
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
            <div class="title">4. Ensemble (Optional)</div>
            <div class="content">
                Average probabilities from all models for more robust predictions
            </div>
        </div>
        <div class="about-section">
            <div class="title">5. Report</div>
            <div class="content">
                <code>exp()</code> of log-probabilities gives the confidence scores shown in the dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            📂 Categories
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
                <div style="
                    background:{color}20;
                    border:2px solid {color};
                    border-radius:12px;
                    padding:12px 8px;
                    text-align:center;
                ">
                    <div style="font-size:20px;font-weight:800;color:{color};text-transform:uppercase;">
                        {cat[0]}
                    </div>
                    <div style="font-size:12px;font-weight:600;color:#111827;margin-top:4px;">
                        {cat.capitalize()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.caption(
        "This is the *no-environment* variant — the Environment class was excluded "
        "from the corpus, leaving five balanced-enough categories."
    )
    
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            ⚠️ Known Limitations
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <div style="background:#fef2f2;border-radius:12px;padding:16px 20px;border:1px solid #fecaca;margin:12px 0;">
            <div style="color:#475569;font-size:14px;line-height:1.8;">
                <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">
                    <span style="color:#dc2626;">•</span>
                    <span>Trained only on a handful of Cambodian English-language outlets; may underperform on other regions or styles.</span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">
                    <span style="color:#dc2626;">•</span>
                    <span>English only — Khmer-language or heavily code-switched text is out of scope.</span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:10px;">
                    <span style="color:#dc2626;">•</span>
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
def model_selector() -> None:
    models = available_models()
    if not models:
        st.error(
            "No checkpoints found in models/undersampling_no_environment/. "
            "Make sure the fine-tuned .pt weights are included in your deployment."
        )
        return
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    default_idx = models.index(st.session_state.model_key)
    
    # Format function with recommendation badge
    def format_func(k):
        label = f"{MODEL_INFO[k]['display']}  ·  Acc {MODEL_INFO[k]['accuracy']*100:.1f}% / F1 {MODEL_INFO[k]['macro_f1']*100:.1f}%"
        if k == "distilbert":
            label += " ⭐ RECOMMENDED"
        return label
    
    choice = st.selectbox(
        "🤖 Classification model",
        models,
        index=default_idx,
        format_func=format_func,
        key="model_select",
        help="Pick which fine-tuned encoder runs the classification. DistilBERT is recommended.",
    )
    st.session_state.model_key = choice


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        current = st.session_state.model_key
        if current in MODEL_INFO:
            is_recommended = current == "distilbert"
            st.markdown(f"**Active model:** {MODEL_INFO[current]['display']} {'⭐' if is_recommended else ''}")
            if is_recommended:
                st.success("✅ Recommended model")
            st.caption(
                f"Accuracy {MODEL_INFO[current]['accuracy']*100:.2f}% · "
                f"Macro-F1 {MODEL_INFO[current]['macro_f1']*100:.2f}% (test set)"
            )
        st.divider()
        st.caption("Corpus: undersampling_no_environment · 5 classes · max_length 512")
        if st.session_state.use_ensemble:
            st.success("🎯 Ensemble Mode: ON")
            st.caption("Using all available models")
        else:
            st.info("💡 Tip: Try Ensemble Mode for more robust predictions")


def main() -> None:
    render_header()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📊 Classifier", use_container_width=True, 
                     type="primary" if st.session_state.page == "Classifier" else "secondary"):
            st.session_state.page = "Classifier"
            st.rerun()
    
    with col2:
        if st.button("📋 Session History", use_container_width=True,
                     type="primary" if st.session_state.page == "Session History" else "secondary"):
            st.session_state.page = "Session History"
            st.rerun()
    
    with col3:
        if st.button("ℹ️ About", use_container_width=True,
                     type="primary" if st.session_state.page == "About" else "secondary"):
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
