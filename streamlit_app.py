"""Cambodian News Classifier - Streamlit dashboard.

Run from the project root:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

# Force CPU and disable memory caching
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import streamlit as st

# Add the project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Try to import the predictor with error handling
try:
    from inference.predictor import (
        DEFAULT_MODEL,
        MODEL_INFO,
        available_models,
        classify,
        classify_multiple,
        get_labels,
    )
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    st.error(f"⚠️ Could not import predictor: {e}")

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
    .comparison-table-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e5e7eb;
        margin: 12px 0;
        overflow-x: auto;
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .comparison-table thead th {
        background: #1e3a8a;
        color: white;
        padding: 10px 12px;
        text-align: left;
    }
    .comparison-table thead th:first-child {border-radius: 8px 0 0 0;}
    .comparison-table thead th:last-child {border-radius: 0 8px 0 0;}
    .comparison-table tbody td {
        padding: 10px 12px;
        border-bottom: 1px solid #e5e7eb;
        vertical-align: middle;
    }
    .comparison-table tbody tr:last-child td {border-bottom: none;}
    .comparison-table .model-name {font-weight: 700;}
    .comparison-table .text-center {text-align: center;}
    .comparison-table .text-success {color: #16a34a;}
    .comparison-table .text-danger {color: #ef4444;}
    .comparison-table .text-muted {color: #64748b; font-size: 12px;}
    .comparison-table .badge-thesis {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }
    .comparison-table .badge-app {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
    }
    .comparison-table .badge-app.good {background: #dcfce7; color: #16a34a;}
    .comparison-table .badge-app.bad {background: #fee2e2; color: #ef4444;}
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
    .history-stat-item .number {font-size: 24px; font-weight: 800; color: #111827;}
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
    .about-grid-item .icon {font-size: 28px; margin-bottom: 6px;}
    .about-grid-item .label {font-weight: 700; color: #111827; font-size: 14px;}
    .about-grid-item .desc {color: #64748b; font-size: 12px; margin-top: 4px;}
    .about-section {
        margin: 20px 0 12px 0;
        padding: 16px 20px;
        background: #f8fafc;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
    }
    .about-section .title {font-weight: 700; color: #111827; font-size: 15px;}
    .about-section .content {
        color: #475569;
        font-size: 14px;
        margin-top: 6px;
        line-height: 1.7;
    }
    .ensemble-banner {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    .ensemble-banner .ensemble-title {font-weight: 700; font-size: 14px; margin-bottom: 4px; opacity: 0.9;}
    .ensemble-banner .ensemble-prediction {font-size: 28px; font-weight: 800; margin: 4px 0;}
    .ensemble-banner .ensemble-confidence {font-size: 14px; opacity: 0.9;}
    .ensemble-banner .ensemble-models {font-size: 12px; opacity: 0.7; margin-top: 4px;}
    .model-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #e5e7eb;
    }
    .model-card .model-name {font-weight: 700; font-size: 13px; color: #111827; margin-bottom: 6px;}
    .model-card .model-prediction {font-size: 18px; font-weight: 800; margin: 4px 0;}
    .model-card .model-confidence {font-size: 12px; color: #64748b;}
    .model-card .model-metrics {font-size: 11px; color: #94a3b8; margin-top: 4px;}
    .model-card.recommended {border: 2px solid #16a34a; background: #f0fdf4;}
    .agreement-bar {display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap;}
    .agreement-item {
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: #f1f5f9;
        color: #111827;
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
    st.session_state.setdefault("model_key", DEFAULT_MODEL if PREDICTOR_AVAILABLE else "distilbert")
    st.session_state.setdefault("use_ensemble", False)


_init_state()


def _color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


def html_block(html: str) -> None:
    cleaned = "".join(line.strip() for line in html.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    html_block("""
        <div class="app-header">
          <div class="brand">
            <div class="brand-logo">📰</div>
            <div>
              <div class="brand-title">Cambodian News Classifier</div>
              <div class="brand-sub">Multi-class news article categorization</div>
            </div>
          </div>
        </div>
    """)


# --------------------------------------------------------------------------- #
# Confidence bars
# --------------------------------------------------------------------------- #
def render_scores(scores: dict[str, float], title: str = "📊 Confidence Scores") -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f'<div style="font-weight:700;color:#111827;margin:6px 0 4px;">{title}</div>']
    for cat, prob in ordered:
        pct = prob * 100
        color = _color(cat)
        parts.append(f'''
            <div class="bar-row">
                <div class="bar-name">{cat}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                </div>
                <div class="bar-val" style="color:{color};">{pct:.1f}%</div>
            </div>
        ''')
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Render functions
# --------------------------------------------------------------------------- #
def render_model_comparison(results: dict) -> None:
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
        is_recommended = key == "distilbert"
        with cols[idx]:
            st.markdown(f"""
                <div class="model-card {'recommended' if is_recommended else ''}" style="border-left:4px solid {color};">
                    <div class="model-name">{display_name} {'✅' if is_recommended else ''}</div>
                    <div class="model-prediction" style="color:{color};">{pred_cat}</div>
                    <div class="model-confidence">Confidence: {conf*100:.1f}%</div>
                    <div class="model-metrics">Acc: {result.get('accuracy', 0)*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)


def render_ensemble_results(results: dict) -> dict:
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if len(valid_results) < 2:
        st.info("Need at least 2 models for ensemble prediction.")
        return {}
    ensemble_probs = {}
    for label in get_labels():
        ensemble_probs[label] = sum(r["probs"][label] for r in valid_results.values()) / len(valid_results)
    pred = max(ensemble_probs, key=ensemble_probs.get)
    conf = ensemble_probs[pred]
    predictions = [r["predicted"] for r in valid_results.values()]
    agreement = Counter(predictions)
    st.markdown(f"""
        <div class="ensemble-banner">
            <div class="ensemble-title">🎯 Ensemble Prediction ({len(valid_results)} models)</div>
            <div class="ensemble-prediction" style="color:white;">{pred}</div>
            <div class="ensemble-confidence">Confidence: {conf*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    render_scores(ensemble_probs, "📊 Ensemble Confidence Scores")
    return {"predicted": pred, "confidence": conf, "probs": ensemble_probs, "models_used": len(valid_results)}


# --------------------------------------------------------------------------- #
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())
        html_block(f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;">
                <div>
                    <p class="card-title">Input Section</p>
                    <p class="card-sub">Paste or upload news text for category classification.</p>
                </div>
                <div style="color:#111827;font-size:13px;">{chars:,} chars &nbsp;|&nbsp; {words:,} words</div>
            </div>
        """)

        if PREDICTOR_AVAILABLE:
            col_mode, col_model = st.columns([1, 2])
            with col_mode:
                use_ensemble = st.checkbox("🎯 Ensemble Mode", value=st.session_state.use_ensemble)
                st.session_state.use_ensemble = use_ensemble
            with col_model:
                if not use_ensemble:
                    model_selector()
        else:
            st.error("⚠️ Predictor module not available. Please check your installation.")

        tab_text, tab_pdf = st.tabs(["Direct Text Entry", "PDF Upload"])
        with tab_text:
            text = st.text_area("Text input", value=st.session_state.input_text, height=320,
                               label_visibility="collapsed",
                               placeholder="Paste news text here (English).", key="text_area_input")
            st.session_state.input_text = text

        with tab_pdf:
            pdf = st.file_uploader("Upload a PDF article", type=["pdf"], label_visibility="collapsed")
            if pdf is not None:
                try:
                    import pdfplumber
                    text_parts = []
                    with pdfplumber.open(io.BytesIO(pdf.read())) as pdf_file:
                        for page in pdf_file.pages:
                            text_parts.append(page.extract_text() or "")
                    extracted = "\n".join(text_parts).strip()
                    if extracted:
                        st.session_state.input_text = extracted
                        text = extracted
                        st.success(f"✅ Extracted {len(extracted.split()):,} words from PDF.")
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

        words = len(text.split())
        analyze = st.button("Analyze Text", use_container_width=True, disabled=words == 0 or not PREDICTOR_AVAILABLE)
        
        if analyze and PREDICTOR_AVAILABLE:
            if words < MIN_WORDS:
                st.warning(f"⚠️ Only {words} words detected. {MIN_WORDS}+ words give more reliable results.")
            
            try:
                if st.session_state.use_ensemble:
                    with st.spinner("Running ensemble inference…"):
                        results = classify_multiple(text)
                        st.session_state.last_multiple_results = results
                        st.session_state.last_result = None
                else:
                    with st.spinner("Running inference…"):
                        scores = classify(text, st.session_state.model_key)
                        st.session_state.last_result = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "model": MODEL_INFO[st.session_state.model_key]["display"],
                            "model_key": st.session_state.model_key,
                            "category": max(scores, key=scores.get),
                            "confidence": max(scores.values()),
                            "scores": scores,
                            "chars": len(text),
                            "words": words,
                            "preview": text.strip().replace("\n", " ")[:160],
                        }
                        st.session_state.last_multiple_results = None
                        st.session_state.history.insert(0, st.session_state.last_result)
            except Exception as e:
                st.error(f"Error during classification: {e}")

    with right:
        if st.session_state.use_ensemble:
            results = st.session_state.last_multiple_results
            if results is None:
                st.markdown('<div class="card placeholder-card">'
                           '<div style="font-size:48px;line-height:1;">🎯</div>'
                           '<div style="margin-top:18px;font-size:18px;font-weight:700;color:#111827;">Ensemble Mode Active</div>'
                           '<div style="margin-top:12px;color:#111827;font-size:14px;">All models will run simultaneously.</div>'
                           '</div>', unsafe_allow_html=True)
            else:
                render_model_comparison(results)
                ensemble_result = render_ensemble_results(results)
                if ensemble_result and not any(h.get("is_ensemble", False) for h in st.session_state.history[:1]):
                    st.session_state.history.insert(0, {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": "Ensemble",
                        "category": ensemble_result["predicted"],
                        "confidence": ensemble_result["confidence"],
                        "scores": ensemble_result["probs"],
                        "chars": len(text),
                        "words": words,
                        "preview": text.strip().replace("\n", " ")[:160],
                        "is_ensemble": True,
                    })
        else:
            result = st.session_state.last_result
            if result is None:
                st.markdown('<div class="card placeholder-card">'
                           '<div style="font-size:48px;line-height:1;">🔎</div>'
                           '<div style="margin-top:18px;font-size:18px;font-weight:700;color:#111827;">Ready to classify</div>'
                           '<div style="margin-top:12px;color:#111827;font-size:14px;">Paste text or upload a PDF.</div>'
                           '</div>', unsafe_allow_html=True)
                return

            cat = result["category"]
            conf = result["confidence"] * 100
            html_block(f"""
                <div class="result-head">
                    <span class="conf-pill">{conf:.1f}% confidence</span>
                    <div class="result-kicker">🏆 TOP CLASSIFICATION</div>
                    <div class="result-cat" style="color:{_color(cat)};">{cat}</div>
                </div>
            """)

            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="stat-box"><div class="stat-num">{result["chars"]:,}</div><div class="stat-lab">Characters</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-box"><div class="stat-num">{result["words"]:,}</div><div class="stat-lab">Words</div></div>', unsafe_allow_html=True)

            render_scores(result["scores"])
            st.caption(f"Model: {result['model']}")

            e1, e2 = st.columns(2)
            with e1:
                st.download_button("⬇ Export", data=json.dumps(result, indent=2),
                                 file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                                 mime="application/json", use_container_width=True)
            with e2:
                if st.button("🗑 Clear", use_container_width=True):
                    st.session_state.last_result = None
                    st.session_state.input_text = ""
                    st.rerun()


# --------------------------------------------------------------------------- #
# History page
# --------------------------------------------------------------------------- #
def page_history() -> None:
    history = st.session_state.history
    st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px;">
            <div>
                <div style="font-size:22px;font-weight:800;color:#111827;">📋 Session History</div>
                <div style="font-size:14px;color:#64748b;margin-top:4px;">All classification results from this session</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not history:
        st.info("📭 No classifications yet.")
        return

    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)
    
    st.markdown(f"""
        <div class="history-stats">
            <div class="history-stat-item"><div class="number">{len(history)}</div><div class="label">Total Articles</div></div>
            <div class="history-stat-item"><div class="number">{len(set(cats))}/5</div><div class="label">Categories</div></div>
            <div class="history-stat-item"><div class="number">{(sum(confidences)/len(confidences))*100:.1f}%</div><div class="label">Avg Confidence</div></div>
            <div class="history-stat-item"><div class="number" style="color:{_color(top_cat)};">{top_cat.capitalize()}</div><div class="label">Top Category</div></div>
        </div>
    """, unsafe_allow_html=True)

    for h in history:
        color = _color(h["category"])
        conf_pct = h["confidence"] * 100
        conf_color = "#16a34a" if conf_pct >= 80 else ("#f59e0b" if conf_pct >= 60 else "#ef4444")
        ensemble_badge = ' 🎯 Ensemble' if h.get('is_ensemble', False) else ''
        st.markdown(f"""
            <div class="history-item">
                <div class="header">
                    <div class="badge-group">
                        <span class="badge" style="background:{color};">{h['category']}</span>
                        <span style="font-size:13px;font-weight:600;">{conf_pct:.1f}%</span>
                        <span class="confidence-bar-mini"><span class="fill" style="width:{conf_pct:.1f}%;background:{conf_color};"></span></span>
                        <span style="font-size:11px;color:#64748b;">{h['model']}{ensemble_badge}</span>
                    </div>
                    <div class="meta-info"><span>📝 {h['words']:,} words</span><span>🕐 {h['timestamp']}</span></div>
                </div>
                <div class="preview">{h['preview']}…</div>
            </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #
def page_about() -> None:
    labels = get_labels() if PREDICTOR_AVAILABLE else ["economics", "health", "politics", "sports", "technology"]
    
    st.markdown("""
        <div style="text-align:center;margin-bottom:20px;">
            <div style="font-size:48px;line-height:1.2;">📰</div>
            <div style="font-size:24px;font-weight:800;color:#111827;margin-top:8px;">Cambodian News Classifier</div>
            <div style="font-size:14px;color:#64748b;margin-top:4px;">Thesis Project — Deployment Deliverable (Part 3)</div>
        </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    st.markdown("""
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:16px;margin-bottom:12px;">
            📊 Model Performance: Thesis vs Streamlit App
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="comparison-table-container">
            <table class="comparison-table">
                <thead>
                    <tr><th>Model</th><th style="text-align:center;">In Thesis</th><th style="text-align:center;">In Streamlit App</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="model-name">RoBERTa</td>
                        <td class="text-center"><span class="badge-thesis">🏆 Best on average</span><br><span class="text-muted">(91.75% accuracy)</span></td>
                        <td class="text-center"><span class="badge-app bad">❌ Less reliable</span><br><span class="text-muted" style="color:#ef4444;">May misclassify some articles</span></td>
                    </tr>
                    <tr>
                        <td class="model-name">DistilBERT</td>
                        <td class="text-center"><span class="badge-thesis">Second best</span><br><span class="text-muted">(91.07% accuracy)</span></td>
                        <td class="text-center"><span class="badge-app good">✅ More reliable</span><br><span class="text-muted" style="color:#16a34a;">Consistently accurate on articles</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background:#dbeafe;border-radius:12px;padding:14px 18px;border:1px solid #93c5fd;margin:12px 0;">
            <div style="font-weight:700;color:#1e40af;font-size:14px;margin-bottom:4px;">💡 Why This Happens</div>
            <div style="color:#1e3a8a;font-size:13px;line-height:1.7;">
                <strong>RoBERTa</strong> has the best test-set metrics (91.75% accuracy). However, 
                <strong>DistilBERT</strong> often performs better on individual articles because confidence 
                varies sample by sample. The model with the best average performance may not always 
                produce the highest confidence for every individual article.
                <br><br><span style="font-size:12px;color:#1e40af;">📖 Source: Thesis Section 5.1.1</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background:#f0fdf4;border-radius:12px;padding:12px 16px;border:1px solid #bbf7d0;margin:8px 0;">
            <div style="font-weight:700;color:#16a34a;font-size:14px;margin-bottom:4px;">✅ Conclusion for This App</div>
            <div style="color:#166534;font-size:13px;line-height:1.7;">
                <strong>DistilBERT</strong> is the default model in this Streamlit app because it 
                consistently provides more reliable classifications on individual articles, 
                even though RoBERTa has slightly higher academic test-set metrics.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("""
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">📂 About This Project</div>
        <div class="about-grid">
            <div class="about-grid-item"><div class="icon">📂</div><div class="label">5 Categories</div><div class="desc">Politics, Technology, Economics, Health, Sports</div></div>
            <div class="about-grid-item"><div class="icon">🤖</div><div class="label">4 Models</div><div class="desc">BERT, DistilBERT, RoBERTa, ELECTRA</div></div>
            <div class="about-grid-item"><div class="icon">⭐</div><div class="label">Default Model</div><div class="desc">DistilBERT — Most reliable in practice</div></div>
            <div class="about-grid-item"><div class="icon">🎯</div><div class="label">Ensemble Mode</div><div class="desc">Combine all models for robust predictions</div></div>
        </div>
    """, unsafe_allow_html=True)

    # Academic Rankings
    if PREDICTOR_AVAILABLE:
        st.markdown("""
            <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">🏆 Academic Performance Rankings</div>
        """, unsafe_allow_html=True)
        model_data = []
        for key, info in MODEL_INFO.items():
            is_available = key in available_models()
            model_data.append({"Model": info["display"], "Accuracy": f"{info['accuracy']*100:.2f}%",
                             "Macro F1": f"{info['macro_f1']*100:.2f}%", "Status": "✅ Available" if is_available else "❌"})
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, hide_index=True, use_container_width=True)

    # Pipeline
    st.markdown("""
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">🔧 Pipeline</div>
        <div class="about-section"><div class="title">1. Preprocess</div><div class="content">Lowercase, strip HTML / URLs / emails / digits, drop stop-words</div></div>
        <div class="about-section"><div class="title">2. Tokenize</div><div class="content">Model-specific HuggingFace tokenizer, <code>max_length=512</code></div></div>
        <div class="about-section"><div class="title">3. Classify</div><div class="content"><code>TransformerClassifier</code> ([CLS] → 512 → LogSoftmax over 5 classes)</div></div>
        <div class="about-section"><div class="title">4. Report</div><div class="content"><code>exp()</code> of log-probabilities gives confidence scores</div></div>
    """, unsafe_allow_html=True)

    # Categories
    st.markdown("""
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">📂 Categories</div>
    """, unsafe_allow_html=True)
    cat_cols = st.columns(5)
    for idx, cat in enumerate(labels):
        color = _color(cat)
        with cat_cols[idx]:
            st.markdown(f"""
                <div style="background:{color}20;border:2px solid {color};border-radius:12px;padding:12px 8px;text-align:center;">
                    <div style="font-size:20px;font-weight:800;color:{color};text-transform:uppercase;">{cat[0]}</div>
                    <div style="font-size:12px;font-weight:600;color:#111827;margin-top:4px;">{cat.capitalize()}</div>
                </div>
            """, unsafe_allow_html=True)

    st.caption("The Environment class was excluded from the corpus, leaving five balanced categories.")


# --------------------------------------------------------------------------- #
# Model selector
# --------------------------------------------------------------------------- #
def model_selector() -> None:
    models = available_models() if PREDICTOR_AVAILABLE else []
    if not models:
        st.error("No model checkpoints found. Please add .pt files to models/undersampling_no_environment/")
        return
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    default_idx = models.index(st.session_state.model_key)
    
    def format_func(k):
        label = f"{MODEL_INFO[k]['display']}  ·  Acc {MODEL_INFO[k]['accuracy']*100:.1f}%"
        if k == "distilbert":
            label += " ✅ BEST IN PRACTICE"
        elif k == "roberta":
            label += " 🏆 ACADEMIC BEST"
        return label
    
    choice = st.selectbox("🤖 Classification model", models, index=default_idx, format_func=format_func)
    st.session_state.model_key = choice


def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        if PREDICTOR_AVAILABLE and st.session_state.model_key in MODEL_INFO:
            is_recommended = st.session_state.model_key == "distilbert"
            st.markdown(f"**Active model:** {MODEL_INFO[st.session_state.model_key]['display']} {'✅' if is_recommended else ''}")
            if is_recommended:
                st.success("✅ Best in practice")
            st.caption(f"Accuracy {MODEL_INFO[st.session_state.model_key]['accuracy']*100:.2f}% · Macro-F1 {MODEL_INFO[st.session_state.model_key]['macro_f1']*100:.2f}%")
        st.divider()
        if st.session_state.use_ensemble:
            st.success("🎯 Ensemble Mode: ON")
        else:
            st.info("💡 Try Ensemble Mode for more robust predictions")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    render_header()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("📊 Classifier", use_container_width=True, type="primary" if st.session_state.page == "Classifier" else "secondary"):
            st.session_state.page = "Classifier"
            st.rerun()
    with col2:
        if st.button("📋 Session History", use_container_width=True, type="primary" if st.session_state.page == "Session History" else "secondary"):
            st.session_state.page = "Session History"
            st.rerun()
    with col3:
        if st.button("ℹ️ About", use_container_width=True, type="primary" if st.session_state.page == "About" else "secondary"):
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
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.stop()
