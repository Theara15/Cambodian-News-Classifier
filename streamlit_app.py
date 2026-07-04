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

import pandas as pd
import streamlit as st

from inference.predictor import (
    DEFAULT_MODEL,
    MODEL_INFO,
    available_models,
    classify,
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

    .page-nav-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }
    
    .input-hint {color: #111827; font-size:12px; margin-top:6px;}
    .feature-list {list-style:none; padding-left:0; margin:18px 0 0 0; color:#111827;}
    .feature-list li {margin:10px 0; display:flex; align-items:flex-start; gap:10px; color:#111827;}
    .feature-list li::before {content:'✓'; color:#16a34a; font-weight:700;}

    div.stButton > button {border-radius:12px; font-weight:700; background: #2563eb; color:white; border:none;}
    div.stButton > button:hover {background: #1e40af;}
    div.stButton > button[disabled] {background: #93c5fd; color: #ffffff;}

    .placeholder-card {text-align:center; color:#111827;}
    
    /* Style for file uploader */
    .stFileUploader label {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .stFileUploader > div {
        color: #111827 !important;
    }
    
    /* Make file uploader button text white */
    .stFileUploader button {
        color: white !important;
        background: #2563eb !important;
    }
    .stFileUploader button:hover {
        background: #1e40af !important;
    }
    
    /* Style for success messages */
    .stAlert {
        color: #111827 !important;
    }
    .stAlert p {
        color: #111827 !important;
    }
    
    /* Style for info messages */
    .stAlert[data-baseweb="notification"] {
        color: #111827 !important;
    }
    
    /* Style for selectbox labels */
    .stSelectbox label {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    /* Style for metrics */
    [data-testid="metric-container"] label {
        color: #111827 !important;
    }
    [data-testid="metric-container"] div {
        color: #111827 !important;
    }
    
    /* Style for subheaders in About page */
    .stSubheader {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        margin-top: 24px !important;
        margin-bottom: 12px !important;
    }
    
    /* Style for captions */
    .stCaption {
        color: #111827 !important;
    }
    
    /* Style for success messages */
    .stSuccess {
        color: #111827 !important;
        background: #f0fdf4 !important;
        border-color: #bbf7d0 !important;
    }
    .stSuccess p {
        color: #111827 !important;
    }
    
    /* Style for text input labels */
    .stTextInput label {
        color: #111827 !important;
    }
    
    /* Style for text area labels */
    .stTextArea label {
        color: #111827 !important;
    }
    
    /* Style for tabs */
    .stTabs [data-baseweb="tab"] {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important;
    }
    
    /* Style for download buttons */
    .stDownloadButton button {
        background: #2563eb !important;
        color: white !important;
    }
    .stDownloadButton button:hover {
        background: #1e40af !important;
    }
    
    /* Style for table in About page */
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
    
    /* Style for dataframe */
    .stDataFrame {
        color: #111827 !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #111827 !important;
    }
    
    /* About page specific styles */
    .about-icon {
        font-size: 48px;
        text-align: center;
        margin-bottom: 12px;
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
    .model-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
    }
    .model-badge.best {
        background: #dbeafe;
        color: #1e40af;
    }
    .model-badge.good {
        background: #f0fdf4;
        color: #16a34a;
    }
    .model-badge.available {
        background: #f8fafc;
        color: #475569;
    }
    
    /* History page specific styles */
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
    
    /* Filter section */
    .filter-section {
        display: flex;
        gap: 12px;
        margin: 16px 0;
        flex-wrap: wrap;
        align-items: center;
    }
    .filter-section .search-box {
        flex: 1;
        min-width: 200px;
    }
    .filter-section .category-filter {
        min-width: 150px;
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
    st.session_state.setdefault("model_key", DEFAULT_MODEL)


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
def render_scores(scores: dict[str, float]) -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = [
        '<div style="font-weight:700;color:#111827;margin:6px 0 4px;">'
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

        model_selector()

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
            with st.spinner("Running inference…"):
                scores = classify(text, st.session_state.model_key)
            top_cat = max(scores, key=scores.get)
            result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_INFO[st.session_state.model_key]["display"],
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
                '<div class="card placeholder-card">'
                "<div style=\"font-size:48px;line-height:1;\">🔎</div>"
                "<div style=\"margin-top:18px;font-size:18px;font-weight:700;color:#111827;\">Ready to classify your article</div>"
                "<div style=\"margin-top:12px;color:#111827;font-size:14px;max-width:440px;margin-left:auto;margin-right:auto;\">Paste text or upload a PDF, then use the button below to see the predicted category and confidence scores.</div>"
                "<ul class=\"feature-list\">"
                "<li>Supports news text input</li>"
                "<li>6-category classification model</li>"
                "<li>Confidence scores for all categories</li>"
                "</ul>"
                "</div>",
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
# Session history page - Redesigned
# --------------------------------------------------------------------------- #
def page_history() -> None:
    history = st.session_state.history
    
    # Header
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

    # Stats
    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)
    
    # Create stats grid
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

    # Filters
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

    # Filter and sort
    rows = []
    for h in history:
        if query and query.lower() not in h["preview"].lower():
            continue
        if cat_filter != "All" and h["category"] != cat_filter:
            continue
        rows.append(h)
    
    # Sort
    if sort_by == "Most Recent":
        rows = rows  # Already in reverse chronological order
    elif sort_by == "Oldest First":
        rows = list(reversed(rows))
    elif sort_by == "Highest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"], reverse=True)
    elif sort_by == "Lowest Confidence":
        rows = sorted(rows, key=lambda x: x["confidence"])

    # Results count
    st.markdown(
        f"""
        <div style="margin:12px 0 16px 0;color:#64748b;font-size:13px;">
            Showing <strong>{len(rows)}</strong> article{'s' if len(rows) != 1 else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display items
    for h in rows:
        color = _color(h["category"])
        conf_pct = h["confidence"] * 100
        
        # Determine confidence level color
        if conf_pct >= 80:
            conf_color = "#16a34a"  # Green - high confidence
            conf_emoji = "🟢"
        elif conf_pct >= 60:
            conf_color = "#f59e0b"  # Yellow - medium confidence
            conf_emoji = "🟡"
        else:
            conf_color = "#ef4444"  # Red - low confidence
            conf_emoji = "🔴"
        
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
                    </div>
                    <div class="meta-info">
                        <span>🤖 {h['model']}</span>
                        <span>📝 {h['words']:,} words</span>
                        <span>🕐 {h['timestamp']}</span>
                    </div>
                </div>
                <div class="preview">{h['preview']}…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Export section
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
# About page - Redesigned with RoBERTa/DistilBERT note
# --------------------------------------------------------------------------- #
def page_about() -> None:
    labels = get_labels()
    
    # Header section
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
    
    # Quick stats grid
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
    
    # Description
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
    
    # Model Performance
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            📊 Model Performance
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create a styled dataframe for model performance
    model_data = []
    for key, info in MODEL_INFO.items():
        is_available = key in available_models()
        is_best = key == "roberta"  # RoBERTa is the best performer
        model_data.append({
            "Model": info["display"],
            "Accuracy": f"{info['accuracy']*100:.2f}%",
            "Macro F1": f"{info['macro_f1']*100:.2f}%",
            "Status": "✅ Available" if is_available else "❌ Unavailable",
            "🏆": "⭐ Best" if is_best else ""
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
            "🏆": st.column_config.TextColumn("", width="small"),
        }
    )
    
    # Recommendation with RoBERTa/DistilBERT note
    st.markdown(
        """
        <div style="background:#dbeafe;border-radius:12px;padding:16px 20px;border:1px solid #93c5fd;margin:12px 0;">
            <div style="font-weight:700;color:#1e40af;font-size:15px;margin-bottom:6px;">
                🏆 Model Recommendation
            </div>
            <div style="color:#1e3a8a;font-size:14px;line-height:1.7;">
                <strong>RoBERTa</strong> is best overall by report/test metrics — 
                highest accuracy (91.75%) and macro-F1 (91.77%) on the balanced dataset.
            </div>
            <div style="color:#1e3a8a;font-size:13px;line-height:1.6;margin-top:8px;padding-top:8px;border-top:1px solid #93c5fd;">
                💡 <strong>Note:</strong> <strong>DistilBERT</strong> can still look better on one input because 
                confidence varies sample by sample. The model with the best test-set metrics 
                may not always produce the highest confidence for every individual article.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Pipeline section
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
    
    # Categories section
    st.markdown(
        """
        <div style="font-size:18px;font-weight:700;color:#111827;margin-top:24px;margin-bottom:12px;">
            📂 Categories
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Category chips
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
    
    # Known limitations
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
    
    # Footer
    st.markdown(
        """
        <div style="text-align:center;color:#94a3b8;font-size:12px;margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;">
            Cambodian News Classifier v1.0 · Built with Streamlit · Thesis Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model selector (on-page) + sidebar + routing
# --------------------------------------------------------------------------- #
def model_selector() -> None:
    """On-page transformer picker bound to ``st.session_state.model_key``."""
    models = available_models()
    if not models:
        st.error(
            "No checkpoints found in models/undersampling_no_environment/. "
            "Make sure the fine-tuned .pt weights are included in your deployment "
            "or tracked with Git LFS."
        )
        return
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    default_idx = models.index(st.session_state.model_key)
    choice = st.selectbox(
        "🤖 Classification model",
        models,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_INFO[k]['display']}  ·  "
            f"Acc {MODEL_INFO[k]['accuracy']*100:.1f}% / F1 {MODEL_INFO[k]['macro_f1']*100:.1f}%"
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
        if current in MODEL_INFO:
            st.markdown(f"**Active model:** {MODEL_INFO[current]['display']}")
            st.caption(
                f"Accuracy {MODEL_INFO[current]['accuracy']*100:.2f}% · "
                f"Macro-F1 {MODEL_INFO[current]['macro_f1']*100:.2f}% (test set)"
            )
        st.caption("Switch models from the dropdown on the Classifier page.")
        st.divider()
        st.caption("Corpus: undersampling_no_environment · 5 classes · max_length 512")


def main() -> None:
    render_header()
    
    # Custom navigation using columns with buttons
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


main()
