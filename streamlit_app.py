"""Cambodian News Classifier - Streamlit dashboard.

Run from the project root:

    streamlit run streamlit_app.py
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
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Cambodian News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Category colors
CATEGORY_COLORS = {
    "politics": "#7c3aed",
    "technology": "#10b981", 
    "economics": "#3b82f6",
    "health": "#0ea5e9",
    "sports": "#f59e0b",
}
DEFAULT_COLOR = "#64748b"

MIN_WORDS = 50

# --------------------------------------------------------------------------- #
# CSS - Matches screenshots precisely
# --------------------------------------------------------------------------- #
CSS = """
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
    }
    
    .stApp {
        background: #f0f2f6 !important;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* === HEADER === */
    .app-header {
        background: #1a365d;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .brand-icon {
        width: 40px;
        height: 40px;
        background: rgba(255,255,255,0.12);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .brand-title {
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin: 0;
    }
    
    .brand-sub {
        color: rgba(255,255,255,0.65);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    .header-badge {
        background: rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 4px 14px;
        color: rgba(255,255,255,0.7);
        font-size: 11px;
        font-weight: 500;
    }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border: none;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13px;
        color: #64748b;
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: #eef2ff !important;
        color: #1a365d !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    /* === CARDS === */
    .card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    
    .card-sub {
        font-size: 13px;
        color: #6b7280;
        margin: 4px 0 0 0;
    }
    
    /* === INPUT === */
    .input-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 8px;
    }
    
    .input-header .label {
        font-weight: 700;
        font-size: 16px;
        color: #111827;
    }
    
    .input-header .meta {
        color: #6b7280;
        font-size: 12px;
    }
    
    .input-hint {
        color: #6b7280;
        font-size: 12px;
        font-style: italic;
        margin-top: 4px;
    }
    
    .stTextArea textarea {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        min-height: 260px !important;
        color: #111827 !important;
        line-height: 1.6 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    }
    
    /* === BUTTON === */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        border: none !important;
    }
    
    .stButton button:not(:disabled) {
        background: #1a365d !important;
        color: white !important;
    }
    
    .stButton button:not(:disabled):hover {
        background: #2d4a7a !important;
    }
    
    .stButton button:disabled {
        background: #cbd5e1 !important;
        color: #94a3b8 !important;
    }
    
    /* === RESULTS === */
    .result-header {
        background: white;
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid #e5e7eb;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .result-kicker {
        font-size: 10px;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .result-cat {
        font-size: 28px;
        font-weight: 800;
        margin: 4px 0 0 0;
        text-transform: uppercase;
    }
    
    .conf-pill {
        background: #2563eb;
        color: white;
        font-weight: 700;
        font-size: 12px;
        padding: 5px 14px;
        border-radius: 999px;
        white-space: nowrap;
    }
    
    /* === STATS === */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 12px 0 14px 0;
    }
    
    .stat-box {
        background: #f8fafc;
        border: 1px solid #eef2f7;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }
    
    .stat-num {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
    }
    
    .stat-lab {
        font-size: 11px;
        color: #6b7280;
        margin-top: 2px;
    }
    
    .status-ok {
        color: #16a34a;
        font-weight: 600;
        font-size: 13px;
        margin: 8px 0 4px 0;
    }
    
    .status-warn {
        color: #d97706;
        font-weight: 600;
        font-size: 13px;
        margin: 8px 0 4px 0;
    }
    
    /* === CONFIDENCE BARS === */
    .conf-section {
        margin-top: 12px;
    }
    
    .conf-section .conf-label {
        font-weight: 700;
        color: #374151;
        margin-bottom: 8px;
        font-size: 14px;
    }
    
    .bar-row {
        display: flex;
        align-items: center;
        margin: 8px 0;
        font-size: 13px;
    }
    
    .bar-row .bar-name {
        width: 100px;
        color: #374151;
        font-weight: 500;
        text-transform: capitalize;
        flex-shrink: 0;
    }
    
    .bar-row .bar-track {
        flex: 1;
        background: #f1f5f9;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0 12px;
    }
    
    .bar-row .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    
    .bar-row .bar-val {
        width: 54px;
        text-align: right;
        font-weight: 700;
        color: #111827;
        flex-shrink: 0;
    }
    
    /* === BADGE === */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        color: white;
        text-transform: capitalize;
    }
    
    /* === HISTORY STATS === */
    .hist-stat {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #eef2f7;
    }
    
    .hist-stat .num {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
    }
    
    .hist-stat .lab {
        font-size: 11px;
        color: #6b7280;
        margin-top: 2px;
    }
    
    /* === HISTORY ITEM === */
    .history-item {
        background: white;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .history-item .preview {
        margin-top: 8px;
        color: #374151;
        font-size: 14px;
    }
    
    .history-item .meta {
        float: right;
        color: #6b7280;
        font-size: 12px;
    }
    
    /* === PLACEHOLDER === */
    .placeholder {
        text-align: center;
        padding: 40px 20px;
        color: #6b7280;
    }
    
    .placeholder .icon {
        font-size: 40px;
        margin-bottom: 12px;
    }
    
    .placeholder .headline {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
    }
    
    .placeholder .desc {
        font-size: 14px;
        max-width: 440px;
        margin: 8px auto 0 auto;
        line-height: 1.6;
    }
    
    .placeholder .features {
        list-style: none;
        padding: 0;
        margin: 16px 0 0 0;
        text-align: left;
        max-width: 360px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .placeholder .features li {
        padding: 6px 0;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .placeholder .features li::before {
        content: "✓";
        color: #16a34a;
        font-weight: 700;
    }
    
    /* === SELECTBOX === */
    .stSelectbox label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #111827 !important;
    }
    
    .stSelectbox > div {
        background: white !important;
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* === TABS HIDE LABEL === */
    .stTabs label {
        display: none !important;
    }
    
    .model-info {
        font-size: 11px;
        color: #6b7280;
        margin-top: 6px;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("model_key", DEFAULT_MODEL)


_init_state()


def _color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, DEFAULT_COLOR)


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
# Model selector
# --------------------------------------------------------------------------- #
def model_selector() -> None:
    models = available_models()
    if not models:
        st.error("No checkpoints found. Make sure the fine-tuned weights are included.")
        return
    if st.session_state.model_key not in models:
        st.session_state.model_key = models[0]
    default_idx = models.index(st.session_state.model_key)
    choice = st.selectbox(
        "Classification model",
        models,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_INFO[k]['display']}  ·  "
            f"Acc {MODEL_INFO[k]['accuracy']*100:.1f}% / F1 {MODEL_INFO[k]['macro_f1']*100:.1f}%"
        ),
        key="model_select",
    )
    st.session_state.model_key = choice


# --------------------------------------------------------------------------- #
# Render confidence bars
# --------------------------------------------------------------------------- #
def render_scores(scores: dict[str, float]) -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    html = ['<div class="conf-section"><div class="conf-label">📊 Confidence Scores</div>']
    for cat, prob in ordered:
        pct = prob * 100
        color = _color(cat)
        html.append(
            f"""
            <div class="bar-row">
                <div class="bar-name">{cat}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                </div>
                <div class="bar-val" style="color:{color};">{pct:.1f}%</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Render header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand">
                <div class="brand-icon">📰</div>
                <div>
                    <div class="brand-title">Cambodian News Classifier</div>
                    <div class="brand-sub">Multi-class news article categorization</div>
                </div>
            </div>
            <div class="header-badge">AI Powered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())

        # Input card
        st.markdown(
            f"""
            <div class="card">
                <div class="input-header">
                    <span class="label">📝 Input Section</span>
                    <span class="meta">{chars:,} chars · {words:,} words</span>
                </div>
                <div style="font-size:13px;color:#6b7280;margin-bottom:12px;">
                    Paste news text for classification
                </div>
            """,
            unsafe_allow_html=True,
        )

        model_selector()

        # Tabs for input
        tab_text, tab_pdf = st.tabs(["📄  Text Input", "⬆  PDF Upload"])

        with tab_text:
            st.markdown(
                """
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin:10px 0 6px 0;">
                    <span style="font-weight:600;font-size:14px;color:#111827;">Direct Text Entry</span>
                    <span class="input-hint" style="margin:0;">Perfect for copied articles or short texts</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            text = st.text_area(
                "Text input",
                value=st.session_state.input_text,
                height=280,
                label_visibility="collapsed",
                placeholder="Paste your news article here...\n\nThe government announced new economic policies today...",
                key="text_area_input",
            )
            st.session_state.input_text = text

        with tab_pdf:
            pdf = st.file_uploader("Upload a PDF article", type=["pdf"])
            if pdf is not None:
                extracted = _read_pdf(pdf)
                if extracted:
                    st.session_state.input_text = extracted
                    text = extracted
                    st.success(f"Extracted {len(extracted.split()):,} words from PDF.")
                    st.text_area("Extracted text", value=extracted, height=160)

        st.markdown("</div>", unsafe_allow_html=True)

        words = len(text.split())
        analyze = st.button(
            "🔍 Analyze Text",
            use_container_width=True,
            disabled=words == 0,
        )

        if analyze:
            if words < MIN_WORDS:
                st.warning(
                    f"Only {words} words detected. {MIN_WORDS}+ words give more "
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

    with col2:
        result = st.session_state.last_result

        if result is None:
            st.markdown(
                """
                <div class="card placeholder">
                    <div class="icon">📈</div>
                    <div class="headline">Results Panel</div>
                    <div class="desc">
                        Enter a news article on the left and click <strong>"Analyze Text"</strong>
                        to see classification results, confidence scores, and detailed analytics.
                    </div>
                    <ul class="features">
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
        color = _color(cat)

        # Result header
        st.markdown(
            f"""
            <div class="result-header">
                <div>
                    <div class="result-kicker">🏆 Top Classification</div>
                    <div class="result-cat" style="color:{color};">{cat}</div>
                </div>
                <div class="conf-pill">{conf:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stats
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-num">{result["chars"]:,}</div>
                    <div class="stat-lab">Characters</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num">{result["words"]:,}</div>
                    <div class="stat-lab">Words</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Status
        if result["words"] >= MIN_WORDS:
            st.markdown(
                '<p class="status-ok">✅ Text length is optimal for classification</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-warn">⚠️ Short text — prediction may be less reliable</p>',
                unsafe_allow_html=True,
            )

        # Confidence scores
        render_scores(result["scores"])
        st.markdown(f'<p class="model-info">🤖 Model: {result["model"]}</p>', unsafe_allow_html=True)

        # Actions
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇ Export",
                data=json.dumps(result, indent=2),
                file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.last_result = None
                st.session_state.input_text = ""
                st.rerun()


# --------------------------------------------------------------------------- #
# History page
# --------------------------------------------------------------------------- #
def page_history() -> None:
    history = st.session_state.history

    st.markdown(
        """
        <div class="card">
            <div class="card-title">📋 Session History</div>
            <div class="card-sub">All classification results from this session</div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.info("No classifications yet. Analyze an article to populate the history.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="hist-stat"><div class="num">{len(history)}</div><div class="lab">Total Articles</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="hist-stat"><div class="num">{len(set(cats))}/5</div><div class="lab">Categories Used</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="hist-stat"><div class="num">{(sum(confidences) / len(confidences)) * 100:.1f}%</div><div class="lab">Avg Confidence</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="hist-stat"><div class="num">{top_cat.capitalize()}</div><div class="lab">Top Category</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Filters
    f1, f2 = st.columns([3, 1])
    query = f1.text_input("🔍 Search articles...", placeholder="Filter by preview text…", label_visibility="collapsed")
    cat_filter = f2.selectbox("Category", ["All"] + sorted(set(cats)), label_visibility="collapsed")

    rows = []
    for h in history:
        if query and query.lower() not in h["preview"].lower():
            continue
        if cat_filter != "All" and h["category"] != cat_filter:
            continue
        rows.append(h)

    st.markdown(f"### {len(rows)} article{'s' if len(rows) != 1 else ''}")

    for h in rows:
        color = _color(h["category"])
        st.markdown(
            f"""
            <div class="history-item">
                <span class="badge" style="background:{color};">{h['category']}</span>
                <span class="meta">{h['confidence']*100:.1f}% · {h['model']} · {h['timestamp']}</span>
                <div class="preview">{h['preview']}…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    df = pd.DataFrame([
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
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Export All (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="session_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #
def page_about() -> None:
    labels = get_labels()

    st.markdown(
        """
        <div class="card">
            <div class="card-title">ℹ️ About</div>
            <div class="card-sub">Cambodian News Classifier — Thesis Project</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This dashboard classifies English-language Cambodian news articles into one of
        **five categories** using transformer encoders fine-tuned on a custom corpus
        scraped from Cambodian news outlets. It is the deployment deliverable (Part 3)
        of the thesis project.
        """
    )

    st.subheader("📂 Categories")
    st.write(", ".join(c.capitalize() for c in labels))
    st.caption(
        "This is the *no-environment* variant — the Environment class was excluded "
        "from the corpus, leaving five balanced categories."
    )

    st.subheader("📊 Model Card — Test-Set Performance")
    rank = pd.DataFrame(
        [
            {
                "Model": info["display"],
                "Accuracy": f"{info['accuracy']*100:.2f}%",
                "Macro F1": f"{info['macro_f1']*100:.2f}%",
                "Available": "✅" if key in available_models() else "❌",
            }
            for key, info in MODEL_INFO.items()
        ]
    )
    st.dataframe(rank, hide_index=True, use_container_width=True)

    st.success(
        "**🏆 RoBERTa** is the recommended default — best accuracy (91.75%) and best "
        "macro-F1 (91.77%) on this balanced dataset."
    )

    st.subheader("🔧 Pipeline")
    st.markdown(
        """
        1. **Preprocess** — lowercase, strip HTML / URLs / emails / digits, drop stop-words
        2. **Tokenize** — model-specific HuggingFace tokenizer, `max_length=512`
        3. **Classify** — `TransformerClassifier` ([CLS] → 512 → LogSoftmax over 5 classes)
        4. **Report** — `exp()` of log-probabilities gives the confidence scores shown
        """
    )

    st.subheader("⚠️ Known Limitations")
    st.markdown(
        """
        - Trained only on Cambodian English-language outlets; may underperform on other regions
        - English only — Khmer-language text is out of scope
        - History is per-session and clears on browser refresh
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="brand">
                <div class="brand-icon">📰</div>
                <div>
                    <div class="brand-title">Cambodian News Classifier</div>
                    <div class="brand-sub">Multi-class news article categorization</div>
                </div>
            </div>
            <div class="header-badge">AI Powered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()

    tab1, tab2, tab3 = st.tabs(["Classifier", "Session History", "About"])

    with tab1:
        page_classifier()
    with tab2:
        page_history()
    with tab3:
        page_about()


if __name__ == "__main__":
    main()
