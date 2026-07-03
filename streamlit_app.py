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

# --------------------------------------------------------------------------- #
# Custom CSS matching the design from screenshots
# --------------------------------------------------------------------------- #
CSS = """
<style>
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
    }
    
    /* Main page background */
    .stApp {
        background: #f0f2f6 !important;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* ========== HEADER ========== */
    .app-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d4a7a 100%);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.15);
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .brand-icon {
        width: 48px;
        height: 48px;
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
    }
    
    .brand-text h1 {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .brand-text span {
        color: rgba(255,255,255,0.7);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }
    
    .header-badge {
        background: rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 6px 16px;
        color: rgba(255,255,255,0.8);
        font-size: 12px;
        font-weight: 500;
    }
    
    /* ========== NAV TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #e8eaef;
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
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    /* ========== CARDS ========== */
    .card {
        background: white;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e8eaef;
        margin-bottom: 16px;
    }
    
    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #1a202c;
        margin: 0 0 4px 0;
    }
    
    .card-subtitle {
        font-size: 13px;
        color: #718096;
        margin: 0;
    }
    
    /* ========== INPUT AREA ========== */
    .input-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 12px;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .input-header .label {
        font-weight: 600;
        font-size: 15px;
        color: #2d3748;
    }
    
    .input-header .meta {
        color: #718096;
        font-size: 12px;
    }
    
    .input-hint {
        color: #718096;
        font-size: 12px;
        font-style: italic;
        margin-top: 6px;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        min-height: 280px !important;
        color: #2d3748 !important;
        line-height: 1.6 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4299e1 !important;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.12) !important;
    }
    
    /* ========== BUTTONS ========== */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        transition: all 0.2s !important;
    }
    
    .stButton button:not(:disabled) {
        background: #1a365d !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton button:not(:disabled):hover {
        background: #2d4a7a !important;
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.25) !important;
    }
    
    .stButton button:disabled {
        background: #cbd5e0 !important;
        color: #a0aec0 !important;
        border: none !important;
    }
    
    /* ========== RESULTS ========== */
    .result-header {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #e8eaef;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .result-label {
        font-size: 11px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .result-category {
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
        text-transform: uppercase;
    }
    
    .confidence-pill {
        background: #1a365d;
        color: white;
        font-weight: 700;
        font-size: 12px;
        padding: 6px 16px;
        border-radius: 20px;
        white-space: nowrap;
    }
    
    /* ========== STAT BOXES ========== */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 12px 0 16px 0;
    }
    
    .stat-box {
        background: #f7fafc;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        border: 1px solid #edf2f7;
    }
    
    .stat-box .number {
        font-size: 22px;
        font-weight: 800;
        color: #2d3748;
    }
    
    .stat-box .label {
        font-size: 11px;
        color: #718096;
        margin-top: 2px;
    }
    
    .status-ok {
        color: #38a169;
        font-weight: 600;
        font-size: 13px;
        margin: 8px 0 4px 0;
    }
    
    .status-warn {
        color: #d69e2e;
        font-weight: 600;
        font-size: 13px;
        margin: 8px 0 4px 0;
    }
    
    /* ========== CONFIDENCE BARS ========== */
    .confidence-section {
        margin-top: 12px;
    }
    
    .confidence-section .section-label {
        font-weight: 700;
        color: #2d3748;
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
        color: #4a5568;
        font-weight: 500;
        text-transform: capitalize;
        flex-shrink: 0;
    }
    
    .bar-row .bar-track {
        flex: 1;
        background: #edf2f7;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin: 0 12px;
    }
    
    .bar-row .bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }
    
    .bar-row .bar-value {
        width: 54px;
        text-align: right;
        font-weight: 700;
        color: #2d3748;
        flex-shrink: 0;
    }
    
    /* ========== HISTORY BADGE ========== */
    .category-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        color: white;
        text-transform: capitalize;
    }
    
    /* ========== PLACEHOLDER ========== */
    .placeholder {
        text-align: center;
        padding: 40px 20px;
        color: #718096;
    }
    
    .placeholder .icon {
        font-size: 40px;
        margin-bottom: 12px;
    }
    
    .placeholder .title {
        font-size: 18px;
        font-weight: 700;
        color: #2d3748;
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
        color: #38a169;
        font-weight: 700;
    }
    
    /* ========== SELECTBOX ========== */
    .stSelectbox label {
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #2d3748 !important;
    }
    
    .stSelectbox > div {
        background: white !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* ========== MISC ========== */
    .model-info {
        font-size: 11px;
        color: #718096;
        margin-top: 6px;
    }
    
    /* Remove label on tabs */
    .stTabs label {
        display: none !important;
    }
    
    /* Fix for columns */
    .row-widget.stColumns {
        gap: 20px;
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


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand">
                <div class="brand-icon">📰</div>
                <div class="brand-text">
                    <h1>Cambodian News Classifier</h1>
                    <span>Multi-class news article categorization</span>
                </div>
            </div>
            <div class="header-badge">⚡ AI Powered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Confidence bars
# --------------------------------------------------------------------------- #
def render_scores(scores: dict[str, float]) -> None:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    html = ['<div class="confidence-section"><div class="section-label">📊 Confidence Scores</div>']
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
                <div class="bar-value" style="color:{color};">{pct:.1f}%</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


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
    choice = st.selectbox(
        "🤖 Classification Model",
        models,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_INFO[k]['display']}  ·  "
            f"Acc {MODEL_INFO[k]['accuracy']*100:.1f}% / F1 {MODEL_INFO[k]['macro_f1']*100:.1f}%"
        ),
        key="model_select",
        help="Pick which fine-tuned encoder runs the classification.",
    )
    st.session_state.model_key = choice


# --------------------------------------------------------------------------- #
# PDF reader
# --------------------------------------------------------------------------- #
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
# Classifier page
# --------------------------------------------------------------------------- #
def page_classifier() -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        text = st.session_state.input_text
        chars, words = len(text), len(text.split())

        st.markdown(
            f"""
            <div class="card">
                <div class="input-header">
                    <span class="label">📝 Input Section</span>
                    <span class="meta">{chars:,} chars · {words:,} words</span>
                </div>
                <div style="margin-bottom:12px;">
                    <span style="font-size:13px;color:#718096;">Paste or upload news text for category classification.</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        model_selector()

        tab_text, tab_pdf = st.tabs(["📄 Text Input", "⬆ PDF Upload"])

        with tab_text:
            st.markdown(
                """
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin:8px 0 6px 0;">
                    <span style="font-weight:600;font-size:14px;color:#2d3748;">Direct Text Entry</span>
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
                    st.success(f"✅ Extracted {len(extracted.split()):,} words from PDF.")
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
                    f"⚠️ Only {words} words detected. {MIN_WORDS}+ words give more "
                    "reliable results, but classifying anyway."
                )
            with st.spinner("🧠 Running inference..."):
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
                """
                <div class="card placeholder">
                    <div class="icon">📈</div>
                    <div class="title">Results Panel</div>
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

        st.markdown(
            f"""
            <div class="result-header">
                <div>
                    <div class="result-label">🏆 Top Classification</div>
                    <div class="result-category" style="color:{color};">{cat}</div>
                </div>
                <div class="confidence-pill">{conf:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="number">{result["chars"]:,}</div>
                    <div class="label">Characters</div>
                </div>
                <div class="stat-box">
                    <div class="number">{result["words"]:,}</div>
                    <div class="label">Words</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        render_scores(result["scores"])
        st.markdown(f'<p class="model-info">🤖 Model: {result["model"]}</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Export",
                data=json.dumps(result, indent=2),
                file_name=f"classification_{result['timestamp'].replace(':', '-')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
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
            <div class="card-subtitle">All classification results from this session</div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.info("📭 No classifications yet. Analyze an article to populate the history.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    confidences = [h["confidence"] for h in history]
    cats = [h["category"] for h in history]
    top_cat = max(set(cats), key=cats.count)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Total Articles", len(history))
    col2.metric("🏷️ Categories Used", f"{len(set(cats))}/5")
    col3.metric("📊 Avg Confidence", f"{(sum(confidences) / len(confidences)) * 100:.1f}%")
    col4.metric("🏆 Top Category", top_cat.capitalize())

    st.markdown("---")

    filter_col1, filter_col2 = st.columns([3, 1])
    query = filter_col1.text_input("🔍 Search articles...", placeholder="Filter by preview text…")
    cat_filter = filter_col2.selectbox("Category", ["All"] + sorted(set(cats)))

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
            <div style="background:white;border-radius:12px;padding:14px 18px;margin-bottom:8px;border:1px solid #e8eaef;">
                <span class="category-badge" style="background:{color};">{h['category']}</span>
                <span style="float:right;color:#718096;font-size:12px;">
                    {h['confidence']*100:.1f}% · {h['model']} · {h['timestamp']}
                </span>
                <div style="margin-top:8px;color:#2d3748;font-size:14px;">{h['preview']}…</div>
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

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Export All (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="session_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
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
            <div class="card-subtitle">Cambodian News Classifier — Thesis Project</div>
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
def main() -> None:
    render_header()

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 Classifier", "📋 Session History", "ℹ️ About"])

    with tab1:
        page_classifier()
    with tab2:
        page_history()
    with tab3:
        page_about()


if __name__ == "__main__":
    main()
