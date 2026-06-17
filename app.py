import streamlit as st
import numpy as np
import joblib
import time
import os
from pathlib import Path

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# ROBUST MODEL PATH RESOLUTION
# ----------------------------
# Resolve relative to THIS file's location, not the process's working
# directory — Streamlit Cloud does not guarantee cwd == repo root.
APP_DIR = Path(__file__).parent.resolve()
MODEL_PATH = APP_DIR / "models" / "iris_model.pkl"

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

FLOWER_INFO = {
    "setosa": {
        "emoji": "🌼",
        "color": "#00ffaa",
        "desc": "Small, delicate petals with vivid colors.",
    },
    "versicolor": {
        "emoji": "🌺",
        "color": "#00e0ff",
        "desc": "Medium-sized with a balanced, vibrant structure.",
    },
    "virginica": {
        "emoji": "🌸",
        "color": "#b388ff",
        "desc": "Larger petals with elegant, broad shape.",
    },
}

ACCENT = "#00e0ff"
ACCENT_SOFT = "rgba(0, 224, 255, 0.15)"
ACCENT_GLOW = "rgba(0, 224, 255, 0.55)"

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 20% 20%, #131722 0%, #0f0f0f 45%, #060608 100%);
    color: #f5f5f7;
}}

#MainMenu, footer, header {{visibility: hidden;}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1300px;
    margin: 0 auto;
}}

.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, {ACCENT});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}}

.hero-sub {{
    color: #9aa0ad;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 1.8rem;
    line-height: 1.5;
}}

.glass-card {{
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}}

.glass-card:hover {{
    border-color: {ACCENT_SOFT};
    box-shadow: 0 8px 36px {ACCENT_SOFT};
    transform: translateY(-2px);
}}

.feature-label {{
    font-size: 0.78rem;
    color: {ACCENT};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}}

.section-heading {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #f5f5f7;
    margin-bottom: 0.8rem;
}}

div[data-testid="stSlider"] > div > div > div > div {{
    background: linear-gradient(90deg, {ACCENT}, #6c5ce7) !important;
}}

div[data-testid="stSlider"] label {{
    color: #d8dbe3 !important;
    font-weight: 500;
}}

div.stButton > button {{
    width: 100%;
    background: linear-gradient(135deg, {ACCENT} 0%, #6c5ce7 100%);
    color: #06070a;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 0;
    margin-top: 0.6rem;
    box-shadow: 0 0 18px {ACCENT_GLOW}, 0 0 40px rgba(108, 92, 231, 0.25);
    transition: all 0.25s ease;
    letter-spacing: 0.3px;
}}

div.stButton > button:hover {{
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 0 28px {ACCENT_GLOW}, 0 0 60px rgba(108, 92, 231, 0.4);
    color: #000;
}}

div.stButton > button:active {{
    transform: translateY(0px) scale(0.99);
}}

.result-card {{
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 2.2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    text-align: center;
    min-height: 380px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    animation: fadeIn 0.6s ease;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.result-emoji {{
    font-size: 5.5rem;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 20px var(--glow-color));
}}

.result-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    text-transform: capitalize;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}}

.result-desc {{
    color: #9aa0ad;
    font-size: 0.92rem;
    margin-bottom: 1.4rem;
}}

.confidence-badge {{
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 50px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
}}

.placeholder-text {{
    color: #5e6473;
    font-size: 1rem;
    text-align: center;
    line-height: 1.6;
}}

.placeholder-icon {{
    font-size: 3.5rem;
    opacity: 0.3;
    margin-bottom: 1rem;
}}

.prob-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}}

.prob-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #d8dbe3;
    text-transform: capitalize;
}}

.prob-value {{
    font-size: 0.85rem;
    font-weight: 700;
}}

.prob-track {{
    width: 100%;
    height: 10px;
    border-radius: 50px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
    margin-bottom: 14px;
}}

.prob-fill {{
    height: 100%;
    border-radius: 50px;
    transition: width 0.8s ease;
}}

.error-card {{
    background: rgba(255, 60, 60, 0.06);
    border: 1px solid rgba(255, 80, 80, 0.3);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    color: #ffb3b3;
    font-size: 0.92rem;
    line-height: 1.7;
}}

.error-card code {{
    background: rgba(255,255,255,0.08);
    padding: 2px 6px;
    border-radius: 6px;
    color: #ffd9d9;
}}

.app-footer {{
    text-align: center;
    color: #565c69;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    letter-spacing: 0.3px;
}}

.app-footer span {{
    color: {ACCENT};
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER (rendered first so errors below still show a styled page)
# ----------------------------
st.markdown('<div class="hero-title">🌸 Iris AI Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">A premium machine learning experience for real-time iris species '
    'prediction — powered by a Random Forest model trained on botanical measurements.</div>',
    unsafe_allow_html=True
)

# ----------------------------
# LOAD MODEL (with a clear, styled failure path instead of a raw traceback)
# ----------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.markdown(f"""
    <div class="error-card">
        <b>⚠️ Model file not found.</b><br><br>
        Looked for it at:<br>
        <code>{MODEL_PATH}</code><br><br>
        This usually means one of the following:<br>
        1. <code>models/iris_model.pkl</code> was not committed to GitHub (check it shows up in the repo on github.com).<br>
        2. It's being skipped by a <code>.gitignore</code> rule (e.g. <code>*.pkl</code> or <code>models/</code>).<br>
        3. The file was committed with Git LFS, which Streamlit Cloud does not pull by default.<br><br>
        Fix: run <code>git add models/iris_model.pkl -f</code>, commit, and push, then redeploy.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.markdown(f"""
    <div class="error-card">
        <b>⚠️ Failed to load the model.</b><br><br>
        <code>{type(e).__name__}: {e}</code><br><br>
        The file exists at <code>{MODEL_PATH}</code> but could not be unpickled.
        This often happens when the model was trained with a different scikit-learn /
        numpy version than what's installed here. Re-train and re-save the model using
        the same versions pinned in <code>requirements.txt</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ----------------------------
# SESSION STATE
# ----------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False
    st.session_state.pred_class = None
    st.session_state.pred_proba = None

# ----------------------------
# LAYOUT: SPLIT SCREEN
# ----------------------------
left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📏 Sepal Measurements</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-label">Sepal Length (cm)</div>', unsafe_allow_html=True)
    sepal_length = st.slider("", 4.0, 8.0, 5.8, 0.1, key="sl", label_visibility="collapsed")

    st.markdown('<div class="feature-label">Sepal Width (cm)</div>', unsafe_allow_html=True)
    sepal_width = st.slider("", 2.0, 4.5, 3.0, 0.1, key="sw", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🌿 Petal Measurements</div>', unsafe_allow_html=True)

    st.markdown('<div class="feature-label">Petal Length (cm)</div>', unsafe_allow_html=True)
    petal_length = st.slider("", 1.0, 7.0, 4.0, 0.1, key="pl", label_visibility="collapsed")

    st.markdown('<div class="feature-label">Petal Width (cm)</div>', unsafe_allow_html=True)
    petal_width = st.slider("", 0.1, 2.5, 1.2, 0.1, key="pw", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("✨ Predict Species", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing flower features..."):
            time.sleep(0.9)
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)[0]

            if isinstance(prediction, (int, np.integer)):
                pred_label = CLASS_NAMES[prediction]
            else:
                pred_label = str(prediction).lower()

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]

            st.session_state.predicted = True
            st.session_state.pred_class = pred_label
            st.session_state.pred_proba = proba

with right_col:
    if st.session_state.predicted and st.session_state.pred_class:
        species = st.session_state.pred_class
        info = FLOWER_INFO.get(species, FLOWER_INFO["setosa"])
        proba = st.session_state.pred_proba

        confidence_text = ""
        if proba is not None:
            class_idx = CLASS_NAMES.index(species) if species in CLASS_NAMES else int(np.argmax(proba))
            confidence = proba[class_idx] * 100
            confidence_text = f"{confidence:.1f}% Confidence"

        st.markdown(f"""
        <div class="result-card" style="--glow-color:{info['color']};">
            <div class="result-emoji">{info['emoji']}</div>
            <div class="result-name" style="color:{info['color']};">{species}</div>
            <div class="result-desc">{info['desc']}</div>
            {f'<div class="confidence-badge" style="color:{info["color"]}; border-color:{info["color"]}55;">{confidence_text}</div>' if confidence_text else ''}
        </div>
        """, unsafe_allow_html=True)

        if proba is not None:
            st.markdown('<div class="glass-card" style="margin-top:1.2rem;">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">📊 Probability Breakdown</div>', unsafe_allow_html=True)

            bars_html = ""
            for c in CLASS_NAMES:
                idx = CLASS_NAMES.index(c)
                pct = proba[idx] * 100
                color = FLOWER_INFO[c]["color"]
                bars_html += f"""
                <div class="prob-row">
                    <span class="prob-label">{c}</span>
                    <span class="prob-value" style="color:{color};">{pct:.1f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}aa);"></div>
                </div>
                """
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
            <div class="placeholder-icon">🌷</div>
            <div class="placeholder-text">
                Adjust the measurements on the left<br>
                and click <b>Predict Species</b><br>
                to see the AI result here.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer">Built with <span>Streamlit</span> + <span>Machine Learning</span> · '
    'Random Forest Classifier</div>',
    unsafe_allow_html=True
)