
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #dee2e6;
        font-size: 15px;
    }
    .result-fake {
        background: linear-gradient(135deg, #ff6b6b, #ee0979);
        color: white;
        padding: 20px 28px;
        border-radius: 14px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin: 16px 0;
    }
    .result-real {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        padding: 20px 28px;
        border-radius: 14px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin: 16px 0;
    }
    .keyword-pill {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px 3px;
        font-size: 13px;
        font-weight: 500;
    }
    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #343a40;
        margin-bottom: 8px;
        margin-top: 20px;
    }
    .history-header {
        font-size: 15px;
        font-weight: 600;
        color: #495057;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ── Session state for history ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Model Info")
    st.markdown("""
    <div class='stat-box'>
        <div style='font-size:13px;color:#6c757d;'>Algorithm</div>
        <div style='font-size:16px;font-weight:700;color:#343a40;'>Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("""
    <div class='stat-box'>
        <div style='font-size:13px;color:#6c757d;'>Vectorizer</div>
        <div style='font-size:16px;font-weight:700;color:#343a40;'>TF-IDF</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("""
    <div class='stat-box'>
        <div style='font-size:13px;color:#6c757d;'>Training Data</div>
        <div style='font-size:16px;font-weight:700;color:#343a40;'>44,000+ Articles</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown("""
- Paste full articles for best accuracy
- Headlines alone may be less reliable
- Model trained on English news only
    """)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#adb5bd;text-align:center;'>Built by Shrijita Bhattacharyya<br>Python · scikit-learn · Streamlit</div>",
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📰 Fake News Detection")
st.markdown("<p style='color:#6c757d;font-size:16px;'>Paste a news article below to instantly check if it's real or fake — powered by NLP and machine learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Main layout ───────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("<div class='section-title'>📝 Article Text</div>", unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        placeholder="Paste your news article here...",
        height=280,
        label_visibility="collapsed"
    )

    # Live stats
    word_count = len(user_input.split()) if user_input.strip() else 0
    char_count = len(user_input)
    stat1, stat2 = st.columns(2)
    stat1.metric("Words", word_count)
    stat2.metric("Characters", char_count)

    predict_btn = st.button("🔍 Analyse Article", use_container_width=True, type="primary")

with col_result:
    st.markdown("<div class='section-title'>🧠 Analysis Result</div>", unsafe_allow_html=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Please paste some article text first.")
        elif word_count < 5:
            st.warning("⚠️ Please enter at least a few sentences for accurate results.")
        else:
            with st.spinner("Analysing..."):
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                proba = model.predict_proba(input_vec)[0]
                confidence = round(proba[prediction] * 100, 1)
                label = "FAKE" if prediction == 0 else "REAL"

                # Result banner
                if label == "FAKE":
                    st.markdown(f"<div class='result-fake'>🚨 FAKE News — {confidence}% confidence</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-real'>✅ REAL News — {confidence}% confidence</div>", unsafe_allow_html=True)

                # Confidence bar
                st.markdown("<div class='section-title'>Confidence Breakdown</div>", unsafe_allow_html=True)
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    st.markdown("🚨 **Fake**")
                    st.progress(float(proba[0]))
                    st.caption(f"{round(proba[0]*100, 1)}%")
                with conf_col2:
                    st.markdown("✅ **Real**")
                    st.progress(float(proba[1]))
                    st.caption(f"{round(proba[1]*100, 1)}%")

                # Top influencing keywords
                st.markdown("<div class='section-title'>🔑 Top Influencing Keywords</div>", unsafe_allow_html=True)
                feature_names = np.array(vectorizer.get_feature_names_out())
                tfidf_scores = input_vec.toarray()[0]
                top_indices = tfidf_scores.argsort()[-12:][::-1]
                top_words = feature_names[top_indices]
                top_words = [w for w in top_words if len(w) > 3][:10]

                pills_html = "".join([f"<span class='keyword-pill'>{w}</span>" for w in top_words])
                st.markdown(pills_html, unsafe_allow_html=True)

                # Add to history
                st.session_state.history.insert(0, {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Preview": user_input[:60].strip() + "...",
                    "Result": f"{'🚨 FAKE' if label == 'FAKE' else '✅ REAL'}",
                    "Confidence": f"{confidence}%"
                })
    else:
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:40px 24px;text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);color:#adb5bd;margin-top:8px;'>
            <div style='font-size:48px;margin-bottom:12px;'>🔍</div>
            <div style='font-size:15px;'>Paste an article and click<br><strong>Analyse Article</strong> to see results</div>
        </div>
        """, unsafe_allow_html=True)

# ── Prediction History ────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("<div class='section-title'>🕓 Prediction History (This Session)</div>", unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
