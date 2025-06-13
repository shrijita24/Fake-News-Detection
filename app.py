
import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

model = joblib.load("news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

user_input = st.text_area("Paste a news article below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        result = "✅ REAL News" if prediction == 1 else "🚨 FAKE News"
        st.success(result)
