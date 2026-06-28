# 📰 Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

An end-to-end NLP-powered web application that classifies news articles as **Real or Fake** in real time — built with Python, scikit-learn, TF-IDF, and Streamlit.

> 🚀 **[Live Demo → Try it here](https://your-streamlit-url.streamlit.app)**  
> *(Replace this link with your actual Streamlit Cloud URL)*

---

## ✨ Features

- 🎯 **Instant classification** — paste any article and get a Real/Fake verdict in seconds
- 📊 **Confidence scores** — see percentage confidence for both Real and Fake with visual progress bars
- 🔑 **Top influencing keywords** — understand *why* the model made its prediction
- 🕓 **Session history** — track all predictions made during your session
- 📝 **Live text stats** — real-time word and character count as you type
- 🧹 **Clean, responsive UI** — two-column layout with styled result banners and sidebar info

---

## 🖼️ Screenshot

> *(Add a screenshot of your app here after deploying)*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| ML Model | Logistic Regression |
| Vectorizer | TF-IDF (scikit-learn) |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 🧠 How It Works

```
Raw Article Text
      │
      ▼
Text Preprocessing (stop word removal)
      │
      ▼
TF-IDF Vectorisation
      │
      ▼
Logistic Regression Classifier
      │
      ▼
Prediction + Confidence Score + Top Keywords
```

1. **Data** — Trained on 44,000+ real and fake news articles (`True.csv` + `Fake.csv`)
2. **Preprocessing** — TF-IDF vectoriser removes stop words and caps high-frequency terms (`max_df=0.7`)
3. **Model** — Logistic Regression classifier with `predict_proba` for confidence scoring
4. **Inference** — Input text is vectorised and classified; top TF-IDF scoring tokens surfaced as keywords

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/shrijita24/Fake-News-Detection.git
cd Fake-News-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates .pkl files)
python main.py

# 4. Launch the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Fake-News-Detection/
├── app.py              # Streamlit web application
├── main.py             # Model training script
├── news_model.pkl      # Trained Logistic Regression model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── True.csv            # Real news dataset
├── Fake.csv            # Fake news dataset
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
```

---

## 📊 Dataset

- **Source:** [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size:** ~44,000 articles (23,000 real + 21,000 fake)
- **Features used:** `text` column
- **Labels:** `1` = Real, `0` = Fake

---

## 🔮 Future Improvements

- [ ] URL scraping — paste a news link instead of raw text
- [ ] Transformer-based model (BERT) for higher accuracy
- [ ] Multi-language support
- [ ] SHAP/LIME explainability for deeper keyword analysis
- [ ] REST API endpoint for integration with other apps

---

## 👩‍💻 Author

**Shrijita Bhattacharyya**  
B.Tech Computer Science — IEM Kolkata | CGPA 9.25

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/shrijita-bhattacharyya/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-343a40?style=flat&logo=netlify)](https://shrijitabhattacharyya-portfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/shrijita24)

---

<p align="center">Made with ❤️ and Python</p>
