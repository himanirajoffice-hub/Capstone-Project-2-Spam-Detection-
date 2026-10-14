
import streamlit as st
import joblib
import re
from pathlib import Path

# Locate saved model files
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "spam_model_final.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer_final.pkl")

# NLP text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

# Application interface
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩"
)

st.title("SMS Spam Detection Using NLP")
st.write(
    "This application uses Natural Language Processing "
    "and Logistic Regression to classify SMS messages."
)

message = st.text_area(
    "Enter an SMS message:",
    placeholder="Type or paste your SMS message here..."
)

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        cleaned_message = preprocess_text(message)
        features = vectorizer.transform([cleaned_message])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        if prediction == 1:
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM (HAM)")

        st.write(
            f"Model confidence: {max(probabilities) * 100:.2f}%"
        )

st.caption(
    "Model: Logistic Regression | "
    "Feature Extraction: TF-IDF | "
    "Dataset: Kaggle SMS Spam Classification"
)
