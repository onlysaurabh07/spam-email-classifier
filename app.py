from pathlib import Path

import joblib
import streamlit as st
from model import train_and_save


MODEL_PATH = Path("spam_model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")
DEFAULT_DATA_PATH = Path("spam.csv")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def ensure_artifacts() -> None:
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return

    with st.spinner("Model files missing. Training model for first-time setup..."):
        train_and_save(DEFAULT_DATA_PATH, MODEL_PATH, VECTORIZER_PATH)


def main() -> None:
    st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")
    st.title("Spam Email / SMS Classifier")
    st.write("Classify message text as **SPAM** or **NOT SPAM**.")

    try:
        ensure_artifacts()
    except Exception as exc:
        st.error(f"Unable to prepare model files automatically: {exc}")
        st.stop()

    model, vectorizer = load_artifacts()

    user_input = st.text_area(
        "Enter message text",
        placeholder="Example: Congratulations! You have won a free prize. Click now...",
        height=180,
    )

    if st.button("Predict"):
        clean_input = user_input.strip()
        if not clean_input:
            st.info("Please enter some text before predicting.")
            st.stop()

        vec = vectorizer.transform([clean_input])
        pred = int(model.predict(vec)[0])
        prob_spam = float(model.predict_proba(vec)[0][1])

        if pred == 1:
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: NOT SPAM")

        st.write(f"Spam probability: **{prob_spam * 100:.2f}%**")


if __name__ == "__main__":
    main()
