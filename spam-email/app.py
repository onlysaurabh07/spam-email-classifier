from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path("spam_model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def main() -> None:
    st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")
    st.title("Spam Email / SMS Classifier")
    st.write("Classify message text as **SPAM** or **NOT SPAM**.")

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.warning(
            "Model files not found. Run `python model.py` first to train and save artifacts."
        )
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
