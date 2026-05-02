import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load spam dataset from local CSV/TSV or fallback URL."""
    if data_path.exists():
        if data_path.suffix.lower() == ".tsv":
            df = pd.read_csv(data_path, sep="\t", names=["label", "message"], header=None)
        else:
            df = pd.read_csv(data_path, encoding="latin-1")
            if {"v1", "v2"}.issubset(df.columns):
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            elif {"label", "message"}.issubset(df.columns):
                df = df[["label", "message"]]
            else:
                raise ValueError(
                    "Dataset must have columns [v1, v2] or [label, message]."
                )
    else:
        df = pd.read_csv(DATA_URL, sep="\t", names=["label", "message"], header=None)

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(int)
    df["message"] = df["message"].astype(str)
    return df


def train_and_save(data_path: Path, model_path: Path, vectorizer_path: Path) -> None:
    df = load_dataset(data_path)
    x = df["message"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    x_vec = vectorizer.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"Samples: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam email/SMS classifier.")
    parser.add_argument(
        "--data",
        default="spam.csv",
        help="Path to dataset file. Supports spam.csv or sms.tsv formats.",
    )
    parser.add_argument("--model-out", default="spam_model.pkl", help="Model output path.")
    parser.add_argument(
        "--vectorizer-out", default="vectorizer.pkl", help="Vectorizer output path."
    )
    args = parser.parse_args()

    train_and_save(Path(args.data), Path(args.model_out), Path(args.vectorizer_out))


if __name__ == "__main__":
    main()
