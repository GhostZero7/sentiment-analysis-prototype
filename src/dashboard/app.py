"""Streamlit dashboard for the sentiment analysis prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_texts


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"


def _load_metrics() -> pd.DataFrame:
    preferred = RESULTS_DIR / "model_metrics_rerun.csv"
    fallback = RESULTS_DIR / "model_metrics.csv"
    path = preferred if preferred.is_file() else fallback
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_summary() -> dict[str, object]:
    path = RESULTS_DIR / "training_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_confusion_matrix_image(model_name: str):
    path = RESULTS_DIR / "confusion_matrices" / f"{model_name}.png"
    return path if path.is_file() else None


def main() -> None:
    st.set_page_config(page_title="ZESCO Sentiment Prototype", layout="wide")
    st.title("ZESCO Sentiment Analysis Prototype")
    st.caption("Baseline models trained on the final sarcasm-aware dataset.")

    metrics = _load_metrics()
    summary = _load_summary()

    with st.sidebar:
        st.header("Dataset")
        st.write(f"Train rows: {summary.get('train_rows', 'n/a')}")
        st.write(f"Test rows: {summary.get('test_rows', 'n/a')}")
        st.write(f"Label column: {summary.get('label_column', 'corrected_label')}")
        st.write(f"Vectorizer: {summary.get('vectorizer_path', 'n/a')}")
        st.write(f"Model dir: {MODELS_DIR}")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Model Metrics")
        if metrics.empty:
            st.info("No metrics file found yet.")
        else:
            st.dataframe(metrics, use_container_width=True)

        st.subheader("Try a Comment")
        sample_text = st.text_area(
            "Paste a cleaned or raw comment",
            value="ba zesco this is good news thank you",
            height=120,
        )
        if st.button("Predict sentiment"):
            try:
                predictions = predict_texts([sample_text])
                st.dataframe(predictions, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with col2:
        st.subheader("Confusion Matrix")
        if metrics.empty:
            st.info("Train the models first to generate confusion matrices.")
        else:
            model_names = metrics["model"].tolist() if "model" in metrics.columns else []
            selected_model = st.selectbox("Choose a model", model_names or ["naive_bayes", "logistic_regression", "svm"])
            image_path = _load_confusion_matrix_image(selected_model)
            if image_path is not None:
                st.image(str(image_path), caption=f"{selected_model} confusion matrix", use_container_width=True)
            else:
                st.warning("Confusion matrix image not found. Run `scripts/evaluate_models.py` first.")

        st.subheader("Training Summary")
        if summary:
            st.json(summary)
        else:
            st.info("Training summary not found.")


if __name__ == "__main__":
    main()

