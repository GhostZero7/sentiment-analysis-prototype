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
BRANCHES = {
    "VADER baseline": {
        "results_dir": RESULTS_DIR,
        "models_dir": MODELS_DIR,
        "summary_file": "training_summary.json",
        "metrics_files": ("model_metrics_rerun.csv", "model_metrics.csv"),
    },
    "RoBERTa-labeled branch": {
        "results_dir": RESULTS_DIR / "roberta",
        "models_dir": MODELS_DIR / "roberta",
        "summary_file": "training_summary.json",
        "metrics_files": ("model_metrics_rerun.csv", "model_metrics.csv"),
    },
}


def _branch_config(branch_name: str) -> dict[str, object]:
    return BRANCHES.get(branch_name, BRANCHES["VADER baseline"])


def _load_metrics(branch_name: str) -> pd.DataFrame:
    branch = _branch_config(branch_name)
    results_dir = Path(branch["results_dir"])
    preferred, fallback = branch["metrics_files"]
    path = results_dir / preferred if (results_dir / preferred).is_file() else results_dir / fallback
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_summary(branch_name: str) -> dict[str, object]:
    branch = _branch_config(branch_name)
    results_dir = Path(branch["results_dir"])
    path = results_dir / branch["summary_file"]
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_confusion_matrix_image(model_name: str, results_dir: Path):
    path = results_dir / "confusion_matrices" / f"{model_name}.png"
    return path if path.is_file() else None


def main() -> None:
    st.set_page_config(page_title="ZESCO Sentiment Prototype", layout="wide")
    st.title("ZESCO Sentiment Analysis Prototype")
    st.caption("Baseline models trained on the final sarcasm-aware dataset.")

    with st.sidebar:
        st.header("Analysis Branch")
        branch_name = st.radio("Choose model path", list(BRANCHES.keys()), index=0)

    branch = _branch_config(branch_name)
    metrics = _load_metrics(branch_name)
    summary = _load_summary(branch_name)
    results_dir = Path(branch["results_dir"])
    models_dir = Path(branch["models_dir"])

    with st.sidebar:
        st.header("Dataset")
        st.write(f"Train rows: {summary.get('train_rows', 'n/a')}")
        st.write(f"Test rows: {summary.get('test_rows', 'n/a')}")
        st.write(f"Label column: {summary.get('label_column', 'corrected_label')}")
        st.write(f"Vectorizer: {summary.get('vectorizer_path', 'n/a')}")
        st.write(f"Model dir: {models_dir}")
        st.write(f"Results dir: {results_dir}")

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
                predictions = predict_texts([sample_text], models_dir=models_dir)
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
            image_path = _load_confusion_matrix_image(selected_model, results_dir)
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
