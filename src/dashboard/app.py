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
        "labeled_file": DATA_DIR / "processed" / "labeled" / "final_label.csv",
        "test_file": DATA_DIR / "processed" / "labeled" / "testing" / "final_label_test.csv",
        "summary_file": "training_summary.json",
        "metrics_files": ("evaluation_summary_rerun.csv", "evaluation_summary.csv"),
    },
    "RoBERTa-labeled branch": {
        "results_dir": RESULTS_DIR / "roberta",
        "models_dir": MODELS_DIR / "roberta",
        "labeled_file": DATA_DIR / "processed" / "labeled" / "final_label_roberta.csv",
        "test_file": DATA_DIR / "processed" / "labeled" / "testing" / "final_label_roberta_test.csv",
        "summary_file": "training_summary.json",
        "metrics_files": ("evaluation_summary_rerun.csv", "evaluation_summary.csv"),
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


def _load_labeled_data(branch_name: str) -> pd.DataFrame:
    branch = _branch_config(branch_name)
    labeled_file = Path(branch["labeled_file"])
    if not labeled_file.is_file():
        return pd.DataFrame()
    return pd.read_csv(labeled_file)


def _count_rows(path: Path) -> int | str:
    if not path.is_file():
        return "n/a"
    return len(pd.read_csv(path))


def _load_confusion_matrix_image(model_name: str, results_dir: Path):
    path = results_dir / "confusion_matrices" / f"{model_name}.png"
    return path if path.is_file() else None


def _emotion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    emotion_columns = [
        "nrc_anger",
        "nrc_anticipation",
        "nrc_fear",
        "nrc_trust",
        "nrc_sadness",
        "nrc_hope",
        "nrc_frustration",
    ]
    present = [column for column in emotion_columns if column in frame.columns]
    if not present:
        return pd.DataFrame()
    summary = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0).mean().sort_values(ascending=False)
    return summary.rename_axis("emotion").reset_index(name="average_score")


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
    labeled_data = _load_labeled_data(branch_name)
    results_dir = Path(branch["results_dir"])
    models_dir = Path(branch["models_dir"])
    test_file = Path(branch["test_file"])

    with st.sidebar:
        st.header("Dataset")
        st.write(f"Train rows: {summary.get('train_rows', 'n/a')}")
        st.write(f"Test rows: {_count_rows(test_file)}")
        st.write(f"Label column: {summary.get('label_column', 'corrected_label')}")
        st.write(f"Vectorizer: {summary.get('vectorizer_path', 'n/a')}")
        st.write(f"Model dir: {models_dir}")
        st.write(f"Results dir: {results_dir}")
        st.write(f"Labeled rows: {len(labeled_data) if not labeled_data.empty else 'n/a'}")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Model Metrics")
        if metrics.empty:
            st.info("No evaluation summary found yet. Run the matching evaluate_models.py script.")
        else:
            st.dataframe(metrics, use_container_width=True)

        st.subheader("Emotion Index")
        emotion_summary = _emotion_summary(labeled_data)
        if emotion_summary.empty:
            st.info("No NRC emotion columns found for this branch.")
        else:
            st.dataframe(emotion_summary, use_container_width=True)
            chart_data = emotion_summary.set_index("emotion")[["average_score"]]
            st.bar_chart(chart_data)

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
            st.info("Run the matching evaluate_models.py script to generate confusion matrices.")
        else:
            model_names = metrics["model"].tolist() if "model" in metrics.columns else []
            selected_model = st.selectbox("Choose a model", model_names or ["naive_bayes", "logistic_regression", "svm"])
            image_path = _load_confusion_matrix_image(selected_model, results_dir)
            if image_path is not None:
                st.image(str(image_path), caption=f"{selected_model} confusion matrix", use_container_width=True)
            else:
                st.warning("Confusion matrix image not found. Run `scripts/models/vader/evaluate_models.py` first.")

        st.subheader("Training Summary")
        if summary:
            st.json(summary)
        else:
            st.info("Training summary not found.")

        if not labeled_data.empty and "is_sarcastic" in labeled_data.columns:
            st.subheader("Corpus Snapshot")
            sarcasm_count = int(labeled_data["is_sarcastic"].fillna(False).sum())
            st.metric("Sarcastic comments", sarcasm_count)
            emotion_cols = [c for c in labeled_data.columns if c.startswith("nrc_")]
            st.write(f"Emotion columns: {', '.join(emotion_cols)}")


if __name__ == "__main__":
    main()
