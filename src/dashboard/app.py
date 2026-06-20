"""Streamlit dashboard for the sentiment analysis prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.live_analysis import analyze_facebook_url
from src.models.predict import predict_texts


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"
BRANCHES = {
    "VADER baseline": {
        "results_dir": RESULTS_DIR,
        "models_dir": MODELS_DIR,
        "labeled_file": DATA_DIR / "processed" / "labeled" / "final_label_relevant.csv",
        "test_file": DATA_DIR / "processed" / "labeled" / "testing" / "final_label_test.csv",
        "summary_file": "training_summary.json",
        "metrics_files": ("evaluation_summary_rerun.csv", "evaluation_summary.csv"),
    },
    "RoBERTa-labeled branch": {
        "results_dir": RESULTS_DIR / "roberta",
        "models_dir": MODELS_DIR / "roberta",
        "labeled_file": DATA_DIR / "processed" / "labeled" / "final_label_roberta_relevant.csv",
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


def _prediction_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = [column for column in frame.columns if column.endswith("_prediction")]
    if not prediction_columns:
        return pd.DataFrame()
    rows = []
    for column in prediction_columns:
        rows.append(
            {
                "model": column.replace("_prediction", ""),
                "top_prediction": frame[column].mode().iloc[0] if not frame[column].mode().empty else "n/a",
                "positive": int(frame[column].eq("positive").sum()),
                "neutral": int(frame[column].eq("neutral").sum()),
                "negative": int(frame[column].eq("negative").sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(page_title="ZESCO Sentiment Prototype", layout="wide")
    st.title("ZESCO Sentiment Analysis Prototype")
    st.caption("Analyze saved datasets or fetch and analyze a public Facebook post URL.")

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
        st.subheader("Analyze Facebook URL")
        facebook_url = st.text_input("Paste a public Facebook post/comment URL")
        max_comments = st.number_input("Maximum comments to fetch", min_value=10, max_value=5000, value=500, step=50)
        if st.button("Fetch, analyze, and save URL"):
            if not facebook_url.strip():
                st.warning("Paste a Facebook URL first.")
            else:
                with st.spinner("Fetching comments with Apify, analyzing text, and saving results..."):
                    try:
                        live_results, save_summary = analyze_facebook_url(
                            facebook_url.strip(),
                            models_dir=models_dir,
                            max_comments=int(max_comments),
                        )
                        st.session_state["latest_live_results"] = live_results
                        st.session_state["latest_live_summary"] = save_summary
                    except Exception as exc:
                        st.error(f"URL analysis failed: {exc}")

        latest_live_results = st.session_state.get("latest_live_results")
        latest_live_summary = st.session_state.get("latest_live_summary")
        if isinstance(latest_live_results, pd.DataFrame) and latest_live_summary:
            st.success(
                f"Fetched {latest_live_summary['fetched_rows']} comments; "
                f"analyzed {latest_live_summary['analyzed_rows']}; "
                f"relevant {latest_live_summary['relevant_rows']}."
            )
            metric_cols = st.columns(4)
            metric_cols[0].metric("Fetched", latest_live_summary["fetched_rows"])
            metric_cols[1].metric("Analyzed", latest_live_summary["analyzed_rows"])
            metric_cols[2].metric("Relevant", latest_live_summary["relevant_rows"])
            metric_cols[3].metric("Cumulative candidates", latest_live_summary["training_candidate_rows"])
            st.write(f"Saved analysis file: `{latest_live_summary['single_analysis_path']}`")
            st.write(f"Raw fetched comments: `{latest_live_summary['raw_fetch_path']}`")
            st.write(f"Saved link registry: `{latest_live_summary['links_path']}`")
            st.write(f"Cumulative history: `{latest_live_summary['history_path']}`")
            st.write(f"Future training candidates: `{latest_live_summary['training_candidates_path']}`")

            if "corrected_label" in latest_live_results.columns:
                st.write("VADER/local/sarcasm label counts")
                st.bar_chart(latest_live_results["corrected_label"].value_counts())

            live_emotions = _emotion_summary(latest_live_results)
            if not live_emotions.empty:
                st.write("Live URL emotion index")
                st.bar_chart(live_emotions.set_index("emotion")[["average_score"]])

            consensus = _prediction_consensus(latest_live_results)
            if not consensus.empty:
                st.write("Model prediction counts")
                st.dataframe(consensus, use_container_width=True)

            preview_columns = [
                column
                for column in [
                    "comment_id",
                    "text_raw",
                    "processed_text",
                    "is_relevant",
                    "corrected_label",
                    "is_sarcastic",
                    "local_emotion_terms",
                    "naive_bayes_prediction",
                    "logistic_regression_prediction",
                    "svm_prediction",
                ]
                if column in latest_live_results.columns
            ]
            st.dataframe(latest_live_results[preview_columns].head(50), use_container_width=True)
            st.download_button(
                "Download latest analyzed URL CSV",
                data=latest_live_results.to_csv(index=False).encode("utf-8"),
                file_name="latest_url_analysis.csv",
                mime="text/csv",
            )

        st.subheader("Model Metrics")
        if metrics.empty:
            st.info("No evaluation summary found yet. Run the matching evaluate_models.py script.")
        else:
            st.dataframe(metrics, use_container_width=True)

        st.subheader("Combined NRC + Local Emotion Index")
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
