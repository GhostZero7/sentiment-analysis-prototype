"""Live Facebook URL analysis pipeline used by the Streamlit dashboard."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_collection.apify_client import fetch_comments
from src.models.predict import predict_texts
from src.preprocessing.cleaner import CleaningConfig, clean_frame
from src.preprocessing.relevance import assess_relevance
from src.sentiment.nrc_analyzer import get_nrc_scores
from src.sentiment.sarcasm import is_sarcastic
from src.sentiment.vader_analyzer import get_vader_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LIVE_RAW_DIR = DATA_DIR / "raw" / "url_fetches"
LIVE_RESULTS_DIR = DATA_DIR / "results" / "url_analyses"
LIVE_LABELED_DIR = DATA_DIR / "processed" / "labeled"
LIVE_HISTORY_PATH = LIVE_LABELED_DIR / "url_analysis_history.csv"
LIVE_TRAINING_CANDIDATES_PATH = LIVE_LABELED_DIR / "url_training_candidates.csv"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _url_hash(url: str) -> str:
    return hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:12]


def _dedupe_key_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ["source_url", "comment_id", "processed_text"] if column in frame.columns]


def _append_deduped(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Append rows to a CSV while preventing duplicate URL/comment/text rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()

    key_columns = _dedupe_key_columns(combined)
    if key_columns:
        for column in key_columns:
            combined[column] = combined[column].fillna("").astype(str)
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined.to_csv(output_path, index=False)
    return combined


def _score_comment(text: str, *, sarcastic: bool) -> dict[str, Any]:
    vader_scores = get_vader_scores(text)
    nrc_scores = get_nrc_scores(text, is_sarcastic=sarcastic)
    if sarcastic:
        vader_scores["label"] = "negative"
        vader_scores["corrected_label"] = "negative"
    return {
        "is_sarcastic": sarcastic,
        **vader_scores,
        **nrc_scores,
    }


def _prepare_comments_frame(comments: list[dict[str, Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(comments)
    if raw.empty:
        return raw
    raw["text"] = raw["text"].fillna("").astype(str)
    raw = raw[raw["text"].str.strip().ne("")].copy()
    return raw.drop_duplicates(subset=[column for column in ["comment_id", "text", "source_url"] if column in raw.columns])


def analyze_comments_frame(
    comments: list[dict[str, Any]],
    *,
    source_url: str,
    models_dir: str | Path,
    batch_id: str | None = None,
) -> pd.DataFrame:
    """Run preprocessing, relevance, sentiment, emotion, and ML predictions on comments."""
    raw = _prepare_comments_frame(comments)
    if raw.empty:
        return pd.DataFrame()

    cleaned = clean_frame(
        raw,
        CleaningConfig(
            min_words=4,
            english_only=False,
            remove_emojis=True,
            remove_stopwords=True,
            lemmatize=True,
        ),
    )
    if cleaned.empty:
        return pd.DataFrame()

    scored_rows: list[dict[str, Any]] = []
    for _, row in cleaned.iterrows():
        text_for_rules = " ".join(
            str(value)
            for value in [row.get("text_raw", ""), row.get("text_clean", ""), row.get("processed_text", "")]
            if pd.notna(value)
        )
        sarcastic = bool(is_sarcastic(text_for_rules))
        relevance = assess_relevance(text_for_rules)
        scores = _score_comment(str(row.get("processed_text", "")), sarcastic=sarcastic)
        scored_rows.append({**relevance, **scores})

    analyzed = pd.concat([cleaned.reset_index(drop=True), pd.DataFrame(scored_rows)], axis=1)
    predictions = predict_texts(analyzed["processed_text"].fillna("").astype(str).tolist(), models_dir=models_dir)
    analyzed = pd.concat([analyzed.reset_index(drop=True), predictions.drop(columns=["text"])], axis=1)
    analyzed["analysis_batch_id"] = batch_id or f"{_utc_stamp()}-{_url_hash(source_url)}"
    analyzed["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
    analyzed["source_url"] = source_url
    return analyzed


def save_live_analysis(analyzed: pd.DataFrame, *, source_url: str) -> dict[str, Any]:
    """Save one URL analysis and append it to cumulative future-training files."""
    if analyzed.empty:
        return {
            "single_analysis_path": "",
            "history_path": str(LIVE_HISTORY_PATH),
            "training_candidates_path": str(LIVE_TRAINING_CANDIDATES_PATH),
            "history_rows": 0,
            "training_candidate_rows": 0,
        }

    LIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    single_path = LIVE_RESULTS_DIR / f"analysis_{_url_hash(source_url)}_{_utc_stamp()}.csv"
    analyzed.to_csv(single_path, index=False)

    history = _append_deduped(analyzed, LIVE_HISTORY_PATH)
    relevant = analyzed[analyzed["is_relevant"].fillna(False).astype(bool)].copy()
    candidates = _append_deduped(relevant, LIVE_TRAINING_CANDIDATES_PATH)

    return {
        "single_analysis_path": str(single_path),
        "history_path": str(LIVE_HISTORY_PATH),
        "training_candidates_path": str(LIVE_TRAINING_CANDIDATES_PATH),
        "history_rows": int(len(history)),
        "training_candidate_rows": int(len(candidates)),
    }


def save_raw_fetch(comments: list[dict[str, Any]], *, source_url: str) -> str:
    """Save the raw fetched comments for audit before preprocessing filters run."""
    LIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = LIVE_RAW_DIR / f"comments_{_url_hash(source_url)}_{_utc_stamp()}.csv"
    pd.DataFrame(comments).to_csv(raw_path, index=False)
    return str(raw_path)


def analyze_facebook_url(
    url: str,
    *,
    models_dir: str | Path,
    max_comments: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch a Facebook URL, analyze its comments, and save cumulative outputs."""
    comments = fetch_comments(url, max_comments=max_comments)
    raw_path = save_raw_fetch(comments, source_url=url)
    analyzed = analyze_comments_frame(
        comments,
        source_url=url,
        models_dir=models_dir,
        batch_id=f"{_utc_stamp()}-{_url_hash(url)}",
    )
    save_summary = save_live_analysis(analyzed, source_url=url)
    save_summary["raw_fetch_path"] = raw_path
    save_summary["fetched_rows"] = len(comments)
    save_summary["analyzed_rows"] = int(len(analyzed))
    save_summary["relevant_rows"] = int(analyzed["is_relevant"].fillna(False).sum()) if not analyzed.empty else 0
    return analyzed, save_summary
