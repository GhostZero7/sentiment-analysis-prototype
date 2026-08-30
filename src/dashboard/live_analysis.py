"""Live Facebook URL analysis pipeline used by the Streamlit dashboard."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_collection.apify_client import MAX_COMMENTS_PER_URL, fetch_comments
from src.data_collection.url_utils import canonical_facebook_url, facebook_content_id
from src.models.predict import predict_texts
from src.preprocessing.cleaner import CleaningConfig, clean_frame
from src.preprocessing.relevance import assess_relevance
from src.sentiment.nrc_analyzer import get_nrc_scores
from src.sentiment.sarcasm import is_sarcastic
from src.sentiment.text_selection import select_sentiment_text
from src.sentiment.vader_analyzer import get_vader_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LIVE_RAW_DIR = DATA_DIR / "raw" / "url_fetches"
LIVE_LINKS_PATH = DATA_DIR / "raw" / "url_links.csv"
LIVE_RESULTS_DIR = DATA_DIR / "results" / "url_analyses"
LIVE_LABELED_DIR = DATA_DIR / "processed" / "labeled"
LIVE_HISTORY_PATH = LIVE_LABELED_DIR / "url_analysis_history.csv"
LIVE_TRAINING_CANDIDATES_PATH = LIVE_LABELED_DIR / "url_training_candidates.csv"
COMMENT_METADATA_COLUMNS = (
    "post_title",
    "collected_at",
    "first_seen_at",
    "last_seen_at",
)


@dataclass(frozen=True, slots=True)
class CachedComments:
    comments: list[dict[str, Any]]
    path: Path
    available_rows: int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _url_hash(url: str) -> str:
    return hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:12]


def _normalise_cached_frame(
    frame: pd.DataFrame,
    *,
    source_url: str,
    max_comments: int,
) -> tuple[list[dict[str, Any]], int]:
    if frame.empty or "text" not in frame.columns:
        return [], 0

    cached = frame.copy()
    cached["text"] = cached["text"].fillna("").astype(str).str.strip()
    cached = cached[cached["text"].ne("")].copy()
    if cached.empty:
        return [], 0

    if "comment_id" not in cached.columns:
        cached["comment_id"] = [f"cached-{index}" for index in range(len(cached))]
    if "timestamp" not in cached.columns:
        cached["timestamp"] = ""
    for column in COMMENT_METADATA_COLUMNS:
        if column not in cached.columns:
            cached[column] = ""
    cached["comment_id"] = cached["comment_id"].fillna("").astype(str)
    cached["timestamp"] = cached["timestamp"].fillna("").astype(str)
    for column in COMMENT_METADATA_COLUMNS:
        cached[column] = cached[column].fillna("").astype(str)
    cached["source_url"] = source_url
    cached = cached.drop_duplicates(subset=["comment_id", "text"], keep="first")
    available_rows = int(len(cached))
    selected = cached.head(max_comments)
    return (
        selected[
            ["comment_id", "text", "timestamp", "source_url", *COMMENT_METADATA_COLUMNS]
        ].to_dict("records"),
        available_rows,
    )


def _read_cached_file(
    path: Path,
    *,
    source_url: str,
    content_id: str,
    max_comments: int,
    filter_source: bool,
) -> CachedComments | None:
    try:
        frame = pd.read_csv(path, dtype={"comment_id": str})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None

    if filter_source:
        if "source_url" not in frame.columns:
            return None
        source_ids = frame["source_url"].fillna("").astype(str).map(facebook_content_id)
        frame = frame[source_ids.eq(content_id)].copy()

    comments, available_rows = _normalise_cached_frame(
        frame,
        source_url=source_url,
        max_comments=max_comments,
    )
    if not comments:
        return None
    return CachedComments(comments=comments, path=path, available_rows=available_rows)


def load_cached_comments(url: str, *, max_comments: int) -> CachedComments | None:
    """Load a previously fetched post before spending Apify tokens."""
    source_url = canonical_facebook_url(url)
    content_id = facebook_content_id(source_url)
    if not content_id:
        return None
    max_comments = min(max(int(max_comments), 1), MAX_COMMENTS_PER_URL)

    exact_path = RAW_DIR / f"comments_post_{content_id}.csv"
    if exact_path.is_file():
        cached = _read_cached_file(
            exact_path,
            source_url=source_url,
            content_id=content_id,
            max_comments=max_comments,
            filter_source=False,
        )
        if cached:
            return cached

    if LIVE_RAW_DIR.is_dir():
        live_files = sorted(
            LIVE_RAW_DIR.glob("comments_*.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in live_files:
            cached = _read_cached_file(
                path,
                source_url=source_url,
                content_id=content_id,
                max_comments=max_comments,
                filter_source=True,
            )
            if cached:
                return cached

    for path in (RAW_DIR / "all_comments.csv", RAW_DIR / "comments.csv"):
        if not path.is_file():
            continue
        cached = _read_cached_file(
            path,
            source_url=source_url,
            content_id=content_id,
            max_comments=max_comments,
            filter_source=True,
        )
        if cached:
            return cached
    return None


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


def _save_link_record(record: dict[str, Any]) -> pd.DataFrame:
    """Save a deduped registry of Facebook URLs analyzed through the dashboard."""
    output_path = LIVE_LINKS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([record])
    if output_path.is_file():
        existing = pd.read_csv(
            output_path,
            dtype={"source_url": str, "content_id": str},
        )
        if "source_url" in existing.columns:
            existing["source_url"] = existing["source_url"].fillna("").astype(str)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined["source_url"] = combined["source_url"].fillna("").astype(str)
    if "content_id" not in combined.columns:
        combined["content_id"] = ""
    combined["content_id"] = combined["content_id"].fillna("").astype(str)
    missing_ids = combined["content_id"].eq("")
    combined.loc[missing_ids, "content_id"] = combined.loc[
        missing_ids, "source_url"
    ].map(facebook_content_id).fillna("")
    combined["_source_key"] = combined["content_id"].where(
        combined["content_id"].ne(""),
        combined["source_url"],
    )
    combined = (
        combined.sort_values("last_analyzed_at")
        .drop_duplicates(subset=["_source_key"], keep="last")
        .drop(columns=["_source_key"])
    )
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


def _comment_key(comment: dict[str, Any]) -> tuple[str, ...]:
    comment_id = str(comment.get("comment_id", "")).strip()
    if comment_id:
        return ("id", comment_id)
    return (
        "content",
        str(comment.get("text", "")).strip().casefold(),
        str(comment.get("timestamp", "")).strip(),
    )


def _merge_comment_record(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(previous)
    for key, value in current.items():
        if value is not None and str(value).strip():
            merged[key] = value

    previous_first_seen = str(
        previous.get("first_seen_at") or previous.get("collected_at") or ""
    ).strip()
    current_seen = str(
        current.get("last_seen_at") or current.get("collected_at") or ""
    ).strip()
    merged["first_seen_at"] = previous_first_seen or current_seen
    merged["last_seen_at"] = current_seen or str(
        previous.get("last_seen_at") or previous.get("collected_at") or ""
    ).strip()
    return merged


def _initial_seen_record(comment: dict[str, Any]) -> dict[str, Any]:
    record = dict(comment)
    seen_at = str(record.get("collected_at", "")).strip()
    if not str(record.get("first_seen_at", "")).strip():
        record["first_seen_at"] = seen_at
    if not str(record.get("last_seen_at", "")).strip():
        record["last_seen_at"] = seen_at
    return record


def merge_tracked_comments(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Merge a refreshed discussion with its cache and retain the newest requested rows."""
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for comment in previous:
        key = _comment_key(comment)
        if key not in by_key:
            order.append(key)
            by_key[key] = _initial_seen_record(comment)
        else:
            by_key[key] = _merge_comment_record(by_key[key], comment)

    previous_keys = set(by_key)
    for comment in current:
        key = _comment_key(comment)
        if key not in by_key:
            order.append(key)
            by_key[key] = _initial_seen_record(comment)
        else:
            by_key[key] = _merge_comment_record(by_key[key], comment)

    records = [by_key[key] for key in order]
    if records:
        frame = pd.DataFrame(records)
        frame["_source_order"] = range(len(frame))
        frame["_comment_time"] = pd.to_datetime(
            frame.get("timestamp", pd.Series("", index=frame.index)),
            errors="coerce",
            utc=True,
        )
        frame = frame.sort_values(
            ["_comment_time", "_source_order"],
            kind="stable",
            na_position="first",
        ).tail(max(1, min(int(limit), MAX_COMMENTS_PER_URL)))
        records = frame.drop(columns=["_source_order", "_comment_time"]).to_dict("records")

    title = next(
        (
            str(record.get("post_title", "")).strip()
            for record in records
            if str(record.get("post_title", "")).strip()
        ),
        "",
    )
    if title:
        for record in records:
            record["post_title"] = title
    return records, len({_comment_key(row) for row in current} - previous_keys)


def _comments_up_to(
    comments: list[dict[str, Any]],
    cutoff: datetime,
) -> list[dict[str, Any]]:
    cutoff_value = pd.Timestamp(cutoff)
    included: list[dict[str, Any]] = []
    for comment in comments:
        timestamp = pd.to_datetime(comment.get("timestamp", ""), errors="coerce", utc=True)
        if pd.isna(timestamp) or timestamp <= cutoff_value:
            included.append(comment)
    return included


def _post_title(comments: list[dict[str, Any]]) -> str:
    return next(
        (
            str(comment.get("post_title", "")).strip()
            for comment in comments
            if str(comment.get("post_title", "")).strip()
        ),
        "",
    )


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
            min_words=2,
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
        text_for_rules = select_sentiment_text(row)
        sarcastic = bool(is_sarcastic(text_for_rules))
        relevance = assess_relevance(text_for_rules)
        scores = _score_comment(text_for_rules, sarcastic=sarcastic)
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


def save_post_cache(comments: list[dict[str, Any]], *, source_url: str) -> str:
    """Save a canonical per-post cache for token-free repeat analysis."""
    content_id = facebook_content_id(source_url)
    if not content_id:
        return ""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"comments_post_{content_id}.csv"
    pd.DataFrame(comments).to_csv(cache_path, index=False)
    return str(cache_path)


def analyze_facebook_url(
    url: str,
    *,
    models_dir: str | Path,
    max_comments: int = 300,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyze a URL, optionally refreshing and merging it for trend tracking."""
    max_comments = min(max(int(max_comments), 1), MAX_COMMENTS_PER_URL)
    source_url = canonical_facebook_url(url)
    cutoff = datetime.now(timezone.utc)
    cached = load_cached_comments(source_url, max_comments=max_comments)
    new_comment_count = 0
    if cached and not force_refresh:
        comments = cached.comments
        raw_path = str(cached.path)
        cache_path = str(cached.path)
        collection_source = "local_cache"
        cache_hit = True
        cache_rows_available = cached.available_rows
    else:
        fetched = fetch_comments(source_url, max_comments=max_comments)
        raw_path = save_raw_fetch(fetched, source_url=source_url)
        previous = cached.comments if cached else []
        comments, new_comment_count = merge_tracked_comments(
            previous,
            fetched,
            limit=max_comments,
        )
        cache_path = save_post_cache(comments, source_url=source_url)
        collection_source = "apify_refresh" if force_refresh and cached else "apify"
        cache_hit = False
        cache_rows_available = cached.available_rows if cached else 0
    comments = _comments_up_to(comments, cutoff)
    analyzed = analyze_comments_frame(
        comments,
        source_url=source_url,
        models_dir=models_dir,
        batch_id=f"{_utc_stamp()}-{_url_hash(source_url)}",
    )
    save_summary = save_live_analysis(analyzed, source_url=source_url)
    save_summary["raw_fetch_path"] = raw_path
    save_summary["cache_path"] = cache_path
    save_summary["cache_hit"] = cache_hit
    save_summary["cache_rows_available"] = cache_rows_available
    save_summary["collection_source"] = collection_source
    save_summary["fetched_rows"] = len(comments)
    save_summary["new_comment_rows"] = new_comment_count
    save_summary["analyzed_rows"] = int(len(analyzed))
    save_summary["relevant_rows"] = int(analyzed["is_relevant"].fillna(False).sum()) if not analyzed.empty else 0
    save_summary["source_url"] = source_url
    save_summary["requested_comment_limit"] = max_comments
    save_summary["post_title"] = _post_title(comments)
    save_summary["trend_cutoff"] = cutoff.isoformat()
    save_summary["tracking_refresh"] = bool(force_refresh)
    link_registry = _save_link_record(
        {
            "source_url": source_url,
            "content_id": facebook_content_id(source_url) or "",
            "post_title": save_summary["post_title"],
            "url_hash": _url_hash(source_url),
            "last_analyzed_at": datetime.now(timezone.utc).isoformat(),
            "collection_source": collection_source,
            "cache_hit": cache_hit,
            "cache_path": cache_path,
            "cache_rows_available": cache_rows_available,
            "new_comment_rows": new_comment_count,
            "requested_comment_limit": max_comments,
            "fetched_rows": save_summary["fetched_rows"],
            "analyzed_rows": save_summary["analyzed_rows"],
            "relevant_rows": save_summary["relevant_rows"],
            "raw_fetch_path": raw_path,
            "single_analysis_path": save_summary["single_analysis_path"],
            "history_path": save_summary["history_path"],
            "training_candidates_path": save_summary["training_candidates_path"],
        }
    )
    save_summary["links_path"] = str(LIVE_LINKS_PATH)
    save_summary["saved_link_rows"] = int(len(link_registry))
    return analyzed, save_summary
