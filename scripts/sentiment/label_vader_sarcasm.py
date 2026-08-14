"""Label cleaned comments with sarcasm-aware VADER sentiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.nrc_analyzer import get_nrc_scores
from src.sentiment.sarcasm import is_sarcastic
from src.sentiment.text_selection import select_sentiment_text
from src.sentiment.vader_analyzer import get_vader_scores

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage2_emoji_stopword_lemma.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage1_3plus_english.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label.csv"
DEFAULT_FAILURES = PROJECT_ROOT / "data" / "results" / "label_failures" / "final_label_failures.csv"


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def _fallback_vader_result() -> dict[str, object]:
    return {
        "raw_compound": 0.0,
        "raw_label": "neutral",
        "compound": 0.0,
        "pos": 0.0,
        "neu": 1.0,
        "neg": 0.0,
        "label": "neutral",
        "corrected_compound": 0.0,
        "corrected_label": "neutral",
        "local_correction_applied": False,
        "local_correction_score": 0.0,
        "local_correction_terms": [],
    }


def _fallback_nrc_result() -> dict[str, object]:
    return {
        "nrc_token_count": 0,
        "nrc_matched_token_count": 0,
        "nrc_base_anger": 0.0,
        "nrc_base_anticipation": 0.0,
        "nrc_base_fear": 0.0,
        "nrc_base_trust": 0.0,
        "nrc_base_sadness": 0.0,
        "nrc_base_hope": 0.0,
        "nrc_base_frustration": 0.0,
        "local_emotion_applied": False,
        "local_emotion_terms": "",
        "local_emotion_match_count": 0,
        "local_anger_score": 0.0,
        "local_fear_score": 0.0,
        "local_trust_score": 0.0,
        "local_hope_score": 0.0,
        "local_sadness_score": 0.0,
        "local_frustration_score": 0.0,
        "nrc_anger": 0.0,
        "nrc_anticipation": 0.0,
        "nrc_fear": 0.0,
        "nrc_trust": 0.0,
        "nrc_sadness": 0.0,
        "nrc_hope": 0.0,
        "nrc_frustration": 0.0,
    }


def _process_comment(text: str) -> tuple[dict[str, object], dict[str, object] | None]:
    """Score a single comment and return a labeled row plus optional failure log."""
    try:
        sarcastic = False
        try:
            sarcastic = bool(is_sarcastic(text))
        except Exception:
            sarcastic = False

        vader_scores = get_vader_scores(text)
        nrc_scores = get_nrc_scores(text, is_sarcastic=sarcastic)
        row = {
            "is_sarcastic": sarcastic,
            **vader_scores,
            **nrc_scores,
        }
        if row["is_sarcastic"]:
            if row.get("label") != "negative":
                row["label"] = "negative"
            if row.get("corrected_label") != "negative":
                row["corrected_label"] = "negative"
        return row, None
    except Exception as exc:  # pragma: no cover - defensive fallback
        failure = {
            "analysis_failed": True,
            "analysis_error": str(exc),
        }
        return {
            "is_sarcastic": False,
            **failure,
            **_fallback_vader_result(),
            **_fallback_nrc_result(),
        }, failure


def _sentiment_texts(df: pd.DataFrame, metadata_path: Path) -> list[str]:
    """Match processed rows to original text without changing the output schema."""
    if any(column in df.columns for column in ("text_raw", "text", "text_clean")):
        return [select_sentiment_text(row) for _, row in df.iterrows()]

    if "comment_id" not in df.columns or not metadata_path.is_file():
        return df["processed_text"].fillna("").astype(str).tolist()

    metadata = pd.read_csv(metadata_path, dtype={"comment_id": str})
    if "comment_id" not in metadata.columns:
        return df["processed_text"].fillna("").astype(str).tolist()

    readable_by_id = {
        str(row["comment_id"]): select_sentiment_text(row)
        for _, row in metadata.iterrows()
    }
    return [
        readable_by_id.get(str(row["comment_id"]), "") or str(row.get("processed_text", ""))
        for _, row in df.iterrows()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply sarcasm detection before VADER sentiment labeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    args = parser.parse_args()

    input_path, output_path, failures_path = args.input, args.output, args.failures
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    if "processed_text" not in df.columns:
        print("Input file must contain a 'processed_text' column.")
        return 1

    texts = _sentiment_texts(df, args.metadata)
    total = len(texts)
    labeled_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, text in enumerate(texts, start=1):
        row, failure = _process_comment(text)
        labeled_rows.append(row)
        if failure is not None:
            failures.append(
                {
                    "row_index": index,
                    "comment_id": df.iloc[index - 1].get("comment_id"),
                    "processed_text": text,
                    "analysis_error": failure["analysis_error"],
                }
            )
        if index == 1 or index % 100 == 0 or index == total:
            print(f"Processed {index}/{total} comments; failures: {len(failures)}")

    labeled = pd.concat([df.reset_index(drop=True), pd.DataFrame(labeled_rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_to = _safe_to_csv(labeled, output_path, rerun_suffix="rerun")
    sarcastic_count = int(labeled["is_sarcastic"].sum())
    print(f"Saved sarcasm-aware VADER comments to {saved_to} ({len(labeled)} rows)")
    print(f"Sarcastic comments detected: {sarcastic_count}")

    if failures:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Saved failed comment log to {failures_path} ({len(failures)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
