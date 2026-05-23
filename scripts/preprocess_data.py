"""Preview and run comment cleaning for the collected Facebook data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaner import (
    CleaningConfig,
    build_preview,
    clean_frame,
    load_comment_files,
)


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    """Write a CSV, falling back to a rerun file if the target is locked."""
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def _default_inputs() -> list[Path]:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    return sorted(raw_dir.glob("comments_post_*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the collected Facebook comments.")
    parser.add_argument(
        "--input",
        nargs="*",
        help="CSV files to clean. Defaults to all comments_post_*.csv files in data/raw.",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Keep only rows with at least one English word after the word-count filter.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum number of words required to keep a comment. Default is 4, which means > 3 words.",
    )
    parser.add_argument(
        "--stage1-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage1_3plus_english.csv"),
        help="Output CSV path for stage 1 cleaned data.",
    )
    parser.add_argument(
        "--stage2-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage2_emoji_stopword_lemma.csv"),
        help="Output CSV path for stage 2 processed data.",
    )
    parser.add_argument(
        "--strict-stage2-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage2_strict_english.csv"),
        help="Optional output CSV path for strict-English stage 2 data.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print stage counts without writing the cleaned file.",
    )
    parser.add_argument(
        "--strict-min-ratio",
        type=float,
        default=0.36,
        help="Minimum English-token ratio for the strict-English branch.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input] if args.input else _default_inputs()
    df = load_comment_files(input_paths)
    if df.empty:
        print("No input CSV files found.")
        return 1

    preview = build_preview(df)
    print("Cleaning preview:")
    print(f"  raw rows: {preview['raw_rows']}")
    print(f"  rows with more than 3 words: {preview['word_count_gt_3']}")
    print(f"  english rows after word filter: {preview['english_rows']}")
    print(f"  processed rows after stopwords/lemmatization: {preview['processed_rows']}")

    if args.preview_only:
        return 0

    stage1 = clean_frame(
        df,
        CleaningConfig(
            min_words=args.min_words,
            english_only=args.english_only,
            remove_emojis=False,
            remove_stopwords=False,
            lemmatize=False,
        ),
    )
    stage1 = stage1[
        [
            "comment_id",
            "text",
            "timestamp",
            "author_name",
            "source_url",
            "text_raw",
            "text_clean",
            "word_count",
            "english_word_count",
            "has_english_word",
        ]
    ].copy()
    stage1_output = Path(args.stage1_output)
    stage1_output.parent.mkdir(parents=True, exist_ok=True)
    stage1_saved_to = _safe_to_csv(stage1, stage1_output, rerun_suffix="rerun")
    print(f"Saved stage 1 comments to {stage1_saved_to} ({len(stage1)} rows)")

    stage2 = clean_frame(
        df,
        CleaningConfig(
            min_words=args.min_words,
            english_only=args.english_only,
            remove_emojis=True,
            remove_stopwords=True,
            lemmatize=True,
        ),
    )
    stage2 = stage2[["comment_id", "processed_text"]].copy()
    stage2_output = Path(args.stage2_output)
    stage2_output.parent.mkdir(parents=True, exist_ok=True)
    stage2_saved_to = _safe_to_csv(stage2, stage2_output, rerun_suffix="rerun")
    print(f"Saved stage 2 comments to {stage2_saved_to} ({len(stage2)} rows)")

    strict_stage2 = clean_frame(
        df,
        CleaningConfig(
            min_words=args.min_words,
            english_only=True,
            strict_english_only=True,
            strict_english_min_ratio=args.strict_min_ratio,
            strict_english_min_words=5,
            remove_emojis=True,
            remove_stopwords=True,
            lemmatize=True,
        ),
    )
    strict_stage2 = strict_stage2[["comment_id", "processed_text"]].copy()
    strict_stage2_output = Path(args.strict_stage2_output)
    strict_stage2_output.parent.mkdir(parents=True, exist_ok=True)
    strict_stage2_saved_to = _safe_to_csv(strict_stage2, strict_stage2_output, rerun_suffix="rerun")
    print(f"Saved strict-English stage 2 comments to {strict_stage2_saved_to} ({len(strict_stage2)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
