"""Label the stage 2 cleaned comments with VADER sentiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.vader_analyzer import get_vader_scores
from src.preprocessing.anonymiser import anonymise_text


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    """Write a CSV, falling back to a rerun file if the target is locked."""
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Add VADER scores to the processed comments file.")
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Use the strict-English stage 2 file as input and write an English-only labeled output.",
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage2_emoji_stopword_lemma.csv"),
        help="Input CSV path from the preprocessing stage.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "comments_labeled_vader.csv"),
        help="Output CSV path for the VADER-labeled file.",
    )
    args = parser.parse_args()

    if args.english_only:
        args.input = str(PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage2_strict_english.csv")
        args.output = str(PROJECT_ROOT / "data" / "processed" / "labeled" / "comments_labeled_vader_english_only.csv")

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    if "processed_text" not in df.columns:
        print("Input file must contain a 'processed_text' column.")
        return 1

    df = df.copy()
    df["processed_text"] = df["processed_text"].fillna("").map(anonymise_text)
    scores = df["processed_text"].map(get_vader_scores)
    scores_df = pd.json_normalize(scores)
    labeled = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_to = _safe_to_csv(labeled, output_path, rerun_suffix="rerun")
    print(f"Saved VADER-labeled comments to {saved_to} ({len(labeled)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
