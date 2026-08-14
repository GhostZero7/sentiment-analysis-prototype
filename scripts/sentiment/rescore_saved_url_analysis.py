"""Refresh saved URL-analysis sentiment fields using readable comment text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_texts
from src.sentiment.nrc_analyzer import get_nrc_scores
from src.sentiment.sarcasm import is_sarcastic
from src.sentiment.text_selection import select_sentiment_text
from src.sentiment.vader_analyzer import get_vader_scores


DEFAULT_PATHS = (
    PROJECT_ROOT / "data" / "processed" / "labeled" / "url_analysis_history.csv",
    PROJECT_ROOT / "data" / "processed" / "labeled" / "url_training_candidates.csv",
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "url_analyses"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "models"


def rescore_frame(frame: pd.DataFrame, *, models_dir: Path) -> pd.DataFrame:
    """Overwrite rule-based and model scores while preserving collection metadata."""
    rescored = frame.copy()
    score_rows: list[dict[str, object]] = []
    for _, row in rescored.iterrows():
        text = select_sentiment_text(row)
        sarcastic = bool(is_sarcastic(text))
        vader = get_vader_scores(text)
        if sarcastic:
            vader["label"] = "negative"
            vader["corrected_label"] = "negative"
        score_rows.append(
            {
                "is_sarcastic": sarcastic,
                **vader,
                **get_nrc_scores(text, is_sarcastic=sarcastic),
            }
        )

    score_frame = pd.DataFrame(score_rows, index=rescored.index)
    for column in score_frame.columns:
        rescored[column] = score_frame[column]

    if "processed_text" in rescored.columns and models_dir.is_dir():
        predictions = predict_texts(
            rescored["processed_text"].fillna("").astype(str).tolist(),
            models_dir=models_dir,
        )
        for column in predictions.columns:
            if column != "text":
                rescored[column] = predictions[column].values
    return rescored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    paths = list(args.paths) or list(DEFAULT_PATHS)
    if not args.paths and DEFAULT_RESULTS_DIR.is_dir():
        paths.extend(sorted(DEFAULT_RESULTS_DIR.glob("analysis_*.csv")))

    refreshed = 0
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        before = frame.get("corrected_label", pd.Series(dtype=str)).value_counts().to_dict()
        updated = rescore_frame(frame, models_dir=args.models_dir)
        updated.to_csv(path, index=False)
        after = updated.get("corrected_label", pd.Series(dtype=str)).value_counts().to_dict()
        print(f"Refreshed {path} ({len(updated)} rows): {before} -> {after}")
        refreshed += 1

    if not refreshed:
        print("No saved URL-analysis CSV files were found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
