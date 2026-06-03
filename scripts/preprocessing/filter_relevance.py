"""Split labeled comments into relevance review files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.relevance import assess_relevance


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "cleaned" / "comments_stage1_3plus_english.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "relevance"


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def _load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    metadata = pd.read_csv(path)
    keep_columns = [
        "comment_id",
        "text",
        "timestamp",
        "source_url",
        "text_clean",
        "word_count",
    ]
    present = [column for column in keep_columns if column in metadata.columns]
    return metadata[present].drop_duplicates(subset=["comment_id"])


def _comment_text_for_relevance(row: pd.Series) -> str:
    parts = [
        row.get("text", ""),
        row.get("text_clean", ""),
        row.get("processed_text", ""),
    ]
    return " ".join(str(part) for part in parts if pd.notna(part))


def split_relevance(input_path: Path, metadata_path: Path, output_dir: Path) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    labeled = pd.read_csv(input_path)
    if "comment_id" not in labeled.columns:
        raise ValueError("Input file must contain a 'comment_id' column.")

    metadata = _load_metadata(metadata_path)
    if not metadata.empty:
        review = metadata.merge(labeled, on="comment_id", how="right")
    else:
        review = labeled.copy()

    relevance_rows = []
    total = len(review)
    for index, row in review.iterrows():
        relevance_rows.append(assess_relevance(_comment_text_for_relevance(row)))
        current = index + 1
        if current == 1 or current % 250 == 0 or current == total:
            print(f"Scored relevance for {current}/{total} comments")

    review = pd.concat([review.reset_index(drop=True), pd.DataFrame(relevance_rows)], axis=1)
    relevant = review[review["is_relevant"]].copy()
    irrelevant = review[~review["is_relevant"]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = _safe_to_csv(review, output_dir / "all_comments_relevance.csv", rerun_suffix="rerun")
    relevant_path = _safe_to_csv(relevant, output_dir / "relevant_comments.csv", rerun_suffix="rerun")
    irrelevant_path = _safe_to_csv(irrelevant, output_dir / "irrelevant_comments.csv", rerun_suffix="rerun")

    return {
        "total_rows": int(len(review)),
        "relevant_rows": int(len(relevant)),
        "irrelevant_rows": int(len(irrelevant)),
        "all_path": str(all_path),
        "relevant_path": str(relevant_path),
        "irrelevant_path": str(irrelevant_path),
        "reason_counts": review["relevance_reason"].value_counts().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create relevance review files from final labeled comments.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Final labeled CSV to score.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Stage 1 cleaned CSV with readable text.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder for relevance review files.")
    args = parser.parse_args()

    try:
        summary = split_relevance(Path(args.input), Path(args.metadata), Path(args.output_dir))
    except Exception as exc:
        print(f"Relevance split failed: {exc}")
        return 1

    print(f"Saved all relevance-scored comments to {summary['all_path']}")
    print(f"Saved relevant comments to {summary['relevant_path']} ({summary['relevant_rows']} rows)")
    print(f"Saved irrelevant comments to {summary['irrelevant_path']} ({summary['irrelevant_rows']} rows)")
    print("Relevance reasons:")
    for reason, count in summary["reason_counts"].items():
        print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
