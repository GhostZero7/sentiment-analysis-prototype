"""Detect sarcasm in ZESCO comments and optionally flip sentiment labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.sarcasm import annotate_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect sarcasm in comments CSV files.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw" / "all_comments.csv"),
        help="Input CSV path containing a text column.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "all_comments_with_sarcasm.csv"),
        help="Output CSV path for the sarcasm-annotated file.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column to analyze.",
    )
    parser.add_argument(
        "--label-column",
        default="corrected_label",
        help="Label column to flip to negative when sarcasm is detected.",
    )
    args = parser.parse_args()

    try:
        annotated = annotate_csv(
            args.input,
            args.output,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    sarcastic_count = int(annotated["is_sarcastic"].sum())
    print(f"Saved sarcasm-annotated comments to {args.output} ({len(annotated)} rows)")
    print(f"Sarcastic comments detected: {sarcastic_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
