"""Generate RoBERTa sentiment labels for the alternate training path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment.roberta_analyzer import DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME, label_dataframe


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the final dataset with RoBERTa sentiment predictions.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label.csv"),
        help="Input sarcasm-aware labeled CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label_roberta.csv"),
        help="Output CSV with RoBERTa labels.",
    )
    parser.add_argument(
        "--text-column",
        default="processed_text",
        help="Text column to label.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face RoBERTa sentiment model name.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device index. Use -1 for CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token length for truncation.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        df = pd.read_csv(input_path)
        labeled = label_dataframe(
            df,
            text_column=args.text_column,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    except Exception as exc:
        print(f"RoBERTa labeling failed: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)

    print(f"Saved RoBERTa-labeled dataset to {output_path} ({len(labeled)} rows)")
    if "roberta_label" in labeled.columns:
        print("RoBERTa label distribution:\n" + labeled["roberta_label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
