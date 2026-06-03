"""Split a RoBERTa-labeled dataset into persistent train and test CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.utils import DEFAULT_TEXT_COLUMN, load_labeled_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Split the RoBERTa-labeled dataset into train and test files.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label_roberta.csv"),
        help="Input RoBERTa-labeled dataset.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Text column used for splitting.",
    )
    parser.add_argument(
        "--label-column",
        default="roberta_label",
        help="Target label column used for stratification.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--train-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "training" / "final_label_roberta_train.csv"),
        help="Output CSV path for the training split.",
    )
    parser.add_argument(
        "--test-output",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "testing" / "final_label_roberta_test.csv"),
        help="Output CSV path for the test split.",
    )
    args = parser.parse_args()

    try:
        df = load_labeled_data(args.input, text_column=args.text_column, label_column=args.label_column)
        if args.label_column not in df.columns:
            raise ValueError(f"Expected '{args.label_column}' column in RoBERTa-labeled dataset.")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(df[args.text_column].astype(str), df[args.label_column].astype(str)))
    except Exception as exc:
        print(f"Split failed: {exc}")
        return 1

    train_rows = df.iloc[train_idx].copy()
    test_rows = df.iloc[test_idx].copy()

    train_output = Path(args.train_output)
    test_output = Path(args.test_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(train_output, index=False)
    test_rows.to_csv(test_output, index=False)

    print(f"Saved training split to {train_output} ({len(train_rows)} rows)")
    print(f"Saved testing split to {test_output} ({len(test_rows)} rows)")
    print("Train label distribution:\n" + train_rows[args.label_column].value_counts().to_string())
    print("Test label distribution:\n" + test_rows[args.label_column].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
