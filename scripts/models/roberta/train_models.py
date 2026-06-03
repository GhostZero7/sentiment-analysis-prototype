"""Train baseline sentiment models against a RoBERTa-labeled dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import train_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Train models on RoBERTa labels.")
    parser.add_argument(
        "--train-input",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "training" / "final_label_roberta_train.csv"),
        help="Training split CSV file for the RoBERTa-labeled path.",
    )
    parser.add_argument(
        "--label-column",
        default="roberta_label",
        help="Target label column to train on.",
    )
    parser.add_argument(
        "--text-column",
        default="processed_text",
        help="Text column to vectorize.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--artifact-subdir",
        default="roberta",
        help="Subdirectory for RoBERTa-trained model artifacts and metrics.",
    )
    args = parser.parse_args()

    try:
        results = train_models(
            args.train_input,
            text_column=args.text_column,
            label_column=args.label_column,
            random_state=args.random_state,
            artifact_subdir=args.artifact_subdir,
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    print(f"Training completed. Models saved to {results['model_dir']}")
    print(f"Training summary saved to {results['summary_path']}")
    for model_name, model_info in results["models"].items():
        print(f"{model_name}: saved to {model_info['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
