"""Train baseline sentiment models on the final labeled dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .utils import (
        DEFAULT_LABEL_COLUMN,
        DEFAULT_TEXT_COLUMN,
        MODELS_DIR,
        RESULTS_DIR,
        build_vectorizer,
        ensure_directories,
        load_labeled_data,
        save_joblib,
        select_target_column,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from src.models.utils import (
        DEFAULT_LABEL_COLUMN,
        DEFAULT_TEXT_COLUMN,
        MODELS_DIR,
        RESULTS_DIR,
        build_vectorizer,
        ensure_directories,
        load_labeled_data,
        save_joblib,
        select_target_column,
    )


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _safe_write_text(text: str, output_path: Path, *, rerun_suffix: str) -> Path:
    """Write text, falling back to a rerun file if the target is locked."""
    try:
        output_path.write_text(text, encoding="utf-8")
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def train_models(
    train_dataset_path: str | Path | None = None,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    random_state: int = 42,
    artifact_subdir: str = "",
) -> dict[str, object]:
    """Train Naive Bayes, Logistic Regression, and SVM models.

    Evaluation is intentionally handled by scripts/models/*/evaluate_models.py
    so training does not read or report on the held-out test split.
    """
    model_output_dir = MODELS_DIR / artifact_subdir if artifact_subdir else MODELS_DIR
    results_output_dir = RESULTS_DIR / artifact_subdir if artifact_subdir else RESULTS_DIR
    ensure_directories(model_output_dir, results_output_dir)
    print(f"Loading training data from {train_dataset_path or Path('data/processed/labeled/training/final_label_train.csv')}")
    train_df = load_labeled_data(
        train_dataset_path,
        text_column=text_column,
        label_column=label_column,
    )
    target_column = select_target_column(train_df, preferred=label_column)

    if len(train_df) < 10:
        raise ValueError("Training dataset must contain at least 10 rows.")
    if train_df[target_column].nunique() < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least 2 classes.")
    if target_column not in train_df.columns and "label" in train_df.columns:
        train_df = train_df.copy()
        train_df[target_column] = train_df["label"]

    vectorizer = build_vectorizer()
    x_train_vec = vectorizer.fit_transform(train_df[text_column].astype(str))

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "svm": LinearSVC(random_state=random_state),
    }

    results: dict[str, object] = {
        "train_dataset_path": str(
            Path(train_dataset_path)
            if train_dataset_path is not None
            else Path("data/processed/labeled/training/final_label_train.csv")
        ),
        "text_column": text_column,
        "label_column": target_column,
        "random_state": random_state,
        "artifact_subdir": artifact_subdir,
        "train_rows": int(len(train_df)),
        "model_dir": str(model_output_dir),
        "models": {},
    }

    vectorizer_path = model_output_dir / "vectorizer.joblib"
    save_joblib(vectorizer, vectorizer_path)
    print(f"Saved vectorizer to {vectorizer_path}")

    failed_models: list[dict[str, str]] = []
    model_total = len(models)
    for model_name, model in models.items():
        print(f"[{len(results['models']) + len(failed_models) + 1}/{model_total}] Training {model_name}...")
        try:
            fitted = model.fit(x_train_vec, train_df[target_column].astype(str))
            model_path = model_output_dir / f"{model_name}.joblib"
            save_joblib(fitted, model_path)
            results["models"][model_name] = {"model_path": str(model_path)}
            print(f"Saved {model_name} to {model_path}")
        except Exception as exc:
            failed_models.append({"model": model_name, "error": str(exc)})
            print(f"Failed {model_name}: {exc}")

    results["vectorizer_path"] = str(vectorizer_path)
    results["failed_models"] = failed_models

    summary_path = results_output_dir / "training_summary.json"
    summary_saved_to = _safe_write_text(json.dumps(results, indent=2, default=_json_default), summary_path, rerun_suffix="rerun")
    results["summary_path"] = str(summary_saved_to)
    if failed_models:
        print("Models that failed to train:\n" + pd.DataFrame(failed_models).to_string(index=False))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline sentiment models.")
    parser.add_argument(
        "--train-input",
        default=str(Path("data") / "processed" / "labeled" / "training" / "final_label_train.csv"),
        help="Training split CSV file.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Target label column to train on.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Text column to vectorize.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--artifact-subdir",
        default="",
        help="Optional subdirectory under data/models and data/results for saved artifacts.",
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
