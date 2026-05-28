"""Train baseline sentiment models on the final labeled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .utils import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
    MODELS_DIR,
    RESULTS_DIR,
    build_vectorizer,
    ensure_directories,
    evaluate_predictions,
    fit_vectorizer,
    load_labeled_data,
    save_joblib,
    select_target_column,
)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    """Write a CSV, falling back to a rerun file if the target is locked."""
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def _safe_write_text(text: str, output_path: Path, *, rerun_suffix: str) -> Path:
    """Write text, falling back to a rerun file if the target is locked."""
    try:
        output_path.write_text(text, encoding="utf-8")
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _score_model(model, x_test_vec) -> pd.Series:
    predictions = model.predict(x_test_vec)
    return pd.Series(predictions)


def train_models(
    train_dataset_path: str | Path | None = None,
    test_dataset_path: str | Path | None = None,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    random_state: int = 42,
) -> dict[str, object]:
    """Train Naive Bayes, Logistic Regression, and SVM models."""
    ensure_directories()
    train_df = load_labeled_data(
        train_dataset_path,
        text_column=text_column,
        label_column=label_column,
    )
    test_df = load_labeled_data(
        test_dataset_path,
        text_column=text_column,
        label_column=label_column,
    )
    target_column = select_target_column(train_df, preferred=label_column)

    if len(train_df) < 10 or len(test_df) < 10:
        raise ValueError("Train and test datasets must each contain at least 10 rows.")
    if train_df[target_column].nunique() < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least 2 classes.")
    if target_column not in test_df.columns and "label" not in test_df.columns:
        raise ValueError(f"Test dataset must contain a '{target_column}' or 'label' column.")

    if target_column not in test_df.columns and "label" in test_df.columns:
        test_df = test_df.copy()
        test_df[target_column] = test_df["label"]
    elif target_column not in train_df.columns and "label" in train_df.columns:
        train_df = train_df.copy()
        train_df[target_column] = train_df["label"]

    vectorizer = build_vectorizer()
    x_train_vec, x_test_vec = fit_vectorizer(
        vectorizer,
        train_df[text_column].astype(str),
        test_df[text_column].astype(str),
    )

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "svm": LinearSVC(random_state=random_state),
    }

    results: dict[str, object] = {
        "train_dataset_path": str(
            Path(train_dataset_path)
            if train_dataset_path is not None
            else Path("data/processed/labeled/final_label_train.csv")
        ),
        "test_dataset_path": str(
            Path(test_dataset_path)
            if test_dataset_path is not None
            else Path("data/processed/labeled/final_label_test.csv")
        ),
        "text_column": text_column,
        "label_column": target_column,
        "random_state": random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": {},
    }

    save_joblib(vectorizer, MODELS_DIR / "vectorizer.joblib")

    metrics_rows = []
    for model_name, model in models.items():
        fitted = model.fit(x_train_vec, train_df[target_column].astype(str))
        y_pred = _score_model(fitted, x_test_vec)
        metrics = evaluate_predictions(test_df[target_column].astype(str), y_pred)
        results["models"][model_name] = metrics
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )
        save_joblib(fitted, MODELS_DIR / f"{model_name}.joblib")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    metrics_saved_to = _safe_to_csv(metrics_df, metrics_path, rerun_suffix="rerun")
    results["metrics_path"] = str(metrics_saved_to)
    results["vectorizer_path"] = str(MODELS_DIR / "vectorizer.joblib")

    summary_path = RESULTS_DIR / "training_summary.json"
    summary_saved_to = _safe_write_text(json.dumps(results, indent=2, default=_json_default), summary_path, rerun_suffix="rerun")
    results["summary_path"] = str(summary_saved_to)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline sentiment models.")
    parser.add_argument(
        "--train-input",
        default=str(Path("data") / "processed" / "labeled" / "final_label_train.csv"),
        help="Training split CSV file.",
    )
    parser.add_argument(
        "--test-input",
        default=str(Path("data") / "processed" / "labeled" / "final_label_test.csv"),
        help="Testing split CSV file.",
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
    args = parser.parse_args()

    try:
        results = train_models(
            args.train_input,
            args.test_input,
            text_column=args.text_column,
            label_column=args.label_column,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    print(f"Training completed. Metrics saved to {results['metrics_path']}")
    print(f"Training summary saved to {results['summary_path']}")
    for model_name, metrics in results["models"].items():
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
