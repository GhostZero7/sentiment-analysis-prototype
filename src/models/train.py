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
    split_stratified,
)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _score_model(model, x_test_vec) -> pd.Series:
    predictions = model.predict(x_test_vec)
    return pd.Series(predictions)


def train_models(
    dataset_path: str | Path | None = None,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    """Train Naive Bayes, Logistic Regression, and SVM models."""
    ensure_directories()
    df = load_labeled_data(dataset_path, text_column=text_column, label_column=label_column)
    target_column = select_target_column(df, preferred=label_column)

    if len(df) < 10:
        raise ValueError("Dataset is too small to train meaningful models.")
    if df[target_column].nunique() < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least 2 classes.")

    bundle = split_stratified(
        df,
        text_column=text_column,
        label_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    vectorizer = build_vectorizer()
    x_train_vec, x_test_vec = fit_vectorizer(vectorizer, bundle.x_train, bundle.x_test)

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "svm": LinearSVC(random_state=random_state),
    }

    results: dict[str, object] = {
        "dataset_path": str(Path(dataset_path) if dataset_path is not None else Path("data/processed/labeled/final_label.csv")),
        "text_column": text_column,
        "label_column": target_column,
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(bundle.x_train)),
        "test_rows": int(len(bundle.x_test)),
        "models": {},
    }

    save_joblib(vectorizer, MODELS_DIR / "vectorizer.joblib")

    metrics_rows = []
    for model_name, model in models.items():
        fitted = model.fit(x_train_vec, bundle.y_train)
        y_pred = _score_model(fitted, x_test_vec)
        metrics = evaluate_predictions(bundle.y_test, y_pred)
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
    metrics_df.to_csv(metrics_path, index=False)
    results["metrics_path"] = str(metrics_path)
    results["vectorizer_path"] = str(MODELS_DIR / "vectorizer.joblib")

    summary_path = RESULTS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    results["summary_path"] = str(summary_path)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train baseline sentiment models.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "processed" / "labeled" / "final_label.csv"),
        help="Input labeled dataset.",
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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    try:
        results = train_models(
            args.input,
            text_column=args.text_column,
            label_column=args.label_column,
            test_size=args.test_size,
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
