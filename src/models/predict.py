"""Load trained sentiment models and run inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import MODELS_DIR, load_joblib


def load_artifacts(models_dir: str | Path = MODELS_DIR) -> dict[str, Any]:
    """Load the fitted vectorizer and the trained model artifacts."""
    models_path = Path(models_dir)
    vectorizer = load_joblib(models_path / "vectorizer.joblib")
    models = {
        "naive_bayes": load_joblib(models_path / "naive_bayes.joblib"),
        "logistic_regression": load_joblib(models_path / "logistic_regression.joblib"),
        "svm": load_joblib(models_path / "svm.joblib"),
    }
    return {
        "vectorizer": vectorizer,
        "models": models,
    }


def predict_texts(
    texts: list[str],
    *,
    models_dir: str | Path = MODELS_DIR,
) -> pd.DataFrame:
    """Run all saved models on a list of texts."""
    artifacts = load_artifacts(models_dir)
    vectorizer = artifacts["vectorizer"]
    models = artifacts["models"]

    text_series = pd.Series([str(text or "") for text in texts], dtype=str)
    x_vec = vectorizer.transform(text_series)

    out = pd.DataFrame({"text": text_series})
    for model_name, model in models.items():
        out[f"{model_name}_prediction"] = model.predict(x_vec)
    return out


def predict_csv(
    input_path: str | Path,
    *,
    text_column: str = "processed_text",
    models_dir: str | Path = MODELS_DIR,
) -> pd.DataFrame:
    """Load a CSV and append predictions from all stored models."""
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    if text_column not in df.columns:
        raise ValueError(f"Input file must contain a '{text_column}' column.")

    predictions = predict_texts(df[text_column].fillna("").astype(str).tolist(), models_dir=models_dir)
    return pd.concat([df.reset_index(drop=True), predictions.drop(columns=["text"])], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run saved sentiment models on text.")
    parser.add_argument(
        "--input",
        help="Optional CSV input file. If omitted, use --text instead.",
    )
    parser.add_argument(
        "--text",
        help="Single text string to classify.",
    )
    parser.add_argument(
        "--text-column",
        default="processed_text",
        help="Text column when using --input.",
    )
    args = parser.parse_args()

    try:
        if args.input:
            predicted = predict_csv(args.input, text_column=args.text_column)
            print(predicted.head().to_string())
            return 0
        if args.text:
            predicted = predict_texts([args.text])
            print(predicted.to_string(index=False))
            return 0
        print("Provide either --input or --text.")
        return 1
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

