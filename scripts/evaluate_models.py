"""Evaluate trained sentiment models on the persisted test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import load_artifacts
from src.models.utils import (
    DEFAULT_LABEL_COLUMN,
    DEFAULT_TEXT_COLUMN,
    RESULTS_DIR,
    evaluate_predictions,
    load_labeled_data,
)


def _safe_write_text(text: str, output_path: Path, *, rerun_suffix: str) -> Path:
    try:
        output_path.write_text(text, encoding="utf-8")
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _safe_to_csv(frame: pd.DataFrame, output_path: Path, *, rerun_suffix: str) -> Path:
    try:
        frame.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_{rerun_suffix}{output_path.suffix}")
        frame.to_csv(fallback, index=False)
        return fallback


def evaluate_models(
    test_dataset_path: str | Path,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    models_dir: str | Path = PROJECT_ROOT / "data" / "models",
    results_dir: str | Path = RESULTS_DIR,
) -> dict[str, object]:
    """Evaluate the saved models against the held-out test split."""
    df = load_labeled_data(test_dataset_path, text_column=text_column, label_column=label_column)
    target_column = label_column if label_column in df.columns else ("label" if "label" in df.columns else label_column)
    if target_column not in df.columns:
        raise ValueError(f"Test dataset must contain a '{label_column}' or 'label' column.")

    artifacts = load_artifacts(models_dir)
    vectorizer = artifacts["vectorizer"]
    models = artifacts["models"]

    x_test_vec = vectorizer.transform(df[text_column].astype(str))
    y_true = df[target_column].astype(str)

    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    cm_dir = results_root / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    reports = {}
    failed_models: list[dict[str, str]] = []
    model_total = len(models)
    for index, (model_name, model) in enumerate(models.items(), start=1):
        print(f"[{index}/{model_total}] Evaluating {model_name}...")
        try:
            y_pred = pd.Series(model.predict(x_test_vec))
            metrics = evaluate_predictions(y_true, y_pred)
            reports[model_name] = metrics["report"]
            summary_rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"],
                    "f1_weighted": metrics["f1_weighted"],
                }
            )

            labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(7, 5))
            disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
            ax.set_title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
            fig.tight_layout()
            fig_path = cm_dir / f"{model_name}.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"Completed {model_name}: accuracy={metrics['accuracy']:.4f}, f1_weighted={metrics['f1_weighted']:.4f}")
        except Exception as exc:
            failed_models.append({"model": model_name, "error": str(exc)})
            print(f"Failed {model_name}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_root / "evaluation_summary.csv"
    summary_saved_to = _safe_to_csv(summary_df, summary_path, rerun_suffix="rerun")

    reports_path = results_root / "classification_reports.json"
    reports_saved_to = _safe_write_text(json.dumps(reports, indent=2), reports_path, rerun_suffix="rerun")

    return {
        "summary_path": str(summary_saved_to),
        "reports_path": str(reports_saved_to),
        "confusion_matrix_dir": str(cm_dir),
        "summary": summary_rows,
        "failed_models": failed_models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the held-out test split.")
    parser.add_argument(
        "--test-input",
        default=str(PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label_test.csv"),
        help="Testing split CSV file.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Target label column.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Text column used for evaluation.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "data" / "models"),
        help="Directory with trained model artifacts.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Directory for evaluation outputs.",
    )
    args = parser.parse_args()

    try:
        results = evaluate_models(
            args.test_input,
            text_column=args.text_column,
            label_column=args.label_column,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
        )
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(f"Evaluation summary saved to {results['summary_path']}")
    print(f"Classification reports saved to {results['reports_path']}")
    print(f"Confusion matrices saved in {results['confusion_matrix_dir']}")
    for row in results["summary"]:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"f1_weighted={row['f1_weighted']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
