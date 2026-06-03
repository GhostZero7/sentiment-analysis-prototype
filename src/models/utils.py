"""Shared utilities for sentiment model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"

DEFAULT_LABEL_COLUMN = "corrected_label"
DEFAULT_TEXT_COLUMN = "processed_text"


@dataclass(slots=True)
class DatasetBundle:
    """Container for train/test text and labels."""

    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def ensure_directories(
    models_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Create data output directories if they do not exist."""
    model_root = Path(models_dir) if models_dir is not None else MODELS_DIR
    results_root = Path(results_dir) if results_dir is not None else RESULTS_DIR
    model_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)


def load_labeled_data(
    path: str | Path | None = None,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Load the final labeled dataset and validate required columns."""
    csv_path = Path(path) if path is not None else DATA_DIR / "processed" / "labeled" / "final_label.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Dataset must contain a '{text_column}' column.")
    if label_column not in df.columns:
        if "label" in df.columns:
            label_column = "label"
        else:
            raise ValueError(f"Dataset must contain a '{label_column}' or 'label' column.")

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[label_column] = df[label_column].fillna("").astype(str)
    if "analysis_failed" in df.columns:
        df = df[~df["analysis_failed"].fillna(False).astype(bool)].copy()
    df = df[df[text_column].str.strip().ne("") & df[label_column].str.strip().ne("")].copy()
    return df


def select_target_column(df: pd.DataFrame, preferred: str = DEFAULT_LABEL_COLUMN) -> str:
    """Pick the most appropriate label column for training."""
    if preferred in df.columns:
        return preferred
    if "corrected_label" in df.columns:
        return "corrected_label"
    if "label" in df.columns:
        return "label"
    raise ValueError("No usable target label column found.")


def split_stratified(
    df: pd.DataFrame,
    *,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    """Create an 80/20 stratified split for training and testing."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    X = df[text_column].astype(str)
    y = df[label_column].astype(str)
    train_idx, test_idx = next(splitter.split(X, y))
    return DatasetBundle(
        x_train=X.iloc[train_idx].reset_index(drop=True),
        x_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
    )


def build_vectorizer(
    *,
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used for all classical models."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
    )


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    x_train: pd.Series,
    x_test: pd.Series,
) -> tuple[Any, Any]:
    """Fit vectorizer on train text and transform train/test text."""
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Compute standard classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist a Python object to disk with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output_path)
    return output_path


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact from disk."""
    return joblib.load(Path(path))
