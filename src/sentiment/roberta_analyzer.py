"""RoBERTa sentiment labeling helpers for the alternate training path."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16

_SENTIMENT_ORDER = ("negative", "neutral", "positive")


def _normalize_label(raw_label: str, fallback_index: int | None = None) -> str:
    """Normalize model labels into the project sentiment classes."""
    label = str(raw_label).strip().lower()
    if "neg" in label:
        return "negative"
    if "neu" in label:
        return "neutral"
    if "pos" in label:
        return "positive"
    if label.startswith("label_") and fallback_index is not None:
        if 0 <= fallback_index < len(_SENTIMENT_ORDER):
            return _SENTIMENT_ORDER[fallback_index]
    if fallback_index is not None and 0 <= fallback_index < len(_SENTIMENT_ORDER):
        return _SENTIMENT_ORDER[fallback_index]
    return label or "neutral"


@lru_cache(maxsize=4)
def load_roberta_components(
    model_name: str = DEFAULT_MODEL_NAME,
    device: int = -1,
) -> tuple[Any, Any, torch.device]:
    """Load tokenizer and model once and reuse them across batches."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    torch_device = torch.device("cpu") if device < 0 or not torch.cuda.is_available() else torch.device(f"cuda:{device}")
    model.to(torch_device)
    model.eval()
    return tokenizer, model, torch_device


def label_text_batch(
    texts: list[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: int = -1,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Label a batch of texts with RoBERTa sentiment predictions."""
    tokenizer, model, torch_device = load_roberta_components(model_name=model_name, device=device)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(torch_device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu()

    results: list[dict[str, Any]] = []
    id2label = getattr(model.config, "id2label", {}) or {}

    for row_index, probs in enumerate(probabilities):
        score_values = probs.tolist()
        best_index = int(torch.argmax(probs).item())
        raw_label = id2label.get(best_index, f"LABEL_{best_index}")
        normalized_label = _normalize_label(raw_label, fallback_index=best_index)
        score_map: dict[str, float] = {}
        for index, score in enumerate(score_values):
            mapped_raw = id2label.get(index, f"LABEL_{index}")
            mapped_label = _normalize_label(mapped_raw, fallback_index=index)
            score_map[mapped_label] = float(score)
        best_score = float(score_map.get(normalized_label, score_values[best_index]))
        results.append(
            {
                "roberta_label": normalized_label,
                "roberta_raw_label": str(raw_label),
                "roberta_confidence": best_score,
                "roberta_negative_score": score_map.get("negative"),
                "roberta_neutral_score": score_map.get("neutral"),
                "roberta_positive_score": score_map.get("positive"),
            }
        )
    return results


def label_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str = "processed_text",
    model_name: str = DEFAULT_MODEL_NAME,
    device: int = -1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> pd.DataFrame:
    """Append RoBERTa sentiment labels to a dataframe."""
    if text_column not in df.columns:
        raise ValueError(f"Dataframe must contain a '{text_column}' column.")

    texts = df[text_column].fillna("").astype(str).tolist()
    if not texts:
        return df.copy()

    batches: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batches.extend(
            label_text_batch(
                batch_texts,
                model_name=model_name,
                device=device,
                max_length=max_length,
            )
        )

    labeled = df.copy().reset_index(drop=True)
    labels_df = pd.DataFrame(batches)
    for column in labels_df.columns:
        labeled[column] = labels_df[column]
    return labeled
