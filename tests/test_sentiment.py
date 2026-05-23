from __future__ import annotations

from src.sentiment.vader_analyzer import get_vader_scores


def test_vader_scores_return_expected_shape():
    scores = get_vader_scores("I love this project")
    assert {
        "raw_compound",
        "raw_label",
        "compound",
        "pos",
        "neu",
        "neg",
        "label",
        "corrected_compound",
        "corrected_label",
        "local_correction_applied",
        "local_correction_score",
        "local_correction_terms",
    }.issubset(scores.keys())
    assert scores["label"] in {"positive", "neutral", "negative"}


def test_local_correction_overrides_known_negative_term():
    scores = get_vader_scores("ba zesco today fyabupuba")
    assert scores["label"] == "negative"
    assert scores["local_correction_applied"] is True
