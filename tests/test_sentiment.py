from __future__ import annotations

from src.sentiment.local_lexicon import load_local_lexicon
from src.sentiment.nrc_analyzer import get_nrc_scores
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
    scores = get_vader_scores("ba zesco today fyabupuba and we are happy")
    assert scores["label"] == "negative"
    assert scores["local_correction_applied"] is True
    assert scores["corrected_compound"] == -1.0


def test_local_lexicon_loads_csv_variants_and_phrases():
    lexicon = load_local_lexicon()
    assert "ifyabupuba" in lexicon
    assert "ba zee" in lexicon


def test_local_correction_matches_phrase_variant():
    scores = get_vader_scores("ba zee this schedule is happy news")
    assert scores["local_correction_applied"] is True
    assert "ba zee" in scores["local_correction_terms"]
    assert scores["corrected_compound"] < scores["raw_compound"]


def test_local_correction_keeps_hard_negative_variant():
    scores = get_vader_scores("ifyabupuba fye but we are happy")
    assert scores["label"] == "negative"
    assert scores["corrected_compound"] == -1.0


def test_nrc_scores_return_expected_shape():
    scores = get_nrc_scores("happy hopeful worried")
    assert {
        "nrc_token_count",
        "nrc_matched_token_count",
        "nrc_anger",
        "nrc_anticipation",
        "nrc_fear",
        "nrc_trust",
        "nrc_sadness",
        "nrc_hope",
        "nrc_frustration",
    }.issubset(scores.keys())
    assert scores["nrc_token_count"] == 3
    assert scores["nrc_hope"] >= 0
