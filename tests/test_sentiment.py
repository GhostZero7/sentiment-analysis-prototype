from __future__ import annotations

from src.sentiment.local_lexicon import load_local_lexicon
from src.sentiment.nrc_analyzer import get_nrc_scores, load_local_emotion_lexicon
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


def test_local_emotion_lexicon_loads_variants_and_phrases():
    lexicon = load_local_emotion_lexicon()
    assert "twafwa" in lexicon
    assert "fyabupuba fye" in lexicon


def test_fyabupuba_produces_strong_anger_and_frustration():
    scores = get_nrc_scores("fyabupuba")
    assert scores["local_emotion_applied"] is True
    assert scores["nrc_anger"] == 1.0
    assert scores["nrc_frustration"] == 1.0


def test_twafwa_produces_fear_sadness_and_frustration():
    scores = get_nrc_scores("twafwa")
    assert scores["nrc_fear"] >= 0.8
    assert scores["nrc_sadness"] >= 0.9
    assert scores["nrc_frustration"] == 1.0


def test_longest_local_emotion_phrase_prevents_overlap():
    scores = get_nrc_scores("fyabupuba fye")
    assert scores["local_emotion_match_count"] == 1
    assert scores["local_emotion_terms"] == "fyabupuba fye"


def test_combined_emotion_scores_are_capped_at_one():
    scores = get_nrc_scores("angry fyabupuba")
    assert scores["nrc_anger"] == 1.0
    assert all(0.0 <= scores[key] <= 1.0 for key in (
        "nrc_anger",
        "nrc_fear",
        "nrc_trust",
        "nrc_hope",
        "nrc_sadness",
        "nrc_frustration",
    ))


def test_sarcasm_suppresses_local_trust_and_hope():
    normal = get_nrc_scores("twashuka", is_sarcastic=False)
    sarcastic = get_nrc_scores("twashuka", is_sarcastic=True)
    assert normal["local_hope_score"] > 0
    assert sarcastic["local_hope_score"] == 0.0
    assert sarcastic["local_trust_score"] == 0.0
