from __future__ import annotations

import pytest

from src.sentiment.local_lexicon import load_local_lexicon
from src.sentiment.nrc_analyzer import get_nrc_scores, load_local_emotion_lexicon
from src.sentiment.sarcasm import is_sarcastic
from src.sentiment.text_selection import select_sentiment_text
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


def test_negated_outage_complaint_is_negative_when_original_text_is_preserved():
    comment = (
        "In case your calendars are not working properly today is first October 2024, "
        "we don't want to hear stories about the stabilization of power Ba Zesco ltd "
        "because yesterday we didn't have power."
    )
    scores = get_vader_scores(comment)
    assert scores["label"] == "negative"
    assert scores["corrected_compound"] < 0


@pytest.mark.parametrize(
    "comment",
    [
        "You're now taking Zambians for granted.",
        "So the increment will increase the level of water in the dam?",
        "Do it quickly and increase by 50% because we are entering an election period",
        "Weren't the current tariffs increased for the same reason?",
        "The same electricity we rarely have",
        "The same power that we can't see?",
        "Even if they increase, as if there is power, ifya bupuba fye",
        "Increasing to reduce load shedding through self reliance haha",
        "Increasing to reduce load shedding through self reliance \U0001f605",
        "Which consumer? Zambians or Namibians?",
        "With what power supply? The same one for 2 hours? Jokers",
        "As if he is the one buying power for us",
        "The only thing we do is laugh because what else can possibly surprise us",
        "Which electricity ba ERB?",
        "Which electricity ba ERB.",
        "I would have commented but I don't have a loya",
    ],
)
def test_audited_tariff_discussion_complaints_are_negative(comment):
    scores = get_vader_scores(comment)
    resolved_label = "negative" if is_sarcastic(comment) else scores["label"]
    assert resolved_label == "negative"


def test_working_hours_business_impact_complaint_is_negative():
    comment = (
        "This is not making sense better to have power during working hours than sleeping "
        "hours knowing that only phones and fridges are left using power the rest are off. "
        "Bane tiyeni tusalapuke this issue of water ku Kariba is not new it started mu pf "
        "meaning that the plan should have been implemented from the word go when power was "
        "handed over to you but I guess someone overlooked this. See now the effect a lot of "
        "businesses have been affected and at the end of the day someone has to pay TAX, "
        "RENTALS, BILLS and take care of his/her family. Come on amigosi"
    )
    scores = get_vader_scores(comment)
    assert scores["raw_label"] == "positive"
    assert scores["label"] == "negative"
    assert scores["local_correction_applied"] is True
    assert "not making sense" in scores["local_correction_terms"]


def test_sentiment_text_prefers_original_over_stopword_processed_text():
    row = {
        "text_raw": "We don't have power.",
        "processed_text": "don t power",
    }
    assert select_sentiment_text(row) == "We don't have power."


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
