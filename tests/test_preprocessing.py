from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaner import (
    CleaningConfig,
    build_preview,
    clean_frame,
    count_words,
    english_word_count,
    has_english_word,
    is_strict_english,
    lemmatize_tokens,
    normalize_text,
    process_text,
    remove_emojis,
    remove_stopwords,
    tokenize,
)
from src.preprocessing.anonymiser import anonymise_text
from src.preprocessing.relevance import assess_relevance


def test_normalize_text_removes_urls_and_noise():
    text = "Hello!!! Visit https://example.com now."
    assert normalize_text(text) == "hello visit now"


def test_normalize_text_removes_mentions():
    assert normalize_text("@john please check this") == "please check this"


def test_remove_emojis_strips_emoji_characters():
    assert remove_emojis("hello 😀 world") == "hello world"


def test_count_words_matches_cleaned_tokens():
    assert count_words("this is a simple test") == 5


def test_english_word_count_detects_presence_of_english_tokens():
    assert english_word_count("uli bwino today") >= 1
    assert has_english_word("uli bwino today") is True
    assert has_english_word("uli bwino chani") is False


def test_strict_english_detection_returns_boolean():
    assert isinstance(is_strict_english("this is an english comment"), bool)


def test_strict_english_rejects_mixed_code_switched_text():
    assert is_strict_english("please give us update uko today") is False


def test_strict_english_accepts_english_heavy_text():
    assert is_strict_english("please give us an update on the power situation today") is True


def test_stopword_removal_and_processing_pipeline():
    tokens = tokenize("This is a simple test for the pipeline")
    filtered = remove_stopwords(tokens)
    assert "is" not in filtered
    assert "the" not in filtered
    assert process_text("This is a simple test for the pipeline") == "simple test pipeline"


def test_lemmatize_tokens_keeps_output_list():
    lemmatized = lemmatize_tokens(["cars", "running"])
    assert isinstance(lemmatized, list)
    assert len(lemmatized) == 2


def test_clean_frame_filters_short_comments():
    df = pd.DataFrame(
        {
            "text": [
                "hi",
                "this is enough words",
                "another english comment here",
                None,
            ]
        }
    )
    cleaned = clean_frame(df, CleaningConfig(min_words=4, english_only=False))
    assert len(cleaned) == 2
    assert cleaned["word_count"].tolist() == [4, 4]
    assert "processed_text" in cleaned.columns


def test_build_preview_reports_pipeline_counts():
    df = pd.DataFrame({"text": ["short", "this is long enough", "another long comment here"]})
    preview = build_preview(df)
    assert preview["raw_rows"] == 3
    assert preview["word_count_gt_3"] == 2


def test_anonymise_text_removes_manual_names():
    text = "angela baker exactly we are waiting for mark mwandila"
    assert anonymise_text(text) == "exactly we are waiting for"


def test_relevance_accepts_zesco_power_comment():
    result = assess_relevance("Ba Zesco the power has gone again after two hours")
    assert result["is_relevant"] is True
    assert "zesco" in result["relevance_terms"]


def test_relevance_accepts_load_shedding_phrase():
    result = assess_relevance("Load shedding schedule is confusing today")
    assert result["is_relevant"] is True
    assert "load shedding" in result["relevance_terms"]


def test_relevance_rejects_unrelated_comment():
    result = assess_relevance("Happy birthday my friend enjoy your day")
    assert result["is_relevant"] is False
    assert result["relevance_reason"] == "matched_off_topic_terms_only"
