from __future__ import annotations

import pandas as pd

from src.sentiment.sarcasm import annotate_sarcasm_frame, is_sarcastic


def test_known_sarcasm_examples_return_true():
    test_cases = [
        "Thank God I bought Kuma 01, like I knew 🤷🏽‍♀️",
        "Finally the stable schedule is out... 3hrs nonsense. Awe",
        "Welcome back to civilization admin... you can do better thanks",
        "Happy 1st October Ba Zesco 😂",
        "So uncle Zesco this is the good news you were counting down for? You play too much 😂",
        "At least we had power from 10-14hrs... lol",
        "Thank you ba zesco, slept with electricity and woke up with electricity. I even ironed my boxer",
    ]
    for case in test_cases:
        assert is_sarcastic(case) is True


def test_non_sarcastic_thanks_remains_false():
    assert is_sarcastic("This is a genuine thanks for fixing the power") is False


def test_label_flip_on_sarcasm():
    df = pd.DataFrame(
        {
            "text": [
                "Happy 1st October Ba Zesco 😂",
                "This is a genuine thanks for fixing the power",
            ],
            "corrected_label": ["positive", "positive"],
        }
    )
    annotated = annotate_sarcasm_frame(df)
    assert bool(annotated.loc[0, "is_sarcastic"]) is True
    assert annotated.loc[0, "corrected_label"] == "negative"
    assert bool(annotated.loc[1, "is_sarcastic"]) is False
    assert annotated.loc[1, "corrected_label"] == "positive"

