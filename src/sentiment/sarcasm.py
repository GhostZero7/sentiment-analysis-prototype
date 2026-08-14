"""Heuristic sarcasm detection for ZESCO comments."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------
# Edit these patterns to add/remove sarcasm cues without changing the flow.

THANK_PATTERNS = (
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bthank god\b",
    r"\bthank goodness\b",
    r"\bappreciate\b",
)

LIKE_I_KNEW_PATTERNS = (
    r"\blike i knew\b",
    r"\bas i knew\b",
    r"\bi knew\b",
    r"\bknew it\b",
)

POSITIVE_PATTERNS = (
    r"\bhappy\b",
    r"\bgreat\b",
    r"\bgood\b",
    r"\bfinally\b",
    r"\bstable\b",
    r"\bbetter\b",
    r"\bwelcome\b",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bat least\b",
)

COMPLAINT_PATTERNS = (
    r"\bno power\b",
    r"\bdark\b",
    r"\bdarkness\b",
    r"\bcut\b",
    r"\bgone\b",
    r"\bonly\b",
    r"\bload shedding\b",
    r"\bloadshedding\b",
    r"\bslept with electricity\b",
    r"\bwoke(?: up)? with electricity\b",
    r"\bsleep(?:ed|ing)? with electricity\b",
    r"\byou can do better\b",
    r"\bplay too much\b",
    r"\bnonsense\b",
    r"\blie\b",
    r"\blies\b",
    r"\bbroken promise\b",
    r"\brestore power\b",
    r"\bstabiliz(?:e|ed|ing)\b",
    r"\bstability\b",
    r"\bsuffered\b",
    r"\bhours?\b",
    r"\bhrs?\b",
    r"\b10-14hrs\b",
    r"\bpower from\b",
)

QUOTE_PROMISE_PATTERNS = (
    r'(?:["“”\']|ati|said|promised).*(?:ended load shedding|thing of the past|stabiliz(?:e|ed|ing)|8\s?hours?|6\s?hours?|stable schedule)',
    r'(?:["“”\']|ati|said|promised).*(?:load shedding|loadshedding)',
)

LOCAL_INTENSIFIER_PATTERNS = (
    r"\bawe\b",
    r"\bche\b",
    r"\bmwe\b",
    r"\bfye\b",
    r"\bkaili\b",
    r"\bmukwai\b",
    r"\bpantu\b",
    r"\befyo\b",
    r"\bshani\b",
)

LAUGHTER_PATTERN = re.compile(
    r"(?:\U0001f602|\U0001f605|\U0001f923|\blol\b|\bkkk\b|\bhaha\b|\bhehe\b|\blmao\b)",
    re.IGNORECASE,
)
QUESTION_PATTERN = re.compile(r"\?")

THANK_RE = re.compile("|".join(THANK_PATTERNS), re.IGNORECASE)
LIKE_I_KNEW_RE = re.compile("|".join(LIKE_I_KNEW_PATTERNS), re.IGNORECASE)
POSITIVE_RE = re.compile("|".join(POSITIVE_PATTERNS), re.IGNORECASE)
COMPLAINT_RE = re.compile("|".join(COMPLAINT_PATTERNS), re.IGNORECASE)
QUOTE_PROMISE_RE = re.compile("|".join(QUOTE_PROMISE_PATTERNS), re.IGNORECASE | re.DOTALL)
LOCAL_INTENSIFIER_RE = re.compile("|".join(LOCAL_INTENSIFIER_PATTERNS), re.IGNORECASE)
AT_LEAST_RE = re.compile(r"\bat least\b|\bfinally\b", re.IGNORECASE)
DOMAIN_RE = re.compile(
    r"\b(?:zesco|power|electricity|tariffs?|load\s*shedding|consumer|water|dam)\b",
    re.IGNORECASE,
)
RHETORICAL_CUE_RE = re.compile(
    r"\b(?:which|what|same|weren't|wasn't|rarely|can't|cannot|only|increase|increment|again)\b",
    re.IGNORECASE,
)
AS_IF_DOMAIN_RE = re.compile(
    r"\bas if\b.*\b(?:zesco|power|electricity|tariffs?|buying)\b",
    re.IGNORECASE | re.DOTALL,
)
ELECTION_INCREASE_RE = re.compile(
    r"\b(?:increase|increment)\b.*\b(?:election|vote|2026)\b|"
    r"\b(?:election|vote|2026)\b.*\b(?:increase|increment)\b",
    re.IGNORECASE | re.DOTALL,
)
LAUGHED_OFF_COMPLAINT_RE = re.compile(
    r"\b(?:only thing we do is laugh|what else can possibly surprise us)\b",
    re.IGNORECASE,
)


def _coerce_text(text: object) -> str:
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    return str(text)


def is_sarcastic(text: object) -> bool:
    """Return True when a comment matches a sarcasm pattern."""
    text_str = _coerce_text(text)
    if not text_str.strip():
        return False

    text_lower = text_str.lower()

    # Rule 1: gratitude terms paired with a complaint or an ironic
    # "like I knew" construction.
    if THANK_RE.search(text_lower) and (COMPLAINT_RE.search(text_lower) or LIKE_I_KNEW_RE.search(text_lower)):
        return True

    # Rule 2: positive wording followed by laughter emoji or markers.
    if POSITIVE_RE.search(text_lower) and LAUGHTER_PATTERN.search(text_str):
        return True

    # Rule 3: quoted promise or attributed promise followed by a false
    # promise phrase such as "ended load shedding" or "stabilize".
    if QUOTE_PROMISE_RE.search(text_lower):
        return True

    # Rule 4: local intensifier plus positive wording. In this corpus, words
    # like "awe", "mwe" and "fye" often soften or reverse the apparent praise.
    if LOCAL_INTENSIFIER_RE.search(text_lower) and POSITIVE_RE.search(text_lower):
        return True

    # Rule 5: rhetorical question with laughter is a strong sarcasm marker.
    if QUESTION_PATTERN.search(text_str) and LAUGHTER_PATTERN.search(text_str):
        return True

    # Rule 6: domain-specific rhetorical questions often express disbelief
    # even when they contain no word VADER recognizes as negative.
    if (
        QUESTION_PATTERN.search(text_str)
        and DOMAIN_RE.search(text_lower)
        and RHETORICAL_CUE_RE.search(text_lower)
    ):
        return True

    # Rule 7: explicit "as if" contradictions about power or tariffs.
    if AS_IF_DOMAIN_RE.search(text_lower):
        return True

    # Rule 8: laughter paired with an outage/load-shedding complaint.
    if LAUGHTER_PATTERN.search(text_str) and COMPLAINT_RE.search(text_lower):
        return True

    # Rule 9: calls for a price increase near an election are commonly ironic
    # in this discourse and indicate dissatisfaction with the proposed policy.
    if ELECTION_INCREASE_RE.search(text_lower):
        return True

    # Rule 10: resigned laughter and surprise are complaint signals.
    if LAUGHED_OFF_COMPLAINT_RE.search(text_lower):
        return True

    # Rule 11: faint praise plus a complaint context.
    if AT_LEAST_RE.search(text_lower) and COMPLAINT_RE.search(text_lower):
        return True

    return False


def annotate_sarcasm_frame(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "corrected_label",
) -> pd.DataFrame:
    """Add an `is_sarcastic` column and optionally flip sentiment labels."""
    if text_column not in df.columns:
        raise ValueError(f"Input DataFrame must contain a '{text_column}' column.")

    out = df.copy()
    out["is_sarcastic"] = out[text_column].map(is_sarcastic)

    target_label_column = label_column if label_column in out.columns else ("label" if "label" in out.columns else "")
    if target_label_column:
        sarcastic_mask = out["is_sarcastic"] & out[target_label_column].fillna("").ne("negative")
        out.loc[sarcastic_mask, target_label_column] = "negative"

    return out


def annotate_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "corrected_label",
) -> pd.DataFrame:
    """Load, annotate, and save a CSV file with sarcasm detection."""
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    annotated = annotate_sarcasm_frame(df, text_column=text_column, label_column=label_column)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_file, index=False)
    return annotated
