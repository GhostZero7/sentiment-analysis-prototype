"""Simple anonymisation helpers for comment text."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|facebook\.com/\S+|profile\.php\S*", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Small manual list from names that appeared in collected/raw examples.
MANUAL_NAME_TOKENS = {
    "amos",
    "angela",
    "baker",
    "chanda",
    "chewe",
    "chillz",
    "francis",
    "kayenda",
    "mark",
    "muloto",
    "mwandila",
    "mwaba",
    "mwewa",
    "remnant",
    "richard",
    "simz",
    "tembo",
}


def anonymise_text(text: str) -> str:
    """Remove obvious links, mentions, and manually identified names from text."""
    cleaned = URL_PATTERN.sub(" ", str(text).lower())
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    tokens = [token for token in re.findall(r"\b[\w']+\b", cleaned) if token not in MANUAL_NAME_TOKENS]
    return WHITESPACE_PATTERN.sub(" ", " ".join(tokens)).strip()
