"""Text cleaning helpers for Facebook comments."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 42
except Exception:  # pragma: no cover - optional dependency fallback
    detect = None
    LangDetectException = Exception

try:
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - optional dependency fallback
    WordNetLemmatizer = None

try:
    from nltk.corpus import words as nltk_words
except Exception:  # pragma: no cover - optional dependency fallback
    nltk_words = None


WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

_FALLBACK_ENGLISH_HINTS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "back",
    "be",
    "been",
    "before",
    "best",
    "but",
    "by",
    "can",
    "come",
    "do",
    "does",
    "done",
    "for",
    "from",
    "get",
    "give",
    "go",
    "good",
    "got",
    "great",
    "had",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "just",
    "like",
    "love",
    "me",
    "more",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "people",
    "please",
    "power",
    "really",
    "said",
    "see",
    "she",
    "so",
    "some",
    "situation",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "time",
    "today",
    "to",
    "up",
    "update",
    "was",
    "we",
    "well",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "work",
    "would",
    "you",
    "your",
}

_FALLBACK_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "more",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "she",
    "so",
    "some",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}

_LOCAL_LANGUAGE_MARKERS = {
    "awe",
    "ati",
    "bemba",
    "ba",
    "bashi",
    "bwino",
    "chimo",
    "fye",
    "fyabupuba",
    "ifyo",
    "ifi",
    "imwe",
    "imwebo",
    "ine",
    "kanshi",
    "kuli",
    "lelo",
    "mfw",
    "imo",
    "mwe",
    "muli",
    "nangu",
    "nshi",
    "nshani",
    "ok",
    "pa",
    "shani",
    "tuli",
    "uko",
    "umwe",
    "umwebo",
    "umwenso",
    "wapya",
    "wene",
    "ya",
    "yama",
    "zambian",
    "zulu",
}


@dataclass(slots=True)
class CleaningConfig:
    min_words: int = 4
    english_only: bool = False
    strict_english_only: bool = False
    strict_english_min_ratio: float = 0.36
    strict_english_min_words: int = 4
    remove_emojis: bool = False
    remove_stopwords: bool = True
    lemmatize: bool = True


def normalize_text(text: str) -> str:
    """Lowercase and remove URLs, punctuation-heavy noise, and extra spaces."""
    if text is None:
        return ""

    cleaned = str(text).strip().lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    cleaned = PUNCT_PATTERN.sub(" ", cleaned)
    cleaned = SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def remove_emojis(text: str) -> str:
    """Remove emoji and symbol ranges while preserving normal text."""
    if not text:
        return ""
    cleaned = EMOJI_PATTERN.sub(" ", str(text))
    cleaned = SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def count_words(text: str) -> int:
    """Count token-like words in a normalized string."""
    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def tokenize(text: str) -> list[str]:
    """Split text into lowercased word tokens."""
    if not text:
        return []
    return WORD_PATTERN.findall(text.lower())


def _load_english_vocabulary() -> set[str]:
    """Load a token-level English vocabulary with a safe fallback."""
    if nltk_words is None:
        return set(_FALLBACK_ENGLISH_HINTS)

    try:
        return {word.lower() for word in nltk_words.words()}
    except Exception:  # pragma: no cover - corpus availability varies
        return set(_FALLBACK_ENGLISH_HINTS)


ENGLISH_VOCABULARY = _load_english_vocabulary()


def _load_stopwords() -> set[str]:
    """Load stopwords with a fallback list."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:  # pragma: no cover - corpus availability varies
        return set(_FALLBACK_STOPWORDS)


STOPWORDS = _load_stopwords()
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def english_word_count(text: str) -> int:
    """Count English-looking tokens in a normalized string."""
    if not text:
        return 0

    tokens = WORD_PATTERN.findall(text.lower())
    if not tokens:
        return 0

    return sum(1 for token in tokens if token in ENGLISH_VOCABULARY)


def has_english_word(text: str) -> bool:
    """Return True when at least one English word is present."""
    return english_word_count(text) > 0


def is_strict_english(text: str, *, min_ratio: float = 0.9, min_english_words: int = 4) -> bool:
    """Return True when a comment is overwhelmingly English with no obvious local markers."""
    if not text:
        return False
    tokens = tokenize(text)
    if not tokens:
        return False
    if any(token in _LOCAL_LANGUAGE_MARKERS for token in tokens):
        return False

    english_count = sum(1 for token in tokens if token in ENGLISH_VOCABULARY)
    ratio = english_count / len(tokens)
    if english_count < min_english_words or ratio < min_ratio:
        return False

    if detect is None:
        return True

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords from a token list."""
    return [token for token in tokens if token not in STOPWORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens with a safe fallback when WordNet is unavailable."""
    if not tokens:
        return []
    if LEMMATIZER is None:
        return tokens
    try:
        return [LEMMATIZER.lemmatize(token) for token in tokens]
    except Exception:  # pragma: no cover - corpus availability varies
        return tokens


def process_text(text: str, *, remove_stopwords_enabled: bool = True, lemmatize_enabled: bool = True) -> str:
    """Run the full text-processing chain and return a single normalized string."""
    tokens = tokenize(normalize_text(text))
    if remove_stopwords_enabled:
        tokens = remove_stopwords(tokens)
    if lemmatize_enabled:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens).strip()


def clean_frame(df: pd.DataFrame, config: CleaningConfig | None = None) -> pd.DataFrame:
    """
    Clean a comment DataFrame in stages:
    1. normalize text
    2. compute word count
    3. keep rows with more than min_words - 1 words
    4. optionally keep English-only rows
    """
    if config is None:
        config = CleaningConfig()

    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    out = df.copy()
    out["text_raw"] = out["text"].fillna("").astype(str)
    out["text_clean"] = out["text_raw"].map(normalize_text)
    out["word_count"] = out["text_clean"].map(count_words)
    out = out[out["word_count"] >= config.min_words].copy()

    if config.english_only:
        out["english_word_count"] = out["text_clean"].map(english_word_count)
        out["has_english_word"] = out["english_word_count"] > 0
        out = out[out["has_english_word"]].copy()
    else:
        out["english_word_count"] = pd.NA
        out["has_english_word"] = pd.NA

    if config.strict_english_only:
        out["is_strict_english"] = out["text_clean"].map(
            lambda text: is_strict_english(
                text,
                min_ratio=config.strict_english_min_ratio,
                min_english_words=config.strict_english_min_words,
            )
        )
        out = out[out["is_strict_english"]].copy()
    else:
        out["is_strict_english"] = pd.NA

    out["emoji_removed_text"] = out["text_clean"].map(remove_emojis) if config.remove_emojis else out["text_clean"]
    out["tokens"] = out["emoji_removed_text"].map(tokenize)
    if config.remove_stopwords:
        out["tokens_no_stopwords"] = out["tokens"].map(remove_stopwords)
    else:
        out["tokens_no_stopwords"] = out["tokens"]

    if config.lemmatize:
        out["tokens_lemmatized"] = out["tokens_no_stopwords"].map(lemmatize_tokens)
    else:
        out["tokens_lemmatized"] = out["tokens_no_stopwords"]

    out["processed_text"] = out["tokens_lemmatized"].map(lambda tokens: " ".join(tokens).strip())

    out.reset_index(drop=True, inplace=True)
    return out


def load_comment_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple CSV files that contain a text column."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        csv_path = Path(path)
        if csv_path.is_file():
            frames.append(pd.read_csv(csv_path))

    if not frames:
        return pd.DataFrame(columns=["text"])

    return pd.concat(frames, ignore_index=True)


def build_preview(df: pd.DataFrame) -> dict[str, int]:
    """Return stage counts so we can inspect the cleaning pipeline."""
    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    base = df.copy()
    base["text_raw"] = base["text"].fillna("").astype(str)
    base["text_clean"] = base["text_raw"].map(normalize_text)
    base["word_count"] = base["text_clean"].map(count_words)
    base["english_word_count"] = base["text_clean"].map(english_word_count)
    base["emoji_removed_text"] = base["text_clean"].map(remove_emojis)

    word_filtered = base[base["word_count"] >= 4]
    english_filtered = word_filtered[word_filtered["english_word_count"] > 0]
    processed = clean_frame(
        df,
        CleaningConfig(min_words=4, english_only=True, strict_english_only=False, remove_emojis=True, remove_stopwords=True, lemmatize=True),
    )

    return {
        "raw_rows": int(len(base)),
        "word_count_gt_3": int(len(word_filtered)),
        "english_rows": int(len(english_filtered)),
        "processed_rows": int(len(processed)),
    }
