# Project Work Log

## Scope Completed

We built the initial ZESCO sentiment analysis prototype and then iterated it into a sarcasm-aware labeling pipeline for Zambian Facebook comments.

## Data Collection

- Set up Apify-based Facebook scraping.
- Collected raw comment files per post in `data/raw/`.
- Combined all raw comments into `data/raw/all_comments.csv`.
- Removed `author_name` from the collector output so raw records do not keep personal names.

Current raw corpus:
- 8 posts with successful comment files
- 4,446 collected comments in the post-level raw files
- 4,793 rows in the combined raw file

## Preprocessing

The preprocessing flow currently does:

1. Remove URLs and mentions.
2. Keep comments with more than 3 words.
3. Optionally keep English-heavy comments.
4. Remove emojis.
5. Remove stop words.
6. Lemmatize tokens.

Saved cleaned outputs:
- `data/processed/cleaned/comments_stage1_3plus_english.csv`
- `data/processed/cleaned/comments_stage2_emoji_stopword_lemma.csv`
- `data/processed/cleaned/comments_stage2_strict_english.csv`

## Sentiment Labeling

We added VADER-based sentiment labeling with a local correction layer for Zambian code-switching.

Local sentiment rules:
- `fyabupuba` is a hard negative override.
- `bwino` is a positive hint.
- `awe` is a negative hint.

Saved sentiment outputs:
- `data/processed/labeled/comments_labeled_vader.csv`
- `data/processed/labeled/comments_labeled_vader_english_only.csv`

## NRC Emotion Analysis

We added NRC emotion scoring to match the system design and methodology chapters.

The NRC layer now extracts normalized emotion scores for:
- anger
- anticipation
- fear
- trust
- sadness

We also derive project-specific values for:
- hope, mapped from anticipation
- frustration, mapped from anger and sadness

These emotion scores are appended to the labeled datasets alongside VADER sentiment so the final data includes both polarity and emotion features.

## Sarcasm Detection

We added a standalone heuristic sarcasm detector tuned for ZESCO comments.

Detected sarcasm patterns include:
- gratitude plus complaint
- positive wording plus laughter emojis
- quoted false promises
- local intensifiers like `awe`, `mwe`, `fye`
- rhetorical questions plus laughter
- faint praise plus complaint context

Sarcasm-aware outputs:
- `data/processed/labeled/comments_labeled_vader_sarcasm.csv`
- `data/processed/labeled/comments_labeled_vader_sarcasm_english_only.csv`

Canonical final outputs:
- `data/processed/labeled/final_label.csv`
- `data/processed/labeled/final_label_english_only.csv`

These final files include:
- `is_sarcastic`
- VADER scores
- NRC emotion scores
- `label`
- `corrected_label`

## Final Pipeline

Current recommended flow:

1. Preprocess cleaned comments.
2. Run sarcasm detection.
3. Run VADER labeling with sarcasm-aware overrides.

The scripts are now organized by purpose:

- `scripts/collection/`
- `scripts/preprocessing/`
- `scripts/sentiment/`
- `scripts/models/`
- `scripts/models/vader/`
- `scripts/models/roberta/`
- `scripts/models/shared/` for shared model evaluation helpers

The VADER labeling and model training steps now run row-by-row or model-by-model with progress messages and failure logs, so any problematic comments or model steps are visible during execution instead of failing silently.

Train/test split outputs now live in dedicated folders:

- `data/processed/labeled/training/` for the 80% training comments
- `data/processed/labeled/testing/` for the 20% testing comments

The training and evaluation scripts read from those folders by default.

## Repository State

- Git repository initialized locally and connected to GitHub.
- Corpus and processed outputs are tracked in git.
- Changes are pushed to the `master` branch.

## Notes

- The codebase still keeps preprocessing, sentiment, and sarcasm logic separated into dedicated scripts and modules.
- The final labeled dataset is now sarcasm-aware and better suited for ZESCO code-switched comments.
