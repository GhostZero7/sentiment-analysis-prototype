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

## Relevance Review

We added a separate relevance review step before making relevance affect training.

The script:
- `scripts/preprocessing/filter_relevance.py`

It scores each final labeled comment using ZESCO/electricity/load-shedding/tariff/energy-service keywords and writes:
- `data/processed/relevance/relevant_comments.csv`
- `data/processed/relevance/irrelevant_comments.csv`

Each row includes:
- `is_relevant`
- `relevance_score`
- `relevance_reason`
- `relevance_terms`

These files are kept for manual monitoring so we can inspect false positives and false negatives after each refinement.

Current relevance review output:
- total scored comments: 4,135
- relevant comments: 2,813
- irrelevant comments: 1,322

Canonical relevance outputs:
- `data/processed/labeled/final_label_relevant.csv`
- `data/processed/labeled/final_label_irrelevant.csv`

The model training scripts now use `final_label_relevant.csv` by default.

## Sentiment Labeling

We added VADER-based sentiment labeling with a local correction layer for Zambian code-switching.

The local lexicon now lives in:
- `data/lexicons/local_sentiment_lexicon.csv`

The lexicon stores:
- local/code-switched term
- English meaning
- sentiment type
- correction weight
- notes for explanation

The VADER correction layer now reads this CSV, supports slash-separated variants, supports short phrase matches, and records matched terms in `local_correction_terms`.

Important current rule:
- `fyabupuba` and `ifyabupuba` are hard negative overrides.

Saved sentiment output:
- `data/processed/labeled/final_label.csv`

## NRC Emotion Analysis

We added combined NRC and local Zambian emotion scoring to match the system design and methodology chapters.

The NRC layer now extracts normalized emotion scores for:
- anger
- anticipation
- fear
- trust
- sadness

We derive project-specific NRC base values for:
- hope, mapped from anticipation
- frustration, mapped from anger and sadness

The local emotion lexicon is stored at:
- `data/lexicons/local_emotion_lexicon.csv`

It contains 151 valid entries with weights from 0.0 to 1.0 for anger, fear, trust, hope, sadness, and frustration. Slash-separated variants and phrases are supported, and longer phrases are matched first to prevent overlap.

Combined scoring:
1. Calculate normalized NRC base scores.
2. Average local weights across matched non-overlapping terms.
3. Add local scores to the corresponding NRC base scores.
4. Cap each combined score at 1.0.
5. Suppress local hope and trust when sarcasm is detected.

Explainability columns include:
- `nrc_base_*`
- `local_emotion_applied`
- `local_emotion_terms`
- `local_emotion_match_count`
- `local_*_score`

Current relevant-dataset coverage:
- 386 of 2,813 relevant comments contain at least one local emotion match.

The weights are manually assigned research heuristics. They should be reviewed against manually annotated comments and refined as the corpus grows.

## Sarcasm Detection

We added a standalone heuristic sarcasm detector tuned for ZESCO comments.

Detected sarcasm patterns include:
- gratitude plus complaint
- positive wording plus laughter emojis
- quoted false promises
- local intensifiers like `awe`, `mwe`, `fye`
- rhetorical questions plus laughter
- faint praise plus complaint context

Canonical final outputs:
- `data/processed/labeled/final_label.csv`
- `data/processed/labeled/final_label_relevant.csv`
- `data/processed/labeled/final_label_irrelevant.csv`

These final files include:
- `is_sarcastic`
- VADER scores
- NRC emotion scores
- `label`
- `corrected_label`

## Final Pipeline

Current recommended flow:

1. Preprocess cleaned comments.
2. Run VADER labeling with sarcasm-aware and local-lexicon overrides.
3. Run relevance filtering to split relevant and irrelevant comments.
4. Train models on relevant comments only.

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

Training is now intentionally train-only:

- `scripts/models/vader/train_models.py` reads the VADER training file and saves model artifacts.
- `scripts/models/roberta/train_models.py` reads the RoBERTa-labeled training file and saves model artifacts.
- evaluation/testing is handled separately by the matching `evaluate_models.py` scripts.
- accuracy, F1, classification reports, and confusion matrices are generated only during evaluation.

The VADER models were retrained after expanding the local lexicon and moving training to relevant-only comments.

Latest relevant-only VADER evaluation:
- Naive Bayes accuracy: 0.5595
- Logistic Regression accuracy: 0.6377
- SVM accuracy: 0.6448

The RoBERTa-labeled branch was refreshed from `final_label_relevant.csv` and retrained/evaluated.

Latest relevant-only RoBERTa-branch evaluation:
- Naive Bayes accuracy: 0.6661
- Logistic Regression accuracy: 0.7229
- SVM accuracy: 0.7123

## Repository State

- Git repository initialized locally and connected to GitHub.
- Corpus and processed outputs are tracked in git.
- Changes are pushed to the `master` branch.

## Notes

- The codebase still keeps preprocessing, sentiment, and sarcasm logic separated into dedicated scripts and modules.
- The final labeled dataset is now sarcasm-aware and better suited for ZESCO code-switched comments.
