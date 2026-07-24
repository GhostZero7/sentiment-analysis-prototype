# Supervisor Briefing: Sentiment Analysis Prototype

Updated on 2026-07-24.

## Project Summary

This project is a prototype for analysing public Facebook comments about ZESCO and electricity-service experiences in Zambia. The system collects comments, cleans them, checks whether they are relevant to the energy-service topic, labels sentiment and emotions, detects sarcasm, trains baseline machine-learning models, and presents results through a Streamlit dashboard.

The main aim is not only to classify comments as positive, neutral, or negative, but also to make the classification more suitable for Zambian social-media language. This is why the project includes sarcasm rules, local/code-switched sentiment terms, and a local emotion lexicon.

## Current System Flow

```text
Facebook post URL
        |
        v
Apify comment collection
        |
        v
Raw CSV files
        |
        v
Preprocessing
        |
        v
Sentiment, sarcasm, emotion, and relevance analysis
        |
        v
Relevant/irrelevant labeled datasets
        |
        v
Train/test split
        |
        v
Baseline model training and evaluation
        |
        v
Streamlit dashboard and saved analysis outputs
```

## What Happens at Each Stage

### 1. Data Collection

Facebook comments are collected through Apify. The system can collect comments from saved target URLs or from a URL entered directly in the dashboard.

Current corpus status:

- 8 Facebook posts with successful collected comment files.
- 4,446 comments in the post-level raw files.
- 4,793 rows in the combined raw file.
- Personal author names are removed from collector output to reduce privacy risk.

Key files:

- `scripts/collection/download_data.py`
- `src/data_collection/apify_client.py`
- `data/raw/`

### 2. Preprocessing

The preprocessing stage prepares noisy Facebook comments for analysis. It removes URLs and mentions, filters very short comments, keeps English-heavy comments where needed, removes emojis, removes stop words, and lemmatizes tokens.

Saved cleaned outputs:

- `data/processed/cleaned/comments_stage1_3plus_english.csv`
- `data/processed/cleaned/comments_stage2_emoji_stopword_lemma.csv`
- `data/processed/cleaned/comments_stage2_strict_english.csv`

Key files:

- `scripts/preprocessing/preprocess_data.py`
- `src/preprocessing/cleaner.py`
- `src/preprocessing/anonymiser.py`

### 3. Relevance Filtering

Not every Facebook comment is about ZESCO, electricity, load shedding, tariffs, or power-service experience. A relevance filter scores comments using energy-service keywords and separates useful comments from unrelated ones.

Current relevance output:

- 4,135 total scored comments.
- 2,813 relevant comments.
- 1,322 irrelevant comments.

The training scripts now use only the relevant comments by default. This helps prevent the model from learning from unrelated discussion.

Key files:

- `scripts/preprocessing/filter_relevance.py`
- `src/preprocessing/relevance.py`
- `data/processed/labeled/final_label_relevant.csv`
- `data/processed/labeled/final_label_irrelevant.csv`

### 4. Sentiment Labeling

The first sentiment layer uses VADER, which is suitable for social-media text. Because standard VADER does not understand all local/code-switched expressions, the project adds a local correction layer.

The local sentiment lexicon stores Zambian or code-switched terms, their meanings, sentiment type, correction weight, and notes. For example, strongly negative local terms can override or adjust the original VADER score.

Important current rule:

- `fyabupuba` and `ifyabupuba` are treated as hard negative overrides.

Key files:

- `src/sentiment/vader_analyzer.py`
- `src/sentiment/local_lexicon.py`
- `data/lexicons/local_sentiment_lexicon.csv`
- `scripts/sentiment/label_vader_sarcasm.py`

### 5. Sarcasm Detection

Sarcasm is common in social-media complaints, especially when people use positive wording to express frustration. The project includes a heuristic sarcasm detector tuned to ZESCO-related comments.

It detects patterns such as:

- Gratitude followed by a complaint.
- Positive wording mixed with laughter.
- Quoted or mocked promises.
- Local intensifiers such as `awe`, `mwe`, and `fye`.
- Rhetorical questions with laughter.
- Faint praise followed by complaint context.

When sarcasm is detected, the corrected sentiment can be adjusted so the system does not mistake sarcastic praise for genuine positive sentiment.

Key files:

- `src/sentiment/sarcasm.py`
- `tests/test_sarcasm.py`

### 6. Emotion Analysis

The system includes NRC-style emotion analysis and a local Zambian emotion lexicon. It focuses on emotions relevant to electricity-service comments:

- Anger
- Hope
- Fear
- Trust
- Sadness
- Frustration

The system first calculates base NRC scores, then adds matching local emotion scores from the local lexicon. Local hope and trust are suppressed when sarcasm is detected, because sarcastic positive wording should not be treated as genuine trust or hope.

Current local emotion coverage:

- 386 of 2,813 relevant comments contain at least one local emotion match.

Key files:

- `src/sentiment/nrc_analyzer.py`
- `data/lexicons/local_emotion_lexicon.csv`

### 7. Model Training and Evaluation

The project trains baseline machine-learning models using relevant comments only. The training and testing data are stored separately to avoid testing on training data.

Models used:

- Naive Bayes
- Logistic Regression
- Support Vector Machine

Latest relevant-only VADER evaluation:

- Naive Bayes accuracy: 0.5595
- Logistic Regression accuracy: 0.6377
- SVM accuracy: 0.6448

Latest relevant-only RoBERTa-branch evaluation:

- Naive Bayes accuracy: 0.6661
- Logistic Regression accuracy: 0.7229
- SVM accuracy: 0.7123

The RoBERTa-labeled branch performs better in the current evaluation, but it is kept as a separate branch so the original VADER-based flow remains explainable and easy to inspect.

Key files:

- `scripts/models/vader/split_train_test.py`
- `scripts/models/vader/train_models.py`
- `scripts/models/vader/evaluate_models.py`
- `scripts/models/roberta/split_train_test.py`
- `scripts/models/roberta/train_models.py`
- `scripts/models/roberta/evaluate_models.py`
- `data/results/`
- `data/models/`

### 8. Dashboard and Live URL Analysis

The Streamlit dashboard shows metrics and supports live analysis of a public Facebook post URL. A user can paste a URL, select a fetch limit, and run analysis directly from the dashboard.

For each live URL analysis, the system:

1. Fetches comments using Apify.
2. Saves the analyzed URL in a link registry.
3. Saves raw fetched comments.
4. Cleans the comments.
5. Applies relevance, sarcasm, sentiment, emotion, and model prediction steps.
6. Saves one analysis CSV for that URL.
7. Appends all analyzed rows to a cumulative history file.
8. Appends relevant rows to a cumulative training-candidate file.

Current live URL analysis snapshot:

- 1 analyzed URL.
- 576 cumulative analyzed rows.
- 223 relevant candidate rows for possible future training review.

Key files:

- `src/dashboard/app.py`
- `src/dashboard/live_analysis.py`
- `data/raw/url_links.csv`
- `data/raw/url_fetches/`
- `data/results/url_analyses/`
- `data/processed/labeled/url_analysis_history.csv`
- `data/processed/labeled/url_training_candidates.csv`

## Current Repository Status

The project is maintained on the `master` branch and connected to GitHub. The latest saved URL-analysis outputs are now part of the local repository history.

Latest saved URL-analysis commit:

- `d749dbc Add saved URL analysis outputs`

Automated tests currently pass:

- 33 tests passed.

## What Is Working Now

- The project has an end-to-end pipeline from Facebook comments to analyzed sentiment outputs.
- The dashboard can run live analysis on a public Facebook post URL.
- Relevance filtering keeps the training data focused on energy-service comments.
- Sarcasm detection reduces misclassification of sarcastic praise.
- Local sentiment and emotion lexicons make the system more appropriate for Zambian/code-switched comments.
- The project has saved model metrics, confusion matrices, and classification reports.
- The codebase has tests for major sentiment, sarcasm, preprocessing, model, and live-analysis behavior.

## Current Limitations

- The local lexicons are manually built and should be reviewed against manually annotated comments.
- Sarcasm detection is heuristic, so it can miss subtle sarcasm or over-detect some patterns.
- The relevance filter is keyword-driven and may need refinement for edge cases.
- The labeled data is generated through rule/model-assisted methods, not a large manually annotated gold-standard dataset.
- The live URL analysis depends on Apify access and public availability of Facebook post comments.

## Suggested Supervisor Discussion Points

1. Confirm whether the current methodology is acceptable: VADER plus local correction, sarcasm heuristics, NRC/local emotion analysis, and baseline ML models.
2. Ask whether a small manually annotated validation set should be created for stronger evaluation.
3. Discuss whether the RoBERTa-labeled branch should become the main model path because it currently gives better metrics.
4. Review whether the local sentiment and emotion lexicons need expert/manual validation.
5. Decide how much emphasis the final write-up should place on dashboard functionality versus model performance.
6. Confirm whether live Facebook URL analysis should remain part of the final prototype demonstration.

## Short Demo Script

1. Open the dashboard:

```powershell
streamlit run src/dashboard/app.py
```

2. Show the existing evaluation metrics and confusion matrices.
3. Explain that the system trains on relevant comments only.
4. Show examples of sarcasm-aware and local-lexicon sentiment correction.
5. Paste a public Facebook post URL into the live analysis section.
6. Run analysis and show the saved outputs:

- Raw fetched comments.
- Per-link analyzed CSV.
- Cumulative analysis history.
- Relevant training candidates.

## One-Sentence Explanation

This prototype analyses ZESCO-related Facebook comments by collecting them, cleaning them, filtering for relevance, applying sarcasm-aware and locally adapted sentiment/emotion analysis, training baseline models, and presenting both stored and live URL results through a dashboard.
