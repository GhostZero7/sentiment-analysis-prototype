# Sentiment Analysis Prototype

## Overview

This prototype now includes:

- Apify-based Facebook comment collection
- preprocessing and sarcasm-aware sentiment labeling
- relevance review files for separating energy-related and unrelated comments
- CSV-based local Zambian sentiment lexicon for code-switched comments
- NRC emotion scoring for anger, hope, fear, trust, and frustration
- persistent train/test dataset splits
- baseline model training with Naive Bayes, Logistic Regression, and SVM
- evaluation reports with confusion matrices
- an alternate RoBERTa-labeled training branch
- a Streamlit dashboard for metrics and live predictions

## Step 1 Implemented: Apify Data Collection

This stage includes Facebook URL validation and comment collection through Apify.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Edit `.env` and set `APIFY_API_KEY`.

### Run Data Download

```bash
python scripts/collection/download_data.py --url "https://www.facebook.com/..."
```

Output will be written to `data/raw/comments.csv` by default.

## Local Lexicon

The local sentiment lexicon is stored at:

- `data/lexicons/local_sentiment_lexicon.csv`

It contains Zambian/code-switched terms, meanings, sentiment type, weights, and notes. The VADER analyzer loads this CSV during labeling and records matched terms in `local_correction_terms`, making the correction layer explainable.

## Relevance Review

Create files for manually checking whether comments are actually about ZESCO, electricity, load shedding, tariffs, or energy-service experience:

```powershell
python scripts/preprocessing/filter_relevance.py
```

This writes:

- `data/processed/relevance/all_comments_relevance.csv`
- `data/processed/relevance/relevant_comments.csv`
- `data/processed/relevance/irrelevant_comments.csv`

These files are for review first. Training still uses `final_label.csv` until the relevance filter has been inspected and approved.

## Current Training Flow

Scripts are grouped by purpose:

- `scripts/collection/`
- `scripts/preprocessing/`
- `scripts/sentiment/`
- `scripts/models/`
- `scripts/models/vader/`
- `scripts/models/roberta/`
- `scripts/models/shared/` for shared evaluation helpers

1. Use `scripts/models/vader/split_train_test.py` to create:
   - `data/processed/labeled/training/final_label_train.csv`
   - `data/processed/labeled/testing/final_label_test.csv`
2. Train models on the 80% training split only:
   - `python scripts/models/vader/train_models.py`
3. Evaluate models on the 20% testing split:
   - `python scripts/models/vader/evaluate_models.py`
4. Launch the dashboard:
   - `streamlit run src/dashboard/app.py`

Training no longer prints accuracy/F1 because it does not touch the testing data. Run the evaluation script whenever you want the test results and confusion matrices.

## Optional RoBERTa Branch

If you want to generate a separate RoBERTa-labeled dataset and train a second model branch without touching the VADER flow:

```powershell
python scripts/sentiment/label_roberta.py
python scripts/models/roberta/split_train_test.py
python scripts/models/roberta/train_models.py
python scripts/models/roberta/evaluate_models.py
```
