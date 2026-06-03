# Sentiment Analysis Prototype

## Overview

This prototype now includes:

- Apify-based Facebook comment collection
- preprocessing and sarcasm-aware sentiment labeling
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
python scripts/download_data.py --url "https://www.facebook.com/..."
```

Output will be written to `data/raw/comments.csv` by default.

## Current Training Flow

Scripts are grouped by purpose:

- `scripts/collection/`
- `scripts/preprocessing/`
- `scripts/sentiment/`
- `scripts/models/`
- `scripts/models/vader/`
- `scripts/models/roberta/`

1. Use `scripts/models/vader/split_train_test.py` to create:
   - `data/processed/labeled/training/final_label_train.csv`
   - `data/processed/labeled/testing/final_label_test.csv`
2. Train models:
   - `python scripts/models/vader/train_models.py`
3. Evaluate models:
   - `python scripts/models/vader/evaluate_models.py`
4. Launch the dashboard:
   - `streamlit run src/dashboard/app.py`

## Optional RoBERTa Branch

If you want to generate a separate RoBERTa-labeled dataset and train a second model branch without touching the VADER flow:

```powershell
python scripts/sentiment/label_roberta.py
python scripts/models/roberta/split_train_test.py
python scripts/models/roberta/train_models.py
python scripts/models/roberta/evaluate_models.py
```
