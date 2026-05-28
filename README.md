# Sentiment Analysis Prototype

## Overview

This prototype now includes:

- Apify-based Facebook comment collection
- preprocessing and sarcasm-aware sentiment labeling
- persistent train/test dataset splits
- baseline model training with Naive Bayes, Logistic Regression, and SVM
- evaluation reports with confusion matrices
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

1. Use `scripts/split_train_test.py` to create:
   - `data/processed/labeled/final_label_train.csv`
   - `data/processed/labeled/final_label_test.csv`
2. Train models:
   - `python scripts/train_models.py --train-input data/processed/labeled/final_label_train.csv --test-input data/processed/labeled/final_label_test.csv`
3. Evaluate models:
   - `python scripts/evaluate_models.py --test-input data/processed/labeled/final_label_test.csv`
4. Launch the dashboard:
   - `streamlit run src/dashboard/app.py`
