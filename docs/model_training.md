# Model Training

This project trains classical sentiment models on the finalized labeled dataset:

- `data/processed/labeled/final_label.csv`
- optional English-only variant: `data/processed/labeled/final_label_english_only.csv`

## Training Flow

1. Load the persisted train and test CSV files.
2. Use `processed_text` as the input feature text.
3. Use `corrected_label` as the target label when available.
4. Fit a shared `TfidfVectorizer` on the training split.
5. Train three baseline models:
   - Naive Bayes
   - Logistic Regression
   - Linear SVM
6. Evaluate on the held-out test split.
7. Save models and metrics to disk.

## Persistent Train/Test Files

You can also persist the split as separate CSV files:

- `data/processed/labeled/final_label_train.csv`
- `data/processed/labeled/final_label_test.csv`

Create them with:

```powershell
python scripts/split_train_test.py --input data/processed/labeled/final_label.csv
```

## Training Command

```powershell
python scripts/train_models.py --train-input data/processed/labeled/final_label_train.csv --test-input data/processed/labeled/final_label_test.csv
```

## Prediction Command

```powershell
python src/models/predict.py --text "sample comment here"
```

## Saved Artifacts

Training writes artifacts to:

- `data/models/vectorizer.joblib`
- `data/models/naive_bayes.joblib`
- `data/models/logistic_regression.joblib`
- `data/models/svm.joblib`

Evaluation outputs are written to:

- `data/results/model_metrics.csv`
- `data/results/training_summary.json`
