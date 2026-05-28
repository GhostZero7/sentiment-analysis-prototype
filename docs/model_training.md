# Model Training

This project trains classical sentiment models on the finalized labeled dataset:

- `data/processed/labeled/final_label.csv`
- optional English-only variant: `data/processed/labeled/final_label_english_only.csv`

## Training Flow

1. Load the labeled CSV.
2. Use `processed_text` as the input feature text.
3. Use `corrected_label` as the target label when available.
4. Split the dataset with stratified sampling:
   - 80% train
   - 20% test
5. Fit a shared `TfidfVectorizer`.
6. Train three baseline models:
   - Naive Bayes
   - Logistic Regression
   - Linear SVM
7. Evaluate on the held-out 20% test set.
8. Save models and metrics to disk.

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
python scripts/train_models.py --input data/processed/labeled/final_label.csv
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
