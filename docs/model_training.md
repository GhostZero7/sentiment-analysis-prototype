# Model Training

This project trains classical sentiment models on the finalized labeled dataset:

- `data/processed/labeled/final_label_relevant.csv`

The full labeled corpus remains available at:

- `data/processed/labeled/final_label.csv`

The excluded review set is:

- `data/processed/labeled/final_label_irrelevant.csv`

## Training Flow

1. Load the persisted training CSV file.
2. Use `processed_text` as the input feature text.
3. Use `corrected_label` as the target label when available.
4. Fit a shared `TfidfVectorizer` on the training split.
5. Train three baseline models:
   - Naive Bayes
   - Logistic Regression
   - Linear SVM
6. Save models and the vectorizer to disk.
7. Run the separate evaluation script when you want test metrics.

The labeled files also now carry NRC emotion scores so the dataset retains both polarity and emotion features for analysis and future modeling.

## Evaluation Report

Generate confusion matrices and a metrics summary from the held-out test split:

```powershell
python scripts/models/vader/evaluate_models.py
```

This writes:

- `data/results/evaluation_summary.csv`
- `data/results/classification_reports.json`
- `data/results/confusion_matrices/*.png`

## Persistent Train/Test Files

You can also persist the split as separate CSV files:

- `data/processed/labeled/training/final_label_train.csv`
- `data/processed/labeled/testing/final_label_test.csv`

Create them with:

```powershell
python scripts/models/vader/split_train_test.py
```

## Training Command

```powershell
python scripts/models/vader/train_models.py
```

This command trains only. It does not read the testing file and does not print accuracy or F1 scores.

## Prediction Command

```powershell
python scripts/models/vader/predict_models.py --text "sample comment here"
```

## Dashboard

Launch the Streamlit dashboard after training:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard shows:

- model metrics
- a live text prediction form
- confusion matrix images if they have been generated

## Saved Artifacts

Training writes artifacts to:

- `data/models/vectorizer.joblib`
- `data/models/naive_bayes.joblib`
- `data/models/logistic_regression.joblib`
- `data/models/svm.joblib`
- `data/results/training_summary.json`

Evaluation outputs are written to:

- `data/results/evaluation_summary.csv`
- `data/results/classification_reports.json`
- `data/results/confusion_matrices/*.png`

## Alternate RoBERTa-Labeled Path

The VADER path remains the default. If you later produce a separate RoBERTa-labeled dataset, the pipeline is already scaffolded to train on that path without changing the VADER flow.

Expected alternate files:

- `data/processed/labeled/final_label_roberta_relevant.csv`
- `data/processed/labeled/training/final_label_roberta_train.csv`
- `data/processed/labeled/testing/final_label_roberta_test.csv`

Helper scripts:

- `scripts/sentiment/label_roberta.py`
- `scripts/models/roberta/split_train_test.py`
- `scripts/models/roberta/train_models.py`
- `scripts/models/roberta/evaluate_models.py`

Create the RoBERTa-labeled dataset with:

```powershell
python scripts/sentiment/label_roberta.py
```

Then split and train the RoBERTa branch separately:

```powershell
python scripts/models/roberta/split_train_test.py
python scripts/models/roberta/train_models.py
python scripts/models/roberta/evaluate_models.py
```

As with the VADER path, RoBERTa training only saves model artifacts. RoBERTa testing is done by `scripts/models/roberta/evaluate_models.py`.
