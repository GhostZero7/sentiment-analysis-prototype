# Recommended Next Features

This document lists practical next features for improving the sentiment analysis prototype after the current end-to-end pipeline.

## 1. Manual Annotation Review Set

Create a small manually reviewed dataset of comments with human-approved labels for:

- Relevance
- Sentiment
- Sarcasm
- Main emotion

Why it matters:

- Gives the project a stronger evaluation baseline.
- Helps validate whether VADER, RoBERTa labels, sarcasm rules, and local lexicons match human judgment.
- Makes supervisor/examiner discussions stronger because results can be compared against a human-labeled reference set.

Suggested first version:

- 200 to 500 comments.
- Mix of clearly negative, neutral, positive, sarcastic, and code-switched comments.
- Store as `data/processed/annotation/manual_review_sample.csv`.

## 2. Annotation Dashboard

Add a simple Streamlit page for reviewing and correcting labels.

Useful fields:

- Original comment
- Cleaned comment
- Current predicted sentiment
- Corrected sentiment dropdown
- Sarcasm checkbox
- Relevance checkbox
- Emotion dropdown
- Notes field

Why it matters:

- Turns model improvement into a repeatable workflow.
- Lets reviewed live URL analysis rows become approved training data.
- Reduces the need to edit CSV files manually.

## 3. Training Candidate Promotion

Add a controlled process for promoting rows from `url_training_candidates.csv` into the main training dataset.

Suggested flow:

1. Fetch and analyze a Facebook URL.
2. Save relevant rows to training candidates.
3. Manually review candidate labels.
4. Promote approved rows into the training dataset.
5. Retrain and compare model metrics.

Why it matters:

- Keeps the model improving as new public discussions are analyzed.
- Prevents noisy auto-labeled rows from entering the training set without review.

## 4. Model Comparison Page

Add a dashboard page comparing VADER-based and RoBERTa-labeled branches.

Include:

- Accuracy
- F1 score
- Confusion matrices
- Class distribution
- Example disagreements

Why it matters:

- The RoBERTa branch currently performs better, but the VADER branch is more explainable.
- A comparison page makes the tradeoff visible.

## 5. Error Analysis Report

Generate an error-analysis CSV or dashboard section showing misclassified test comments.

Useful columns:

- Original comment
- True label
- Predicted label
- Sarcasm flag
- Local sentiment terms
- Local emotion terms
- Relevance score

Why it matters:

- Makes model weaknesses easier to inspect.
- Helps identify missing local lexicon terms and weak sarcasm patterns.

## 6. Lexicon Review Tools

Add a small report showing how often local sentiment and emotion terms are used.

Include:

- Most frequent matched local terms
- Terms linked to label changes
- Terms linked to sarcasm cases
- Terms with no current matches

Why it matters:

- Helps refine the local lexicons using evidence from the dataset.
- Makes it easier to justify the local-language adaptation in the methodology.

## 7. Trend Analysis Over Time

Use collection timestamps or post dates to show sentiment and emotion trends over time.

Possible views:

- Negative sentiment trend
- Frustration trend
- Sarcasm frequency trend
- Topic/relevance volume trend

Why it matters:

- Moves the project from static classification toward monitoring public sentiment.
- Useful for demonstrating how the system could support service-feedback tracking.

## 8. Topic Clustering

Add topic grouping for relevant comments.

Possible topic groups:

- Load shedding
- Tariffs
- Fault reporting
- Customer service
- Connection issues
- Praise or appreciation

Why it matters:

- Sentiment alone says how people feel; topics explain what they are reacting to.
- This would make the dashboard more useful for decision-making.

## Recommended Priority Order

1. Manual annotation review set.
2. Annotation dashboard.
3. Training candidate promotion.
4. Error analysis report.
5. Model comparison page.
6. Lexicon review tools.
7. Trend analysis over time.
8. Topic clustering.

The strongest next step is the manual annotation review set, because it improves evaluation quality and gives a clearer foundation for all later model improvements.
