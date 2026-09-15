# Project Defence: Model Questions and Answers

## Thirty-Second Opening

This project developed a locally adapted sentiment-analysis prototype for public Facebook
comments about ZESCO and electricity-service experiences in Zambia. It collects public
top-level comments, removes unnecessary personal references, filters comments for energy
relevance, detects sentiment, sarcasm, topics, and emotions, and presents explainable
findings through a mobile-responsive Streamlit dashboard. Its main contribution is adapting
standard social-media sentiment methods to Zambian and code-switched expressions while
preserving an auditable link between dashboard findings and source comments.

## Core Project Questions

### 1. What problem does the project solve?

It helps analysts summarize large volumes of public electricity-service feedback that would
be slow and inconsistent to review manually. The system highlights sentiment, emotions,
topics, and changes across discussions, while keeping the underlying comments available as
evidence.

### 2. What is the main objective?

The main objective is to design and evaluate an explainable sentiment-analysis prototype for
Zambian electricity discourse that can process public Facebook comments and communicate
policy-relevant findings through an accessible dashboard.

### 3. Why focus on ZESCO and electricity comments?

Electricity reliability, tariffs, faults, and load shedding directly affect households and
businesses. These topics generate substantial online discussion, making them a relevant case
for testing whether locally adapted sentiment analysis can support public-service monitoring.

### 4. Why use Facebook comments?

Facebook contains unsolicited and timely public reactions to announcements. It is useful for
studying expressed concerns, but the project does not claim that Facebook users represent the
whole Zambian population.

### 5. What is the end-to-end system flow?

A public Facebook URL is submitted, Apify retrieves top-level comments, the system cleans and
anonymises the text, filters for energy relevance, applies sentiment, sarcasm, emotion, topic,
and machine-learning analysis, saves the result, and presents aggregated evidence through the
dashboard and downloadable reports.

### 6. What makes the project locally relevant?

It adds Zambian and code-switched sentiment and emotion lexicons, preserves readable context
for negation, and introduces local sarcasm patterns and energy-specific relevance terms. This
reduces errors caused by applying a generic English tool without adaptation.

### 7. What is the main contribution or novelty?

The contribution is the integration of locally adapted lexical rules, explainable relevance
and topic taxonomies, sarcasm-aware corrections, classical models, and evidence-linked
visual reporting in one working Zambian energy-discourse prototype.

## Data and Ethics Questions

### 8. How was the data collected?

The system uses Apify to collect publicly accessible, top-level comments from supplied
Facebook post URLs. Each request is capped at 1,000 comments, and cached comments are reused
to reduce repeated collection and token consumption.

### 9. Why exclude replies?

Replies often contain side conversations, repeated names, and context that is not directly
about the original post. Restricting the study to top-level comments gives a clearer and more
consistent unit of analysis.

### 10. How does the system address privacy?

It analyzes public comments, does not intentionally store commenter profile names, removes
structured mentions and `@` mentions, reports aggregated findings, and exposes comments only
as supporting evidence. It should still be treated as a research prototype governed by the
institution's ethics requirements.

### 11. Is the dataset representative of Zambia?

No. It represents people who chose to comment on selected public Facebook posts. The findings
describe observed online discourse and must not be presented as population-level opinion or
as a substitute for a representative survey.

### 12. How are irrelevant comments handled?

An explainable keyword-based relevance stage identifies electricity, ZESCO, load-shedding,
tariff, infrastructure, and related terms. Irrelevant comments are retained for audit but are
excluded from model training and the default policy-facing analysis.

## Preprocessing and Feature Questions

### 13. What preprocessing is performed?

The pipeline removes URLs and mentions, handles emojis, normalizes text, tokenizes, removes
stop words where appropriate, lemmatizes tokens, and preserves readable text for rules that
depend on negation, phrases, or sarcasm.

### 14. Why preserve negation and readable text?

Removing words such as “not” can reverse meaning. For example, “power is not stable” must not
be reduced to “power stable.” The system therefore keeps a context-preserving text form for
sentiment rules while also generating processed text for TF-IDF models.

### 15. How are code-switched expressions handled?

Local lexicon CSV files map Zambian or code-switched words and phrases to sentiment and
emotion weights. Matches are recorded in explanatory columns so a reviewer can see why a
local correction was made.

### 16. How is sarcasm detected?

The prototype uses transparent heuristic patterns, including gratitude followed by a
complaint, positive wording combined with laughter, mocked promises, rhetorical questions,
and selected local intensifiers. Sarcasm can change apparently positive language to negative
and can suppress false hope or trust scores.

## Sentiment and Emotion Questions

### 17. Why use VADER?

VADER is designed for short, informal social-media text and provides an interpretable compound
sentiment score. It is lightweight enough for an interactive prototype, and its limitations
can be addressed through explicit local correction rules.

### 18. What labels does the system produce?

Sentiment is classified as positive, neutral, or negative. The system also estimates anger,
fear, trust, sadness, hope, and frustration, and assigns one explainable energy-policy topic
to each relevant comment.

### 19. Why include emotions in addition to sentiment?

Polarity alone cannot distinguish anger from fear or frustration from sadness. Emotion scores
provide more useful context for communication and service-response decisions.

### 20. What topics are detected?

The taxonomy includes load shedding and reliability, tariffs and affordability, renewable
energy and solar, customer service and communication, faults and infrastructure, governance
and trust, jobs and economic impact, environment and climate, and other energy concerns.

## Machine-Learning Model Questions

### 21. Which machine-learning models were trained?

The project trains Multinomial Naive Bayes, Logistic Regression, and a linear Support Vector
Machine. They are appropriate, well-understood baselines for sparse TF-IDF text features.

### 22. What is TF-IDF and why was it used?

TF-IDF converts text into numerical features by increasing the importance of terms that are
frequent in a document but less common across the corpus. It is efficient, interpretable, and
works well with classical text classifiers.

### 23. How was data leakage prevented?

Training and testing are separate persisted files. The vectorizer and models are fitted only
on the 80% training split, while metrics and confusion matrices are generated using the held-
out 20% test split. The split uses a fixed random state for reproducibility.

### 24. What is the class distribution?

The relevant VADER-labeled dataset contains 2,813 comments: 1,011 negative, 904 neutral, and
898 positive. The classes are not perfectly equal, so weighted precision, recall, and F1 are
reported alongside accuracy.

### 25. What were the VADER-branch model results?

On the held-out relevant-comment test split, Naive Bayes achieved 55.06% accuracy and 53.70%
weighted F1; Logistic Regression achieved 58.26% accuracy and 58.03% weighted F1; and SVM
achieved 58.44% accuracy and 58.29% weighted F1. SVM was marginally strongest on both measures.

### 26. What were the optional RoBERTa-labeled branch results?

The alternate branch produced 66.61% accuracy for Naive Bayes, 72.29% for Logistic Regression,
and 71.23% for SVM. Its weighted F1 scores were 56.32%, 69.59%, and 69.89%, respectively.
These results come from a different labeling branch and should not be presented as a direct,
controlled replacement for the corrected VADER branch without fresh common-label evaluation.

### 27. Which model is best?

For the corrected VADER-labeled branch used by the main workflow, linear SVM is the strongest
baseline by a small margin. In the optional RoBERTa-labeled branch, Logistic Regression has
the highest accuracy while SVM has the highest weighted F1. “Best” therefore depends on the
labeling branch and evaluation measure.

### 28. Why is accuracy alone insufficient?

Accuracy can hide poor performance on a smaller class. Weighted precision, recall, F1, the
classification report, and confusion matrices show whether a model performs consistently
across positive, neutral, and negative classes.

### 29. Why are the baseline scores not higher?

The task contains code-switching, spelling variation, sarcasm, short ambiguous comments, and
automatically assisted labels rather than a large expert-annotated gold standard. The scores
are honest baseline results and motivate better annotation and contextual modeling.

### 30. Why not deploy only RoBERTa?

The prototype prioritizes transparency, modest computing requirements, and reproducibility.
The RoBERTa branch is retained as an experiment, but the locally corrected VADER workflow and
classical models are easier to explain and run during a live defence. A future controlled
comparison on one human-reviewed test set would support a stronger deployment decision.

## Dashboard and Trend Questions

### 31. What does the dashboard provide?

It provides URL analysis, a sentiment and emotion overview, trend charts, explainable topics,
comment-level evidence, and downloadable stakeholder reports. It is designed for analysts and
communicators rather than model developers.

### 32. How does five-link trend tracking work?

The Trends tab loads up to five distinct saved links, orders them by Facebook metadata dates,
and compares sentiment percentages across those discussions. It uses the post-publication
timestamp when available and otherwise the earliest valid Facebook comment timestamp. It does
not call Apify and never substitutes the date on which the app performed the analysis.

### 33. Can the system claim that an event caused sentiment to change?

No. A post title, topic, or timing can help explain what was being discussed, but the chart
shows association rather than causation. External operational evidence would be required to
attribute a change to a tariff decision, outage, or policy event.

### 34. How is the mobile interface supported?

At phone widths, multi-column sections stack vertically, tab navigation becomes horizontally
scrollable, touch controls use larger targets, headings and page padding shrink, and wide
tables remain scrollable. The analytical content remains the same as on desktop.

### 35. What happens when Apify fails?

Technical details are logged on the server, while the user sees a short, safe message. Apify
failures are presented as token depletion as requested, and cached analyses remain available
without making another Apify call.

## Evaluation, Limitations, and Future Work

### 36. How was the software validated?

Automated tests cover collection limits, reply exclusion, caching, preprocessing, relevance,
local sentiment, sarcasm, emotions, model utilities, reports, saved-link trends, metadata-date
selection, and user-friendly errors. The final interface is also checked at desktop and mobile
viewports.

### 37. What are the main limitations?

The main limitations are selected-post sampling, automatically assisted labels, heuristic
sarcasm and relevance rules, incomplete coverage of local language, dependence on public
Facebook accessibility and Apify, and ephemeral file storage on a free Render deployment.

### 38. What is the most important future improvement?

Create a human-annotated validation set of 200–500 diverse comments with agreement between
multiple reviewers. That would provide a stronger gold standard for relevance, sentiment,
sarcasm, and emotion evaluation.

### 39. How should production persistence be improved?

Move link history and analyzed comments from local CSV files to persistent storage such as a
managed PostgreSQL database or a paid persistent disk. This is necessary for dependable trend
tracking across restarts and month-long monitoring periods.

### 40. What is the final conclusion?

The project demonstrates that a locally adapted and explainable pipeline can convert noisy
public electricity discourse into structured sentiment, emotion, topic, and trend evidence.
It is suitable as a completed research prototype, while its documented limitations define a
clear path toward stronger validation and production deployment.

## Short Demonstration Sequence

1. Paste a public Facebook post URL and select the comment limit.
2. Explain the cache check and top-level-comment restriction.
3. Show sentiment, dominant emotion, and leading topic in Overview.
4. Show the current discussion pattern and five-link comparison in Trends.
5. Open Topics and Comments to demonstrate explainability.
6. Download the stakeholder report.
7. End by stating the sampling, causality, annotation, and persistence limitations.

## Answers to Avoid

- Do not say that Facebook comments represent all Zambians.
- Do not claim that the system proves an external event caused a sentiment change.
- Do not describe automatically assisted labels as a human-annotated gold standard.
- Do not compare VADER-branch and RoBERTa-branch scores as if only the classifier changed.
- Do not promise that free Render storage preserves month-to-month history.
- Do not describe the dashboard's policy considerations as automatic policy decisions.
