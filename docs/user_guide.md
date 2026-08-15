# End-User Dashboard Guide

## Start the Dashboard

From the project directory, run:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard opens on the **Analyze URL** tab. No research or saved dataset is shown
to end users.

## Analyze URL

Paste a publicly accessible Facebook post URL, choose the maximum number of comments,
and select **Analyze public comments**.

The system checks the local raw dataset before contacting Apify. If that post or video
has already been fetched, its saved comments are analyzed locally and the dashboard
confirms that no Apify tokens were used. Apify is contacted only for an unseen URL, and
all requests are limited to 300 top-level comments. Replies are excluded so reply names
and side conversations do not distort the post-level sentiment result.

When analysis finishes, the Overview, Trends, Topics, and Report tabs show results only
for that URL. Analyzing another URL replaces the current results. Use **Clear current
analysis** in the sidebar to return to the empty state.

The sidebar can restrict the current URL results to relevant energy comments and an
available date range.

## Overview

The Overview tab shows:

- number of comments analyzed;
- percentage of negative sentiment;
- most discussed topic;
- leading detected emotion;
- sentiment and emotion charts;
- priority policy and communication considerations.

Every consideration includes the supporting topic volume and sentiment evidence.

## Trends

The Trends tab groups results by day, week, or month. It shows sentiment percentages,
comment volume, selected emotions, sarcasm frequency, and a plain-language summary of
the negative-sentiment direction.

When the selected data contains only one time period, treat it as a snapshot rather than
evidence that sentiment is rising or falling.

## Topics

The Topics tab groups comments into an explainable Zambia-energy taxonomy. Select a
topic to review representative comments and their sentiment labels.

The topic taxonomy supports policy interpretation but does not replace human review.
Comments that do not contain enough topic evidence remain under `Other energy concerns`.

## Report

The Report tab displays and downloads a stakeholder report in Markdown format. It
contains the executive summary, sentiment distribution, topics, emotions, trend
interpretation, recommendations, and methodological limitations.

The system only processes publicly accessible content and displays aggregated insights.

## Interpretation

- Facebook comments are unsolicited reactions, not a representative population survey.
- Sentiment, emotion, sarcasm, and topic results are automated estimates.
- Policy recommendations are evidence-linked decision support, not automatic decisions.
- High-impact findings should be checked against operational data and human review.
