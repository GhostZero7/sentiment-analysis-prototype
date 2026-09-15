# End-User Dashboard Guide

## Start the Dashboard

From the project directory, run:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard opens on the **Analyze URL** tab. No research or saved dataset is shown
to end users.

## Analyze URL

Paste a publicly accessible Facebook post URL, choose between 10 and 1,000 comments,
and select **Analyze public comments**.

The system checks the local raw dataset before contacting Apify. If that post or video
has already been fetched, its saved comments are analyzed locally and the dashboard
confirms that no Apify tokens were used. Apify is contacted only for an unseen URL, and
all requests are limited to 1,000 top-level comments. Replies are excluded so reply names
and side conversations do not distort the post-level sentiment result.

After analyzing multiple links, open the **Trends** tab and select **Track trend**. The
system loads up to five distinct links ordered by their saved Facebook metadata dates.
It uses the post-publication timestamp when available and otherwise the earliest valid
Facebook comment timestamp. It never uses the analysis or collection date for the trend.
The comparison uses saved relevant comments and does not contact Apify. The Facebook post
title is shown when it is available from the public post metadata.

For comparisons collected over weeks or months, the deployed app must use persistent
storage. A free Render web service does not preserve locally written analysis files across
all restarts, redeploys, and spin-downs.

When analysis finishes, the Overview, Trends, Topics, and Report tabs show results only
for that URL. Analyzing another URL replaces the current results. Use **Clear current
analysis** in the sidebar to return to the empty state.

The sidebar can restrict the current URL results to relevant energy comments and an
available date range.

On phones, dashboard sections stack vertically, tabs can be swiped horizontally, buttons
expand to the available width, and wide evidence tables remain horizontally scrollable.

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

The Trends tab can group results automatically or by hour, day, week, or month. When
calendar timing is unavailable, it compares three ordered stages: beginning, middle, and
recent comments. A 100% stacked bar shows the negative, neutral, and positive share in each
stage or period. A plain-language explanation states exactly how sentiment changed, while a
beginning-versus-latest table shows each change in percentage points. The readable trend
table shows only comment count, sentiment shares, leading sentiment, and leading emotion;
technical NRC columns and equal-sized comment-volume charts are intentionally omitted.

For calendar trends, a separate explanation identifies the strongest change and connects it
to the leading topic, emotion, and an example public comment. This makes the chart suitable
for presentation while preserving supporting evidence.

These explanations identify events within the discussion. They do not claim that an
external real-world event caused a sentiment change unless separate evidence verifies it.

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
