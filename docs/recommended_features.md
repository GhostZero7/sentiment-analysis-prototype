# Product Direction and Future Features

The prototype is currently focused on an end-user dashboard for policy analysts and
energy communicators. Model training, annotation, and diagnostic tools belong to a
separate future developer/admin workflow.

## Implemented End-User Features

### Trend Analysis

The dashboard groups comment timestamps by day, week, or month and shows:

- positive, neutral, and negative sentiment movement;
- comment volume;
- anger, frustration, hope, and trust movement;
- sarcasm frequency;
- a plain-language description of the negative-sentiment direction.

The dashboard states when only one time period is available because a snapshot is not
enough to establish a trend.

### Topic Grouping

Comments receive one primary topic from a transparent Zambia-energy taxonomy:

- load shedding and reliability;
- tariffs and affordability;
- renewable energy and solar;
- customer service and communication;
- faults, connections, and infrastructure;
- governance and public trust;
- jobs and economic impact;
- environment and climate;
- other energy concerns.

Each topic includes volume, share of discussion, sentiment distribution, leading
emotion, and representative comments. The taxonomy is deliberately auditable and can
be refined when new local expressions appear.

### Stakeholder Report

The dashboard generates a downloadable Markdown report containing:

- analysis scope and coverage;
- executive summary;
- sentiment distribution;
- dominant topics;
- emotion index;
- trend interpretation;
- evidence-linked policy and communication considerations;
- interpretation limits.

### Policy and Communication Considerations

Recommendations are generated from aggregated evidence. Every recommendation includes
the topic volume, discussion share, negative-sentiment percentage, and frustration
index when available.

These outputs are decision-support considerations. They do not automatically prescribe
policy, and the report states that Facebook comments are not a representative population
survey.

## Future End-User Features

1. Add an energy-event timeline so trend changes can be compared with tariff decisions,
   load-shedding announcements, renewable-energy launches, and regulatory changes.
2. Add PDF export for formal stakeholder circulation while keeping the Markdown report
   as the transparent source format.
3. Add comparison across multiple Facebook posts, institutions, or monitoring periods.
4. Add saved report snapshots so analysts can compare one monitoring cycle with another.
5. Add configurable alert thresholds for rapid increases in negative sentiment,
   frustration, or a high-priority topic.

## Future Developer/Admin Features

### Manual Annotation Review Set

Create a human-reviewed sample of 200 to 500 comments covering relevance, sentiment,
sarcasm, and main emotion. This will provide a stronger evaluation baseline.

### Annotation Dashboard

Create a separate admin interface for correcting automated labels and adding review
notes. This interface must not appear in the end-user dashboard.

### Training Candidate Promotion

Allow reviewed rows from `url_training_candidates.csv` to be approved before they enter
the main training dataset.

### Error and Model Comparison Tools

Add developer reports for misclassified comments, model disagreements, confusion
matrices, and comparison of the VADER-based and RoBERTa-labeled branches.

### Lexicon Review Tools

Show which local sentiment and emotion terms are frequently matched, change labels, or
have no current examples. This will support evidence-based refinement of the local
lexicons.

## Priority

The current priority is completing and evaluating the end-user insight experience.
Manual annotation and the annotation dashboard remain the strongest next development
steps once the user-facing prototype is stable.
