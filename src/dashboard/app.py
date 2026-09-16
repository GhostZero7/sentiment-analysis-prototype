"""End-user dashboard for Zambian green-energy public sentiment insights."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.live_analysis import analyze_facebook_url, load_recent_link_trends
from src.data_collection.apify_client import ApifyFetchError
from src.insights.policy_report import (
    build_policy_recommendations,
    build_stakeholder_report,
)
from src.insights.pdf_report import build_executive_summary_pdf
from src.insights.topic_analyzer import (
    EMOTION_COLUMNS,
    add_topic_labels,
    summarize_topics,
)
from src.temporal.event_tracker import (
    add_event_time,
    build_comment_progression,
    build_sentiment_trends,
    build_trend_events,
    find_timestamp_column,
    resolve_trend_frequency,
)


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TOPIC_DISPLAY_NAMES = {
    "Load shedding and reliability": "Load shedding & reliability",
    "Tariffs and affordability": "Tariffs & affordability",
    "Renewable energy and solar": "Renewable energy & solar",
    "Customer service and communication": "Customer service & communication",
    "Faults, connections and infrastructure": "Faults & infrastructure",
    "Governance and public trust": "Governance & trust",
    "Jobs and economic impact": "Jobs & economic impact",
    "Environment and climate": "Environment & climate",
    "Other energy concerns": "Other energy concerns",
}
SENTIMENT_COLORS = {
    "Negative": "#D94B4B",
    "Neutral": "#9AA3AF",
    "Positive": "#2E8B57",
}
EMOTION_COLORS = {
    "Anger": "#D94B4B",
    "Fear": "#7A5AF8",
    "Trust": "#1F9D8A",
    "Sadness": "#4C78A8",
    "Hope": "#2E8B57",
    "Frustration": "#E67E22",
}
LOGGER = logging.getLogger(__name__)
DASHBOARD_TAB_LABELS = ["Analyze URL", "Overview", "Trends", "Topics", "Comments", "Report"]
DASHBOARD_TAB_STATE_KEY = "active_dashboard_tab"
RESPONSIVE_CSS = """
<style>
.block-container {
    width: 100%;
    max-width: 1180px;
    padding: 2.25rem 1.5rem 2rem;
}
h1 {font-size: 2rem !important; line-height: 1.2 !important; overflow-wrap: anywhere;}
h2 {font-size: 1.3rem !important; line-height: 1.3 !important; overflow-wrap: anywhere;}
h3 {font-size: 1.08rem !important; line-height: 1.35 !important; overflow-wrap: anywhere;}
[data-testid="stMetric"] {
    border-bottom: 2px solid #E6E9EE;
    padding-bottom: 0.55rem;
    min-height: 4.15rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.28rem;
    line-height: 1.2;
    white-space: normal;
    overflow: visible;
}
[data-testid="stMetricValue"] > div {
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
}
[data-testid="stMetricLabel"] {font-size: 0.82rem; color: #5F6B7A;}
[data-testid="stDataFrame"] {font-size: 0.86rem; max-width: 100%; overflow-x: auto;}
.dashboard-text-metric {
    border-bottom: 2px solid #E6E9EE;
    padding-bottom: 0.55rem;
    min-height: 4.15rem;
}
.dashboard-text-metric-label {font-size: 0.82rem; color: #5F6B7A; margin-bottom: 0.35rem;}
.dashboard-text-metric-value {
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.25;
    color: #172033;
    overflow-wrap: anywhere;
}

@media (max-width: 768px) {
    .block-container {padding: 1.15rem 0.85rem 1.5rem !important;}
    h1 {font-size: 1.62rem !important;}
    h2 {font-size: 1.2rem !important;}
    h3 {font-size: 1rem !important;}

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 0.75rem !important;
    }
    [data-testid="column"],
    [data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }
    [data-baseweb="tab-list"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin;
        -webkit-overflow-scrolling: touch;
    }
    [data-baseweb="tab"] {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    [data-testid="stMetric"],
    .dashboard-text-metric {
        min-height: 0;
        padding: 0.25rem 0 0.65rem;
    }
    [data-testid="stMetricValue"] {font-size: 1.15rem;}
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        width: 100% !important;
        max-width: calc(100vw - 1.7rem) !important;
        overflow-x: auto !important;
    }
    [data-testid="stAlert"] {overflow-wrap: anywhere;}
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        width: 100%;
        min-height: 2.75rem;
        white-space: normal;
    }
    input, textarea, select {font-size: 16px !important;}
    iframe, canvas, svg {max-width: 100% !important;}
}

@media (max-width: 420px) {
    .block-container {padding-left: 0.65rem !important; padding-right: 0.65rem !important;}
    h1 {font-size: 1.48rem !important;}
    [data-baseweb="tab"] {padding-left: 0.6rem !important; padding-right: 0.6rem !important;}
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {max-width: calc(100vw - 1.3rem) !important;}
}
</style>
"""


def _friendly_error_message(
    error: Exception,
    *,
    fallback: str = "Something went wrong. Please try again.",
) -> str:
    """Return a short user-safe message while technical details remain in server logs."""
    if isinstance(error, ApifyFetchError):
        return error.user_message
    if isinstance(error, ValueError):
        return "Enter a valid public Facebook post URL and try again."
    if isinstance(error, FileNotFoundError):
        return "The analysis files are temporarily unavailable. Please contact the administrator."
    if isinstance(error, PermissionError):
        return "The result could not be saved. Please contact the administrator."
    return fallback


def _show_user_error(
    error: Exception,
    *,
    operation: str,
    fallback: str = "Something went wrong. Please try again.",
) -> None:
    LOGGER.error(
        "%s failed",
        operation,
        exc_info=(type(error), error, error.__traceback__),
    )
    st.error(_friendly_error_message(error, fallback=fallback))


def _render_safely(section_name: str, renderer, *args, **kwargs) -> None:
    """Render one dashboard section without exposing an exception traceback to users."""
    try:
        renderer(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        _show_user_error(
            exc,
            operation=f"Render {section_name} section",
            fallback=(
                f"The {section_name} section could not be displayed. "
                "Please analyze the URL again."
            ),
        )


def _inject_styles() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)


def _dashboard_tabs():
    """Create stateful tabs so button reruns preserve the user's selected view."""
    return st.tabs(
        DASHBOARD_TAB_LABELS,
        key=DASHBOARD_TAB_STATE_KEY,
        on_change="rerun",
    )


def _compact_bar_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    height: int = 175,
    bar_size: int = 20,
    horizontal: bool = False,
    colors_by_category: dict[str, str] | None = None,
    value_title: str | None = None,
) -> None:
    if frame.empty:
        return
    color = (
        alt.Color(
            f"{category}:N",
            legend=None,
            scale=alt.Scale(
                domain=list(colors_by_category),
                range=list(colors_by_category.values()),
            ),
        )
        if colors_by_category
        else alt.value("#267A78")
    )
    tooltip = [
        alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()),
        alt.Tooltip(f"{value}:Q", title=value_title or value.replace("_", " ").title()),
    ]
    if horizontal:
        encoding = {
            "x": alt.X(f"{value}:Q", title=value_title, axis=alt.Axis(grid=True, tickCount=5)),
            "y": alt.Y(
                f"{category}:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=190),
            ),
        }
    else:
        encoding = {
            "x": alt.X(f"{category}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            "y": alt.Y(f"{value}:Q", title=value_title, axis=alt.Axis(grid=True, tickCount=5)),
        }
    chart = (
        alt.Chart(frame)
        .mark_bar(size=bar_size, cornerRadius=3)
        .encode(**encoding, color=color, tooltip=tooltip)
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _text_metric(container, label: str, value: str) -> None:
    container.markdown(
        '<div class="dashboard-text-metric">'
        f'<div class="dashboard-text-metric-label">{escape(label)}</div>'
        f'<div class="dashboard-text-metric-value">{escape(value)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _display_periods(trends: pd.DataFrame, *, temporal: bool) -> list[str]:
    if trends.empty:
        return []
    if temporal:
        labels: list[str] = []
        for value in trends["period"]:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                labels.append(str(value))
            elif parsed.hour or parsed.minute:
                labels.append(parsed.strftime("%d %b %Y, %H:%M"))
            else:
                labels.append(parsed.strftime("%d %b %Y"))
        return labels

    raw = trends["period"].fillna("").astype(str).tolist()
    if len(raw) == 1:
        prefixes = ["Discussion"]
    elif len(raw) == 2:
        prefixes = ["Beginning", "Recent"]
    else:
        prefixes = ["Beginning", *["Middle"] * (len(raw) - 2), "Recent"]
    return [f"{prefix}: {period}" for prefix, period in zip(prefixes, raw)]


def _leading_emotion(row: pd.Series) -> str:
    scores = {
        column.removeprefix("nrc_").replace("_", " ").title(): float(
            pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce").fillna(0).iloc[0]
        )
        for column in EMOTION_COLUMNS
    }
    emotion, score = max(scores.items(), key=lambda item: item[1], default=("", 0.0))
    return emotion if score > 0 else "No strong signal"


def _readable_trend_table(trends: pd.DataFrame, *, temporal: bool) -> pd.DataFrame:
    columns = [
        "Stage or period",
        "Comments",
        "Negative %",
        "Neutral %",
        "Positive %",
        "Leading sentiment",
        "Leading emotion",
    ]
    if trends.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    display_periods = _display_periods(trends, temporal=temporal)
    for position, (_, row) in enumerate(trends.iterrows()):
        sentiment_scores = {
            "Negative": float(row.get("negative_percent", 0)),
            "Neutral": float(row.get("neutral_percent", 0)),
            "Positive": float(row.get("positive_percent", 0)),
        }
        leading_sentiment, leading_share = max(
            sentiment_scores.items(), key=lambda item: item[1]
        )
        rows.append(
            {
                "Stage or period": display_periods[position],
                "Comments": int(row.get("comment_count", 0)),
                "Negative %": round(sentiment_scores["Negative"], 1),
                "Neutral %": round(sentiment_scores["Neutral"], 1),
                "Positive %": round(sentiment_scores["Positive"], 1),
                "Leading sentiment": f"{leading_sentiment} ({leading_share:.1f}%)",
                "Leading emotion": _leading_emotion(row),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _sentiment_change_table(trends: pd.DataFrame) -> pd.DataFrame:
    columns = ["Sentiment", "Earlier %", "Latest %", "Change (points)", "Meaning"]
    if len(trends) < 2:
        return pd.DataFrame(columns=columns)

    earlier = trends.iloc[0]
    latest = trends.iloc[-1]
    rows: list[dict[str, object]] = []
    for sentiment in ("Negative", "Neutral", "Positive"):
        column = f"{sentiment.lower()}_percent"
        start = float(earlier.get(column, 0))
        end = float(latest.get(column, 0))
        change = round(end - start, 1)
        if abs(change) < 2:
            meaning = "Broadly stable"
        else:
            meaning = "Increased" if change > 0 else "Decreased"
        rows.append(
            {
                "Sentiment": sentiment,
                "Earlier %": round(start, 1),
                "Latest %": round(end, 1),
                "Change (points)": f"{change:+.1f}",
                "Meaning": meaning,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _sentiment_change_explanation(trends: pd.DataFrame, *, temporal: bool) -> str:
    if trends.empty:
        return "There are not enough comments to explain a sentiment change."
    latest = trends.iloc[-1]
    latest_scores = {
        "negative": float(latest.get("negative_percent", 0)),
        "neutral": float(latest.get("neutral_percent", 0)),
        "positive": float(latest.get("positive_percent", 0)),
    }
    latest_sentiment, latest_share = max(latest_scores.items(), key=lambda item: item[1])
    if len(trends) < 2:
        return (
            f"This is a single snapshot. {latest_sentiment.title()} sentiment is the largest "
            f"share at {latest_share:.1f}%, so a direction of change cannot yet be established."
        )

    earlier = trends.iloc[0]
    changes = {
        sentiment: round(
            float(latest.get(f"{sentiment}_percent", 0))
            - float(earlier.get(f"{sentiment}_percent", 0)),
            1,
        )
        for sentiment in ("negative", "neutral", "positive")
    }
    negative_start = float(earlier.get("negative_percent", 0))
    negative_end = float(latest.get("negative_percent", 0))
    if changes["negative"] >= 5:
        overall = "The discussion became more negative."
    elif changes["negative"] <= -5:
        overall = "The discussion became less negative."
    else:
        overall = "Negative sentiment was broadly stable."

    direction_details = []
    for sentiment in ("neutral", "positive"):
        change = changes[sentiment]
        if abs(change) < 2:
            direction_details.append(f"{sentiment} sentiment stayed broadly stable")
        else:
            direction = "rose" if change > 0 else "fell"
            direction_details.append(
                f"{sentiment} sentiment {direction} by {abs(change):.1f} points"
            )
    context = "between the first and latest period" if temporal else "from the beginning to the recent comments"
    in_ten = min(10, max(0, round(latest_share / 10)))
    return (
        f"{overall} Negative sentiment changed from {negative_start:.1f}% to "
        f"{negative_end:.1f}% ({changes['negative']:+.1f} percentage points) {context}. "
        f"Meanwhile, {direction_details[0]} and {direction_details[1]}. In the latest group, "
        f"{latest_sentiment} was the largest share at {latest_share:.1f}% - about {in_ten} in "
        "every 10 comments."
    )


def _sentiment_composition_chart(trends: pd.DataFrame, *, temporal: bool) -> None:
    if trends.empty:
        return
    display_periods = _display_periods(trends, temporal=temporal)
    chart_data = trends[
        ["negative_percent", "neutral_percent", "positive_percent"]
    ].copy()
    chart_data["Stage or period"] = display_periods
    chart_data = chart_data.melt(
        id_vars="Stage or period",
        var_name="sentiment",
        value_name="share",
    )
    chart_data["sentiment"] = chart_data["sentiment"].map(
        {
            "negative_percent": "Negative",
            "neutral_percent": "Neutral",
            "positive_percent": "Positive",
        }
    )
    order = {"Negative": 0, "Neutral": 1, "Positive": 2}
    chart_data["sentiment_order"] = chart_data["sentiment"].map(order)
    chart_data["start"] = chart_data.groupby("Stage or period", sort=False)["share"].cumsum() - chart_data["share"]
    chart_data["midpoint"] = chart_data["start"] + chart_data["share"] / 2
    chart_data["share_label"] = chart_data["share"].map(lambda value: f"{value:.0f}%")

    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y(
                "Stage or period:N",
                title=None,
                sort=display_periods,
                axis=alt.Axis(labelLimit=240),
            ),
            x=alt.X(
                "share:Q",
                title="Share of comments (%)",
                stack="zero",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "sentiment:N",
                title=None,
                scale=alt.Scale(
                    domain=list(SENTIMENT_COLORS),
                    range=list(SENTIMENT_COLORS.values()),
                ),
            ),
            order=alt.Order("sentiment_order:Q"),
            tooltip=[
                alt.Tooltip("Stage or period:N", title="Stage or period"),
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("share:Q", title="Share", format=".1f"),
            ],
        )
    )
    labels = (
        alt.Chart(chart_data[chart_data["share"].ge(8)])
        .mark_text(color="white", fontWeight="bold", fontSize=11)
        .encode(
            y=alt.Y("Stage or period:N", sort=display_periods),
            x=alt.X("midpoint:Q", scale=alt.Scale(domain=[0, 100])),
            text="share_label:N",
        )
    )
    st.altair_chart((bars + labels).properties(height=max(150, len(trends) * 52)), width="stretch")


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _label_column(frame: pd.DataFrame) -> str:
    if "corrected_label" in frame.columns:
        return "corrected_label"
    if "label" in frame.columns:
        return "label"
    return ""


def _sentiment_summary(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    columns = ["sentiment", "comments", "percent"]
    if frame.empty or not label_column:
        return pd.DataFrame(columns=columns)
    labels = frame[label_column].fillna("unknown").astype(str).str.lower()
    counts = labels.value_counts()
    order = [label for label in ("negative", "neutral", "positive", "unknown") if label in counts]
    return pd.DataFrame(
        {
            "sentiment": [label.title() for label in order],
            "comments": [int(counts[label]) for label in order],
            "percent": [round(float(counts[label] / len(frame) * 100), 1) for label in order],
        }
    )


def _emotion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    present = [column for column in EMOTION_COLUMNS if column in frame.columns]
    if not present:
        return pd.DataFrame(columns=["emotion", "average_score"])
    averages = (
        frame[present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean()
        .sort_values(ascending=False)
    )
    return pd.DataFrame(
        {
            "emotion": [column.removeprefix("nrc_").title() for column in averages.index],
            "average_score": [round(float(value), 3) for value in averages.values],
        }
    )


def _apply_sidebar_filters(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()

    if "is_relevant" in filtered.columns:
        relevant_only = st.sidebar.checkbox(
            "Relevant energy comments only",
            value=True,
            key="relevant_current_url",
        )
        if relevant_only:
            filtered = filtered[_boolean_series(filtered["is_relevant"])].copy()

    timestamp_column = find_timestamp_column(filtered)
    if timestamp_column:
        parsed = pd.to_datetime(filtered[timestamp_column], errors="coerce", utc=True)
        valid = parsed.dropna()
        if not valid.empty and valid.min().date() < valid.max().date():
            start_date = valid.min().date()
            end_date = valid.max().date()
            selected_range = st.sidebar.date_input(
                "Date range",
                value=(start_date, end_date),
                min_value=start_date,
                max_value=end_date,
                key="dates_current_url",
            )
            if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
                selected_start, selected_end = selected_range
                date_values = parsed.dt.date
                filtered = filtered[
                    date_values.between(selected_start, selected_end, inclusive="both")
                ].copy()
    return filtered.reset_index(drop=True)


def _metric_percent(summary: pd.DataFrame, sentiment: str) -> float:
    match = summary[summary["sentiment"].eq(sentiment)]
    return float(match.iloc[0]["percent"]) if not match.empty else 0.0


def _display_topic(topic: str) -> str:
    return TOPIC_DISPLAY_NAMES.get(topic, topic)


def _text_column(frame: pd.DataFrame) -> str | None:
    return next(
        (
            column
            for column in ("text_raw", "text", "text_clean", "processed_text")
            if column in frame.columns
        ),
        None,
    )


def _render_overview(
    frame: pd.DataFrame,
    sentiment: pd.DataFrame,
    emotions: pd.DataFrame,
    topics: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    top_topic = _display_topic(str(topics.iloc[0]["topic"])) if not topics.empty else "Unavailable"
    leading_emotion = str(emotions.iloc[0]["emotion"]) if not emotions.empty else "Unavailable"
    metric_columns = st.columns(4)
    metric_columns[0].metric("Comments analyzed", f"{len(frame):,}")
    metric_columns[1].metric("Negative sentiment", f"{_metric_percent(sentiment, 'Negative'):.1f}%")
    _text_metric(metric_columns[2], "Top topic", top_topic)
    _text_metric(metric_columns[3], "Leading emotion", leading_emotion)

    sentiment_column, emotion_column = st.columns(2)
    with sentiment_column:
        st.subheader("Sentiment Distribution")
        if sentiment.empty:
            st.info("No sentiment labels are available for this selection.")
        else:
            _compact_bar_chart(
                sentiment,
                category="sentiment",
                value="comments",
                height=165,
                bar_size=18,
                colors_by_category=SENTIMENT_COLORS,
                value_title="Comments",
            )

    with emotion_column:
        st.subheader("Emotion Index")
        if emotions.empty:
            st.info("No emotion scores are available for this selection.")
        else:
            _compact_bar_chart(
                emotions,
                category="emotion",
                value="average_score",
                height=165,
                bar_size=16,
                horizontal=True,
                colors_by_category=EMOTION_COLORS,
                value_title="Average score",
            )

    st.subheader("Priority Considerations")
    if recommendations.empty:
        st.info("More repeated evidence is needed before targeted considerations can be generated.")
    else:
        preview = recommendations[["priority", "topic", "evidence", "recommendation"]].copy()
        preview["topic"] = preview["topic"].map(_display_topic)
        st.dataframe(preview, width="stretch", hide_index=True)


def _render_recent_link_trends(trends: pd.DataFrame) -> None:
    if trends.empty:
        st.info(
            "No saved link analyses are available yet. Analyze at least two different "
            "Facebook links, then return here."
        )
        return
    if len(trends) < 2:
        st.info(
            "One saved link is available. Analyze another Facebook link before calculating "
            "a direction of change."
        )
    else:
        st.info(_sentiment_change_explanation(trends, temporal=True))
        st.subheader("How Sentiment Differed Across the Saved Links")
        st.caption(
            "Each bar represents one Facebook discussion. The colored sections show how the "
            "relevant comments in that link were divided between negative, neutral, and positive."
        )
        _sentiment_composition_chart(trends, temporal=True)
        st.subheader("Earlier Link Compared with Latest Link")
        st.dataframe(
            _sentiment_change_table(trends),
            width="stretch",
            hide_index=True,
        )
        changes = trends["negative_percent"].diff().abs()
        if changes.notna().any():
            change_index = int(changes.idxmax())
            previous = trends.iloc[change_index - 1]
            current = trends.iloc[change_index]
            signed_change = float(current["negative_percent"] - previous["negative_percent"])
            direction = "increased" if signed_change > 0 else "decreased"
            st.info(
                f"The largest change was between {previous['link_label']} and "
                f"{current['link_label']}: negative sentiment {direction} by "
                f"{abs(signed_change):.1f} percentage points. The latest link's leading "
                f"topic was {current['top_topic']}."
            )

    table = trends.copy()
    table["period"] = pd.to_datetime(table["period"], errors="coerce", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    st.dataframe(
        table.rename(
            columns={
                "period": "Facebook date",
                "link_number": "Link",
                "link_label": "Post title",
                "source_url": "Facebook URL",
                "comment_count": "Relevant comments",
                "negative_percent": "Negative %",
                "neutral_percent": "Neutral %",
                "positive_percent": "Positive %",
                "top_topic": "Leading topic",
                "date_source": "Date source",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Facebook URL": st.column_config.LinkColumn(width="large"),
            "Post title": st.column_config.TextColumn(width="large"),
        },
    )


def _render_trends(frame: pd.DataFrame, label_column: str) -> None:
    control_text, control_button = st.columns([3, 1])
    with control_text:
        st.subheader("Trend Across Recent Links")
        st.caption(
            "Compare the five newest saved Facebook discussions using dates from their Facebook "
            "metadata, never the date the app analyzed them. This does not consume Apify tokens."
        )
    with control_button:
        track_submitted = st.button(
            "Track trend",
            type="primary",
            use_container_width=True,
            key="track_current_url_trend",
        )

    if track_submitted:
        with st.spinner("Loading five saved links and their Facebook metadata dates..."):
            try:
                st.session_state["recent_link_trends"] = load_recent_link_trends(limit=5)
            except Exception as exc:
                _show_user_error(
                    exc,
                    operation="Load recent-link trend",
                    fallback="The saved-link trend could not be loaded. Please try again.",
                )
                return

    recent_link_trends = st.session_state.get("recent_link_trends")
    if isinstance(recent_link_trends, pd.DataFrame):
        _render_recent_link_trends(recent_link_trends)

    st.divider()
    st.subheader("Current Link Discussion Pattern")
    st.caption(
        "The charts below describe comment timing within the currently selected Facebook link."
    )

    automatic_frequency = resolve_trend_frequency(frame)
    automatic = build_sentiment_trends(
        frame,
        frequency="Automatic",
        label_column=label_column,
    )
    progression = build_comment_progression(frame, segments=3, label_column=label_column)
    has_calendar_trend = len(automatic) >= 2
    if has_calendar_trend:
        basis = st.segmented_control(
            "Trend basis",
            options=["Calendar time", "Comment progression"],
            default="Calendar time",
        )
    else:
        basis = "Comment progression"
        st.info(
            "Facebook supplied only one calendar period, so movement is shown across ordered "
            "comment groups instead. These groups show discussion progression, not elapsed time."
        )

    if basis == "Calendar time":
        frequency = st.segmented_control(
            "Time grouping",
            options=["Automatic", "Hour", "Day", "Week", "Month"],
            default="Automatic",
        )
        resolved_frequency = (
            automatic_frequency if frequency == "Automatic" else str(frequency)
        )
        trends = build_sentiment_trends(
            frame,
            frequency=str(frequency),
            label_column=label_column,
        )
        temporal = True
        if frequency == "Automatic":
            st.caption(f"Automatic grouping selected: {automatic_frequency}.")
    else:
        trends = progression
        temporal = False
        resolved_frequency = ""

    if trends.empty:
        st.info("There are not enough valid comments to calculate movement.")
        return

    st.info(_sentiment_change_explanation(trends, temporal=temporal))
    latest = trends.iloc[-1]
    latest_sentiments = {
        "Negative": float(latest.get("negative_percent", 0)),
        "Neutral": float(latest.get("neutral_percent", 0)),
        "Positive": float(latest.get("positive_percent", 0)),
    }
    latest_name, latest_share = max(latest_sentiments.items(), key=lambda item: item[1])
    negative_change = (
        float(latest.get("negative_percent", 0))
        - float(trends.iloc[0].get("negative_percent", 0))
        if len(trends) >= 2
        else None
    )
    summary_columns = st.columns(3)
    _text_metric(summary_columns[0], "Latest dominant sentiment", f"{latest_name} ({latest_share:.1f}%)")
    _text_metric(
        summary_columns[1],
        "Negative sentiment change",
        f"{negative_change:+.1f} points" if negative_change is not None else "One period only",
    )
    _text_metric(summary_columns[2], "Latest leading emotion", _leading_emotion(latest))

    st.subheader("How Sentiment Changed")
    st.caption(
        "Each bar totals 100%. Read from the beginning at the top to the latest comments at "
        "the bottom; a larger red section means a larger share of negative comments."
        if not temporal
        else "Each bar totals 100%. Read the periods in order; a larger red section means a "
        "larger share of negative comments in that period."
    )
    _sentiment_composition_chart(trends, temporal=temporal)

    if len(trends) >= 2:
        st.subheader("Beginning Compared with Latest")
        st.caption(
            "Change is measured in percentage points. For example, moving from 30% to 50% "
            "is an increase of 20 points."
        )
        st.dataframe(
            _sentiment_change_table(trends),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Readable Trend Table")
    st.caption(
        "This table keeps only the values needed to explain the trend. Technical NRC emotion "
        "columns are summarized as one leading emotion."
    )
    st.dataframe(
        _readable_trend_table(trends, temporal=temporal),
        width="stretch",
        hide_index=True,
        column_config={
            "Stage or period": st.column_config.TextColumn(width="large"),
            "Leading sentiment": st.column_config.TextColumn(width="medium"),
            "Leading emotion": st.column_config.TextColumn(width="medium"),
        },
    )
    if temporal:
        st.subheader("What the Comments Suggest Changed")
        events = build_trend_events(
            frame,
            trends,
            frequency=resolved_frequency,
            label_column=label_column,
        )
        if events.empty:
            st.info(
                "No strong period-to-period change was detected. More timestamped comments may "
                "be needed before discussion events can be identified."
            )
        else:
            strongest = events.iloc[0]
            st.info(
                f"The strongest detected change was: {strongest['movement']}. "
                f"The comments in that period suggest: {strongest['evidence']}"
            )
            event_table = events.rename(
                columns={
                    "period": "Period",
                    "movement": "How sentiment changed",
                    "evidence": "What commenters were discussing",
                    "representative_comment": "Example public comment",
                }
            )
            st.dataframe(
                event_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "What commenters were discussing": st.column_config.TextColumn(width="large"),
                    "Example public comment": st.column_config.TextColumn(width="large"),
                },
            )
            st.caption(
                "This explanation uses comment timing, leading topics, emotions, and example "
                "comments. It shows association, not proof that an external event caused the change."
            )
    st.caption(
        "Calendar trends depend on Facebook timestamps. Comment progression uses ordered groups "
        "when timestamp resolution is insufficient and must not be interpreted as elapsed time."
    )


def _render_topics(frame: pd.DataFrame, topics: pd.DataFrame, label_column: str) -> None:
    if topics.empty:
        st.info("No topic evidence is available for this selection.")
        return

    topic_chart = topics[["topic", "comment_count"]].copy()
    topic_chart["topic"] = topic_chart["topic"].map(_display_topic)
    _compact_bar_chart(
        topic_chart,
        category="topic",
        value="comment_count",
        height=215,
        bar_size=16,
        horizontal=True,
        value_title="Comments",
    )

    topic_table = topics[
        [
            "topic",
            "comment_count",
            "share_percent",
            "negative_percent",
            "positive_percent",
            "leading_emotion",
        ]
    ].copy()
    topic_table["topic"] = topic_table["topic"].map(_display_topic)
    topic_table = topic_table.rename(
        columns={
            "topic": "Topic",
            "comment_count": "Comments",
            "share_percent": "Share %",
            "negative_percent": "Negative %",
            "positive_percent": "Positive %",
            "leading_emotion": "Leading emotion",
        }
    )
    st.dataframe(topic_table, width="stretch", hide_index=True)

    selected_topic = st.selectbox(
        "Review comments from",
        options=topics["topic"].tolist(),
        format_func=_display_topic,
    )
    text_column = _text_column(frame)
    if text_column:
        topic_comments = frame[frame["topic"].eq(selected_topic)].copy()
        comment_audit = _comment_emotion_table(topic_comments, label_column)
        sample_columns = [
            column
            for column in (
                "Comment",
                "Topic",
                "Sentiment",
                "Dominant emotion",
                "Emotion score %",
                "Trust %",
                "Hope %",
                "Frustration %",
                "Matched emotion words",
            )
            if column in comment_audit.columns
        ]
        st.dataframe(
            comment_audit[sample_columns].head(30),
            width="stretch",
            hide_index=True,
            height=360,
            column_config={
                "Comment": st.column_config.TextColumn(width="large"),
                "Topic": st.column_config.TextColumn(width="medium"),
                "Matched emotion words": st.column_config.TextColumn(width="large"),
            },
        )
    st.caption(
        "Topics use a transparent Zambia-energy keyword taxonomy. Comments receive one primary "
        "topic so totals remain easy to audit."
    )


def _comment_emotion_table(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    text_column = _text_column(frame)
    if not text_column:
        return pd.DataFrame()
    table = pd.DataFrame({"Comment": frame[text_column].fillna("").astype(str)})
    if "topic" in frame.columns:
        table["Topic"] = frame["topic"].fillna("Other energy concerns").astype(str).map(_display_topic)
    table["Sentiment"] = (
        frame[label_column].fillna("unknown").astype(str).str.title()
        if label_column
        else "Unknown"
    )
    emotion_columns = [column for column in EMOTION_COLUMNS if column in frame.columns]
    if emotion_columns:
        numeric = frame[emotion_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        dominant_column = numeric.idxmax(axis=1)
        dominant_score = numeric.max(axis=1)
        table["Dominant emotion"] = dominant_column.str.removeprefix("nrc_").str.title()
        table.loc[dominant_score.le(0), "Dominant emotion"] = "No signal"
        table["Emotion score %"] = (dominant_score * 100).round(1)
        for column in emotion_columns:
            table[f"{column.removeprefix('nrc_').title()} %"] = (numeric[column] * 100).round(1)
    if "is_sarcastic" in frame.columns:
        table["Sarcasm"] = _boolean_series(frame["is_sarcastic"]).map({True: "Yes", False: "No"})
    evidence = frame.get("nrc_emotion_terms", pd.Series("", index=frame.index)).fillna("").astype(str)
    local = frame.get("local_emotion_terms", pd.Series("", index=frame.index)).fillna("").astype(str)
    table["Matched emotion words"] = evidence.where(local.eq(""), evidence + ", " + local)
    table["Matched emotion words"] = table["Matched emotion words"].str.strip(" ,")
    table["Positive emotion suppressed"] = (
        _boolean_series(frame["positive_emotion_suppressed"]).map({True: "Yes", False: "No"})
        if "positive_emotion_suppressed" in frame.columns
        else "No"
    )
    return table


def _render_comments(frame: pd.DataFrame, label_column: str) -> None:
    comment_table = _comment_emotion_table(frame, label_column)
    if comment_table.empty:
        st.info("No comment-level emotion evidence is available.")
        return

    filter_column, sentiment_column = st.columns([1.4, 0.6])
    with filter_column:
        query = st.text_input("Search comments", placeholder="Search comment text")
    with sentiment_column:
        sentiment_options = ["All", *sorted(comment_table["Sentiment"].unique())]
        selected_sentiment = st.selectbox("Sentiment", sentiment_options)

    visible = comment_table.copy()
    if query.strip():
        visible = visible[visible["Comment"].str.contains(query.strip(), case=False, na=False)]
    if selected_sentiment != "All":
        visible = visible[visible["Sentiment"].eq(selected_sentiment)]

    st.dataframe(
        visible,
        width="stretch",
        hide_index=True,
        height=390,
        column_config={
            "Comment": st.column_config.TextColumn(width="large"),
            "Matched emotion words": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "Download comment emotion audit",
        data=visible.to_csv(index=False).encode("utf-8"),
        file_name=f"comment_emotion_audit_{datetime.now(timezone.utc):%Y%m%d}.csv",
        mime="text/csv",
    )
    st.caption(
        "Emotion values are lexical evidence scores from 0 to 100, not probabilities. "
        "Matched words show why a score appeared; critical or sarcastic promise language "
        "suppresses misleading hope and trust signals."
    )


def _render_report(frame: pd.DataFrame, source_label: str) -> None:
    report = build_stakeholder_report(frame, source_label=source_label)
    pdf = build_executive_summary_pdf(frame, source_label=source_label)
    pdf_column, detail_column = st.columns(2)
    with pdf_column:
        st.download_button(
            "Download executive summary PDF",
            data=pdf,
            file_name=f"zambian_energy_executive_summary_{datetime.now(timezone.utc):%Y%m%d}.pdf",
            mime="application/pdf",
            type="primary",
        )
    with detail_column:
        st.download_button(
            "Download detailed report",
            data=report.encode("utf-8"),
            file_name=f"zambian_energy_sentiment_report_{datetime.now(timezone.utc):%Y%m%d}.md",
            mime="text/markdown",
        )
    st.markdown(report)


def _render_url_analysis() -> None:
    with st.form("facebook_url_analysis"):
        latest_summary = st.session_state.get("latest_live_summary")
        current_url = (
            str(latest_summary.get("source_url", ""))
            if isinstance(latest_summary, dict)
            else ""
        )
        facebook_url = st.text_input("Public Facebook post URL", value=current_url)
        max_comments = st.number_input(
            "Comments to collect",
            min_value=10,
            max_value=1000,
            value=300,
            step=50,
            help="Choose how many public top-level comments to collect, from 10 to 1,000.",
        )
        submitted = st.form_submit_button(
            "Analyze public comments",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not facebook_url.strip():
            st.warning("Enter a public Facebook URL.")
            return
        with st.spinner("Checking saved data, then collecting only if needed..."):
            try:
                live_results, save_summary = analyze_facebook_url(
                    facebook_url.strip(),
                    models_dir=MODELS_DIR,
                    max_comments=int(max_comments),
                    force_refresh=False,
                )
            except Exception as exc:
                _show_user_error(
                    exc,
                    operation="Facebook URL analysis",
                    fallback="The comments could not be analyzed. Please try again.",
                )
                return

        st.session_state["latest_live_results"] = live_results
        st.session_state["latest_live_summary"] = save_summary
        st.session_state.pop("recent_link_trends", None)
        if save_summary.get("cache_hit"):
            st.session_state["_analysis_flash"] = (
                f"Loaded {save_summary['fetched_rows']:,} saved comments and analyzed them locally; "
                "no Apify tokens were used."
            )
        else:
            st.session_state["_analysis_flash"] = (
                f"Collected and analyzed {save_summary['analyzed_rows']:,} comments; "
                f"{save_summary['relevant_rows']:,} were relevant to the energy discussion."
            )
        st.rerun()

    summary = st.session_state.get("latest_live_summary")
    if isinstance(summary, dict):
        post_title = str(summary.get("post_title", "")).strip()
        if post_title:
            st.subheader(post_title)
        else:
            st.caption("Post title unavailable from Facebook.")
        if summary.get("cache_hit"):
            st.success("Loaded from the local dataset cache. No Apify tokens were used.")
        elif summary.get("tracking_refresh"):
            st.caption("Collection source: Apify trend refresh.")
        else:
            st.caption("Collection source: Apify (new URL).")
        result_columns = st.columns(3)
        result_columns[0].metric("Fetched", f"{summary.get('fetched_rows', 0):,}")
        result_columns[1].metric("Analyzed", f"{summary.get('analyzed_rows', 0):,}")
        result_columns[2].metric("Energy-relevant", f"{summary.get('relevant_rows', 0):,}")
        if summary.get("tracking_refresh"):
            st.caption(
                f"Trend cutoff: {summary.get('trend_cutoff', 'current analysis time')} · "
                f"New comments found: {summary.get('new_comment_rows', 0):,}"
            )

    st.caption(
        "Saved comments are reused before Apify is called. Only publicly accessible comments are "
        "analyzed. After analyzing at least two different links, use Track trend in the Trends "
        "tab to compare up to five saved link analyses."
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(
        page_title="Zambian Green Energy Sentiment",
        layout="wide",
    )
    _inject_styles()
    st.title("Zambian Green Energy Sentiment")
    st.caption(
        "Public Facebook discourse on electricity reliability, affordability, renewable energy, "
        "and institutional trust."
    )

    latest = st.session_state.get("latest_live_results")
    raw_data = latest.copy() if isinstance(latest, pd.DataFrame) else pd.DataFrame()

    with st.sidebar:
        st.header("Current URL")
        if raw_data.empty:
            filtered_data = raw_data
            st.caption("Submit a public Facebook post URL to begin.")
        else:
            filtered_data = _apply_sidebar_filters(raw_data)
            st.divider()
            post_title = str(raw_data.iloc[0].get("post_title", "")).strip()
            if post_title:
                st.markdown(f"**{post_title}**")
            st.write(f"Comments in view: **{len(filtered_data):,}**")
            source_url = str(raw_data.iloc[0].get("source_url", "")).strip()
            if source_url:
                st.caption(source_url)
            if st.button("Clear current analysis"):
                st.session_state.pop("latest_live_results", None)
                st.session_state.pop("latest_live_summary", None)
                st.rerun()
        st.caption("Automated findings should be interpreted with supporting operational evidence.")

    flash = st.session_state.pop("_analysis_flash", None)
    if flash:
        st.success(str(flash))

    analyze_tab, overview_tab, trends_tab, topics_tab, comments_tab, report_tab = (
        _dashboard_tabs()
    )
    with analyze_tab:
        _render_url_analysis()

    result_tabs = (overview_tab, trends_tab, topics_tab, comments_tab, report_tab)
    if filtered_data.empty:
        for tab in result_tabs:
            with tab:
                st.info("Analyze a public Facebook post URL to view these results.")
        return

    try:
        label_column = _label_column(filtered_data)
        topic_data = add_topic_labels(filtered_data)
        sentiment = _sentiment_summary(topic_data, label_column)
        emotions = _emotion_summary(topic_data)
        topics = summarize_topics(topic_data, label_column=label_column)
        daily_trends = build_sentiment_trends(
            topic_data,
            frequency="Day",
            label_column=label_column,
        )
        recommendations = build_policy_recommendations(
            topics,
            trend_data=daily_trends,
            total_comments=len(topic_data),
        )
    except Exception as exc:  # pragma: no cover - defensive UI boundary
        _show_user_error(
            exc,
            operation="Prepare dashboard results",
            fallback="The results could not be prepared. Please analyze the URL again.",
        )
        return
    source_url = str(topic_data.iloc[0].get("source_url", "Current URL"))
    post_title = str(topic_data.iloc[0].get("post_title", "")).strip()
    report_source = f"{post_title} — {source_url}" if post_title else source_url

    with overview_tab:
        _render_safely(
            "overview",
            _render_overview,
            topic_data,
            sentiment,
            emotions,
            topics,
            recommendations,
        )
    with trends_tab:
        _render_safely("trends", _render_trends, topic_data, label_column)
    with topics_tab:
        _render_safely("topics", _render_topics, topic_data, topics, label_column)
    with comments_tab:
        _render_safely("comments", _render_comments, topic_data, label_column)
    with report_tab:
        _render_safely("report", _render_report, topic_data, report_source)


if __name__ == "__main__":
    main()
