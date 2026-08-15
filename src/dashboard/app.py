"""End-user dashboard for Zambian green-energy public sentiment insights."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.live_analysis import analyze_facebook_url
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
    describe_comment_progression,
    describe_negative_trend,
    find_timestamp_column,
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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.25rem; padding-bottom: 2rem;}
        h1 {font-size: 2rem !important; line-height: 1.2 !important;}
        h2 {font-size: 1.3rem !important; line-height: 1.3 !important;}
        h3 {font-size: 1.08rem !important; line-height: 1.35 !important;}
        [data-testid="stMetric"] {border-bottom: 2px solid #E6E9EE; padding-bottom: 0.65rem;}
        [data-testid="stMetricValue"] {font-size: 1.42rem; line-height: 1.2; white-space: normal; overflow: visible;}
        [data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
        [data-testid="stMetricLabel"] {font-size: 0.82rem; color: #5F6B7A;}
        [data-testid="stDataFrame"] {font-size: 0.86rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compact_bar_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    height: int = 210,
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
        .mark_bar(size=30, cornerRadius=3)
        .encode(**encoding, color=color, tooltip=tooltip)
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _compact_line_chart(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    temporal: bool,
    value_title: str,
    height: int = 230,
) -> None:
    if frame.empty or not value_columns:
        return
    chart_data = frame[["period", *value_columns]].melt(
        id_vars="period",
        var_name="series",
        value_name="value",
    )
    chart_data["series"] = chart_data["series"].map(labels).fillna(chart_data["series"])
    x_type = "T" if temporal else "N"
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=alt.OverlayMarkDef(size=48), strokeWidth=2.2)
        .encode(
            x=alt.X(f"period:{x_type}", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=value_title, axis=alt.Axis(grid=True, tickCount=5)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=list(colors), range=list(colors.values())),
            ),
            tooltip=[
                alt.Tooltip(f"period:{x_type}", title="Group" if not temporal else "Period"),
                alt.Tooltip("series:N", title="Measure"),
                alt.Tooltip("value:Q", title=value_title, format=".2f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


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
    metric_columns[2].metric("Top topic", top_topic)
    metric_columns[3].metric("Leading emotion", leading_emotion)

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
                height=210,
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


def _render_trends(frame: pd.DataFrame, label_column: str) -> None:
    daily = build_sentiment_trends(frame, frequency="Day", label_column=label_column)
    progression = build_comment_progression(frame, label_column=label_column)
    has_calendar_trend = len(daily) >= 2
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
            options=["Day", "Week", "Month"],
            default="Day",
        )
        trends = build_sentiment_trends(
            frame,
            frequency=str(frequency),
            label_column=label_column,
        )
        temporal = True
        st.caption(describe_negative_trend(trends))
        sentiment_heading = "Sentiment Over Time"
    else:
        trends = progression
        temporal = False
        st.caption(describe_comment_progression(trends))
        sentiment_heading = "Sentiment Across the Discussion"

    if trends.empty:
        st.info("There are not enough valid comments to calculate movement.")
        return

    st.subheader(sentiment_heading)
    _compact_line_chart(
        trends,
        value_columns=["negative_percent", "neutral_percent", "positive_percent"],
        labels={
            "negative_percent": "Negative",
            "neutral_percent": "Neutral",
            "positive_percent": "Positive",
        },
        colors=SENTIMENT_COLORS,
        temporal=temporal,
        value_title="Share of comments (%)",
    )

    volume_column, emotion_column = st.columns([0.8, 1.2])
    with volume_column:
        st.subheader("Comment Volume")
        _compact_bar_chart(
            trends,
            category="period",
            value="comment_count",
            height=190,
            value_title="Comments",
        )
    with emotion_column:
        st.subheader("Emotion Movement")
        emotion_columns = [
            column
            for column in ("nrc_frustration", "nrc_anger", "nrc_hope", "nrc_trust")
            if column in trends.columns
        ]
        _compact_line_chart(
            trends,
            value_columns=emotion_columns,
            labels={column: column.removeprefix("nrc_").title() for column in emotion_columns},
            colors={
                column.removeprefix("nrc_").title(): EMOTION_COLORS[
                    column.removeprefix("nrc_").title()
                ]
                for column in emotion_columns
            },
            temporal=temporal,
            value_title="Average score",
            height=190,
        )

    st.dataframe(
        trends.rename(
            columns={
                "period": "Period",
                "comment_count": "Comments",
                "negative_percent": "Negative %",
                "neutral_percent": "Neutral %",
                "positive_percent": "Positive %",
                "sarcasm_percent": "Sarcasm %",
            }
        ),
        width="stretch",
        hide_index=True,
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
        height=270,
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
        sample_columns = [text_column]
        if label_column:
            sample_columns.append(label_column)
        if "source_url" in topic_comments.columns:
            sample_columns.append("source_url")
        st.dataframe(
            topic_comments[sample_columns].head(30),
            width="stretch",
            hide_index=True,
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
        facebook_url = st.text_input("Public Facebook post URL")
        max_comments = st.number_input(
            "Maximum comments",
            min_value=10,
            max_value=300,
            value=300,
            step=10,
        )
        submitted = st.form_submit_button("Analyze public comments")

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
                )
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
                return

        st.session_state["latest_live_results"] = live_results
        st.session_state["latest_live_summary"] = save_summary
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
        if summary.get("cache_hit"):
            st.success("Loaded from the local dataset cache. No Apify tokens were used.")
        else:
            st.caption("Collection source: Apify (new URL).")
        result_columns = st.columns(3)
        result_columns[0].metric("Fetched", f"{summary.get('fetched_rows', 0):,}")
        result_columns[1].metric("Analyzed", f"{summary.get('analyzed_rows', 0):,}")
        result_columns[2].metric("Energy-relevant", f"{summary.get('relevant_rows', 0):,}")

    st.caption(
        "Saved comments are reused before Apify is called. Only publicly accessible comments are "
        "analyzed, and results are aggregated for research and decision support."
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

    analyze_tab, overview_tab, trends_tab, topics_tab, comments_tab, report_tab = st.tabs(
        ["Analyze URL", "Overview", "Trends", "Topics", "Comments", "Report"]
    )
    with analyze_tab:
        _render_url_analysis()

    result_tabs = (overview_tab, trends_tab, topics_tab, comments_tab, report_tab)
    if filtered_data.empty:
        for tab in result_tabs:
            with tab:
                st.info("Analyze a public Facebook post URL to view these results.")
        return

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
    source_url = str(topic_data.iloc[0].get("source_url", "Current URL"))

    with overview_tab:
        _render_overview(topic_data, sentiment, emotions, topics, recommendations)
    with trends_tab:
        _render_trends(topic_data, label_column)
    with topics_tab:
        _render_topics(topic_data, topics, label_column)
    with comments_tab:
        _render_comments(topic_data, label_column)
    with report_tab:
        _render_report(topic_data, source_url)


if __name__ == "__main__":
    main()
