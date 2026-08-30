"""PDF executive-summary generation for a single analyzed Facebook URL."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.insights.policy_report import build_policy_recommendations
from src.insights.topic_analyzer import EMOTION_COLUMNS, add_topic_labels, summarize_topics
from src.temporal.event_tracker import (
    build_comment_progression,
    build_sentiment_trends,
    describe_comment_progression,
    describe_negative_trend,
)


INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5F6B7A")
ACCENT = colors.HexColor("#D94B4B")
PALE = colors.HexColor("#F3F5F7")
GRID = colors.HexColor("#D9DEE5")


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "Automated decision-support summary")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _date_range(frame: pd.DataFrame) -> str:
    for column in ("timestamp", "analysis_timestamp", "created_at", "date"):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if values.empty:
            continue
        start = values.min().date().isoformat()
        end = values.max().date().isoformat()
        return start if start == end else f"{start} to {end}"
    return "Date unavailable"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=INK,
            spaceAfter=4 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=INK,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=INK,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Kpi",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=INK,
            alignment=TA_CENTER,
        )
    )
    return styles


def _styled_table(data, widths, *, header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def build_executive_summary_pdf(
    frame: pd.DataFrame,
    *,
    source_label: str,
    generated_at: datetime | None = None,
) -> bytes:
    """Build a concise PDF containing headline findings and recommendations."""
    generated = generated_at or datetime.now(timezone.utc)
    labeled = frame if "topic" in frame.columns else add_topic_labels(frame)
    label_column = "corrected_label" if "corrected_label" in labeled.columns else "label"
    labels = labeled.get(label_column, pd.Series(dtype=str)).fillna("unknown").astype(str).str.lower()
    counts = labels.value_counts()
    dominant = str(counts.index[0]).title() if not counts.empty else "Unavailable"
    negative_percent = float(labels.eq("negative").mean() * 100) if len(labels) else 0.0

    topics = summarize_topics(labeled, label_column=label_column)
    top_topic = str(topics.iloc[0]["topic"]) if not topics.empty else "Unavailable"
    present_emotions = [column for column in EMOTION_COLUMNS if column in labeled.columns]
    emotion_means = (
        labeled[present_emotions].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
        if present_emotions
        else pd.Series(dtype=float)
    )
    leading_emotion = (
        str(emotion_means.idxmax()).removeprefix("nrc_").title()
        if not emotion_means.empty and float(emotion_means.max()) > 0
        else "No strong signal"
    )

    calendar_trends = build_sentiment_trends(
        labeled,
        frequency="Automatic",
        label_column=label_column,
    )
    progression = build_comment_progression(labeled, label_column=label_column)
    if len(calendar_trends) >= 2:
        movement = describe_negative_trend(calendar_trends)
        recommendation_trends = calendar_trends
    else:
        movement = describe_comment_progression(progression)
        recommendation_trends = None
    recommendations = build_policy_recommendations(
        topics,
        trend_data=recommendation_trends,
        total_comments=len(labeled),
        limit=3,
    )

    styles = _styles()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="Zambian Green Energy Sentiment Executive Summary",
        author="Zambian Green Energy Sentiment Prototype",
    )

    story = [
        Paragraph("Zambian Green Energy Sentiment", styles["ReportTitle"]),
        Paragraph("Executive summary for one public Facebook discussion", styles["Meta"]),
        Spacer(1, 2 * mm),
        Paragraph(
            f"<b>Generated:</b> {generated.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC}<br/>"
            f"<b>Coverage:</b> {_date_range(labeled)}<br/>"
            f"<b>Source:</b> {escape(source_label)}",
            styles["Meta"],
        ),
        Spacer(1, 5 * mm),
    ]

    kpis = [
        ["Comments", "Dominant sentiment", "Negative share", "Leading emotion"],
        [
            Paragraph(f"{len(labeled):,}", styles["Kpi"]),
            Paragraph(dominant, styles["Kpi"]),
            Paragraph(f"{negative_percent:.1f}%", styles["Kpi"]),
            Paragraph(leading_emotion, styles["Kpi"]),
        ],
    ]
    story.extend(
        [
            _styled_table(kpis, [38 * mm, 45 * mm, 38 * mm, 45 * mm]),
            Paragraph("Key finding", styles["Section"]),
            Paragraph(
                f"The most discussed issue was <b>{escape(top_topic)}</b>. {escape(movement)}",
                styles["BodySmall"],
            ),
            Paragraph("Sentiment and emotion", styles["Section"]),
        ]
    )

    sentiment_rows = [["Sentiment", "Comments", "Share"]]
    for sentiment in ("negative", "neutral", "positive"):
        count = int(counts.get(sentiment, 0))
        share = count / len(labeled) * 100 if len(labeled) else 0.0
        sentiment_rows.append([sentiment.title(), f"{count:,}", f"{share:.1f}%"])
    emotion_rows = [["Emotion", "Average score"]]
    for column, value in emotion_means.sort_values(ascending=False).items():
        emotion_rows.append([column.removeprefix("nrc_").title(), f"{float(value):.3f}"])
    story.append(
        Table(
            [
                [
                    _styled_table(sentiment_rows, [34 * mm, 25 * mm, 25 * mm]),
                    _styled_table(emotion_rows, [43 * mm, 35 * mm]),
                ]
            ],
            colWidths=[88 * mm, 82 * mm],
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
        )
    )

    story.append(Paragraph("Priority actions", styles["Section"]))
    if recommendations.empty:
        story.append(
            Paragraph(
                "The discussion does not yet contain enough repeated evidence for a targeted action.",
                styles["BodySmall"],
            )
        )
    else:
        for index, row in recommendations.iterrows():
            story.append(
                KeepTogether(
                    [
                        Paragraph(
                            f"<b>{index + 1}. {escape(str(row['topic']))} "
                            f"({escape(str(row['priority']))})</b>",
                            styles["BodySmall"],
                        ),
                        Paragraph(escape(str(row["evidence"])), styles["Meta"]),
                        Paragraph(escape(str(row["recommendation"])), styles["BodySmall"]),
                        Spacer(1, 2 * mm),
                    ]
                )
            )

    story.extend(
        [
            Paragraph("Interpretation note", styles["Section"]),
            Paragraph(
                "These comments are unsolicited public reactions, not a representative survey. "
                "Sentiment and emotion scores are automated lexical estimates and should be "
                "checked against the comment-level evidence before policy decisions are made.",
                styles["Meta"],
            ),
        ]
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
