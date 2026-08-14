"""Build the project-defence results guide from the latest evaluated outputs."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "docs" / "defence_results_guide.docx"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "labeled" / "final_label_relevant.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "results" / "evaluation_summary.csv"
TABLE_HELPER_PATH = (
    Path.home()
    / ".codex"
    / "plugins"
    / "cache"
    / "openai-primary-runtime"
    / "documents"
    / "26.805.11740"
    / "skills"
    / "documents"
    / "scripts"
    / "table_geometry.py"
)

NAVY = "203748"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
GOLD = "9A7218"
INK = "202124"
MUTED = "5F6368"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
LIGHT_GOLD = "FFF7E1"
WHITE = "FFFFFF"
RED = "#B74747"
GRAY = "#7B8088"
GREEN = "#2F7D5A"
FONT_REGULAR = Path("C:/Windows/Fonts/calibri.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/calibrib.ttf")


def _load_table_helper():
    spec = importlib.util.spec_from_file_location("docx_table_geometry", TABLE_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load table helper: {TABLE_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TABLE_HELPER = _load_table_helper()


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#"))


def _set_run_font(
    run,
    *,
    name: str = "Calibri",
    size: float | None = None,
    color: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
) -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = _rgb(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def _set_cell_fill(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), color)


def _set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _set_table_borders(table, color: str = "D6D9DE", size: str = "6") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def _add_page_field(paragraph, field_name: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = field_name
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, text, end):
        run._r.append(element)


def _configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = _rgb(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    heading_tokens = {
        "Heading 1": (16, BLUE, 18, 10),
        "Heading 2": (13, BLUE, 14, 7),
        "Heading 3": (12, DARK_BLUE, 10, 5),
    }
    for style_name, (size, color, before, after) in heading_tokens.items():
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = _rgb(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Number"):
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25

    caption = doc.styles["Caption"]
    caption.font.name = "Calibri"
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = _rgb(MUTED)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.keep_with_next = False

    if "Cover Kicker" not in [style.name for style in doc.styles]:
        kicker = doc.styles.add_style("Cover Kicker", WD_STYLE_TYPE.PARAGRAPH)
        kicker.font.name = "Calibri"
        kicker.font.size = Pt(10)
        kicker.font.bold = True
        kicker.font.color.rgb = _rgb(GOLD)
        kicker.paragraph_format.space_after = Pt(16)

    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    doc.settings.element.append(update_fields)

    for sec in doc.sections:
        _set_header_footer(sec)


def _set_header_footer(section) -> None:
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("PROJECT DEFENCE  |  RESULTS GUIDE")
    _set_run_font(run, size=8.5, color=MUTED, bold=True)

    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run("Mulenga Musamba  |  ")
    _set_run_font(run, size=8.5, color=MUTED)
    _add_page_field(p, "PAGE")
    run = p.add_run(" of ")
    _set_run_font(run, size=8.5, color=MUTED)
    _add_page_field(p, "NUMPAGES")


def _add_paragraph(
    doc: Document,
    text: str = "",
    *,
    size: float | None = None,
    color: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    align=None,
    before: float | None = None,
    after: float | None = None,
) -> object:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    if before is not None:
        p.paragraph_format.space_before = Pt(before)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    run = p.add_run(text)
    _set_run_font(run, size=size, color=color, bold=bold, italic=italic)
    return p


def _add_bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def _add_numbered(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Number")


def _add_callout(doc: Document, label: str, text: str, *, fill: str = LIGHT_BLUE) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(9)
    p.paragraph_format.left_indent = Inches(0.16)
    p.paragraph_format.right_indent = Inches(0.16)
    p.paragraph_format.keep_together = True
    p_pr = p._p.get_or_add_pPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    p_pr.append(shading)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "24")
    left.set(qn("w:color"), GOLD if fill == LIGHT_GOLD else BLUE)
    left.set(qn("w:space"), "8")
    borders.append(left)
    p_pr.append(borders)
    label_run = p.add_run(f"{label}: ")
    _set_run_font(label_run, bold=True, color=NAVY)
    body_run = p.add_run(text)
    _set_run_font(body_run, color=INK)


def _add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    widths_dxa: list[int],
    *,
    header_fill: str = LIGHT_BLUE,
    alignments: list[int] | None = None,
) -> object:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.allow_autofit = False
    header_row = table.rows[0]
    _set_repeat_table_header(header_row)
    for index, header in enumerate(headers):
        cell = header_row.cells[index]
        cell.text = header
        _set_cell_fill(cell, header_fill)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.0
            if alignments:
                paragraph.alignment = alignments[index]
            for run in paragraph.runs:
                _set_run_font(run, size=9.5, color=NAVY, bold=True)

    for row_index, values in enumerate(rows, start=1):
        cells = table.add_row().cells
        for column_index, value in enumerate(values):
            cell = cells[column_index]
            cell.text = str(value)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index % 2 == 0:
                _set_cell_fill(cell, "FAFBFC")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                if alignments:
                    paragraph.alignment = alignments[column_index]
                for run in paragraph.runs:
                    _set_run_font(run, size=9.25, color=INK)

    TABLE_HELPER.apply_table_geometry(
        table,
        widths_dxa,
        table_width_dxa=9360,
        indent_dxa=120,
        cell_margins_dxa={"top": 90, "bottom": 90, "start": 120, "end": 120},
    )
    _set_table_borders(table)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def _add_picture(doc: Document, image: io.BytesIO, width: float, alt_text: str, caption: str) -> None:
    image.seek(0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    inline_shape = run.add_picture(image, width=Inches(width))
    inline_shape._inline.docPr.set("descr", alt_text)
    cap = doc.add_paragraph(caption, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _font(size: int, *, bold: bool = False):
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def _image_buffer(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _draw_horizontal_grid(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    maximum: float,
    steps: int = 5,
) -> None:
    label_font = _font(24)
    for step in range(steps + 1):
        value = maximum * step / steps
        y = bottom - int((bottom - top) * step / steps)
        draw.line((left, y, right, y), fill="#DDE1E6", width=2)
        draw.text((left - 18, y), f"{value:.0f}", font=label_font, fill="#5F6368", anchor="rm")


def _chart_sentiment(counts: pd.Series) -> io.BytesIO:
    order = ["negative", "neutral", "positive"]
    values = [int(counts.get(label, 0)) for label in order]
    total = sum(values)
    image = Image.new("RGB", (1400, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), "Corrected sentiment distribution", font=_font(42, bold=True), fill="#203748")
    left, top, right, bottom = 145, 140, 1330, 590
    maximum = 1200
    _draw_horizontal_grid(draw, left=left, top=top, right=right, bottom=bottom, maximum=maximum)
    bar_width = 210
    centers = [350, 730, 1110]
    colors = [RED, GRAY, GREEN]
    labels = ["Negative", "Neutral", "Positive"]
    for center, value, color, label in zip(centers, values, colors, labels):
        bar_height = int((bottom - top) * value / maximum)
        x0, x1 = center - bar_width // 2, center + bar_width // 2
        y0 = bottom - bar_height
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=color)
        percent = value / total * 100 if total else 0
        draw.text((center, y0 - 25), f"{value:,}  |  {percent:.1f}%", font=_font(27, bold=True), fill="#202124", anchor="ms")
        draw.text((center, bottom + 32), label, font=_font(29, bold=True), fill="#202124", anchor="ma")
    draw.text((32, 365), "Comments", font=_font(26), fill="#5F6368", anchor="mm")
    return _image_buffer(image)


def _chart_topics(topics: pd.DataFrame) -> io.BytesIO:
    selected = topics.head(6).sort_values("comment_count")
    labels = selected["topic"].str.replace(" and ", " & ", regex=False).tolist()
    colors = ["#4472C4", "#5B9BD5", "#70AD47", "#ED7D31", "#A5A5A5", "#FFC000"]
    values = selected["comment_count"].astype(int).tolist()
    image = Image.new("RGB", (1500, 850), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 40), "Most discussed energy topics", font=_font(42, bold=True), fill="#203748")
    left, top, right, bottom = 520, 135, 1410, 755
    maximum = max(values) * 1.12
    for step in range(5):
        value = maximum * step / 4
        x = left + int((right - left) * step / 4)
        draw.line((x, top, x, bottom), fill="#DDE1E6", width=2)
        draw.text((x, bottom + 26), f"{value:.0f}", font=_font(22), fill="#5F6368", anchor="ma")
    row_height = (bottom - top) / len(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        center_y = top + row_height * (index + 0.5)
        bar_height = int(row_height * 0.58)
        bar_width = int((right - left) * value / maximum)
        draw.text((left - 22, center_y), label, font=_font(24, bold=True), fill="#202124", anchor="rm")
        draw.rounded_rectangle(
            (left, center_y - bar_height / 2, left + bar_width, center_y + bar_height / 2),
            radius=7,
            fill=color,
        )
        draw.text((left + bar_width + 16, center_y), f"{value:,}", font=_font(23, bold=True), fill="#202124", anchor="lm")
    draw.text(((left + right) / 2, 820), "Comments", font=_font(24), fill="#5F6368", anchor="mm")
    return _image_buffer(image)


def _chart_models(metrics: pd.DataFrame) -> io.BytesIO:
    labels = metrics["model"].replace(
        {
            "naive_bayes": "Naive Bayes",
            "logistic_regression": "Logistic Regression",
            "svm": "SVM",
        }
    ).tolist()
    accuracy = (metrics["accuracy"] * 100).tolist()
    f1 = (metrics["f1_weighted"] * 100).tolist()
    image = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "Held-out baseline model performance", font=_font(42, bold=True), fill="#203748")
    left, top, right, bottom = 150, 175, 1330, 625
    maximum = 70
    _draw_horizontal_grid(draw, left=left, top=top, right=right, bottom=bottom, maximum=maximum)
    centers = [350, 740, 1130]
    width = 112
    for center, label, accuracy_value, f1_value in zip(centers, labels, accuracy, f1):
        for x_center, value, color in (
            (center - width / 2, accuracy_value, "#4472C4"),
            (center + width / 2, f1_value, "#70AD47"),
        ):
            bar_height = int((bottom - top) * value / maximum)
            x0, x1 = x_center - width / 2 + 4, x_center + width / 2 - 4
            y0 = bottom - bar_height
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=color)
            draw.text((x_center, y0 - 18), f"{value:.1f}", font=_font(23, bold=True), fill="#202124", anchor="ms")
        draw.text((center, bottom + 30), label, font=_font(25, bold=True), fill="#202124", anchor="ma")
    draw.rounded_rectangle((160, 115, 190, 145), radius=3, fill="#4472C4")
    draw.text((205, 130), "Accuracy", font=_font(24), fill="#202124", anchor="lm")
    draw.rounded_rectangle((360, 115, 390, 145), radius=3, fill="#70AD47")
    draw.text((405, 130), "Weighted F1", font=_font(24), fill="#202124", anchor="lm")
    draw.text((34, 400), "Score (%)", font=_font(25), fill="#5F6368", anchor="mm")
    return _image_buffer(image)


def _topic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    from src.insights.topic_analyzer import add_topic_labels, summarize_topics

    return summarize_topics(add_topic_labels(frame), label_column="corrected_label")


def _add_cover(doc: Document) -> None:
    _add_paragraph(
        doc,
        "THE COPPERBELT UNIVERSITY",
        size=11,
        color=MUTED,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        before=62,
        after=5,
    )
    _add_paragraph(
        doc,
        "School of Information and Communication Technology | Computer Science Department",
        size=9.5,
        color=MUTED,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=66,
    )
    _add_paragraph(
        doc,
        "PROJECT DEFENCE RESULTS GUIDE",
        size=10,
        color=GOLD,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=16,
    )
    _add_paragraph(
        doc,
        "A Sentiment Analysis Prototype for Zambian Green Energy Discourse on Social Media",
        size=28,
        color=NAVY,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=16,
    )
    _add_paragraph(
        doc,
        "Corrected results, dashboard demonstration, and defence talking points",
        size=14,
        color=DARK_BLUE,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=54,
    )
    _add_paragraph(
        doc,
        "Mulenga Musamba  |  Student ID: 22110887",
        size=11,
        color=INK,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=5,
    )
    _add_paragraph(
        doc,
        "Supervisor: Dr. George Mufungulwa",
        size=11,
        color=INK,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=5,
    )
    _add_paragraph(
        doc,
        "April 2026",
        size=11,
        color=MUTED,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=34,
    )
    _add_callout(
        doc,
        "Core message",
        "The prototype turns one public Facebook URL into auditable sentiment, emotion, topic, trend, and report views for Zambian energy discourse.",
        fill=LIGHT_GOLD,
    )
    doc.add_page_break()


def _add_at_a_glance(doc: Document, frame: pd.DataFrame, metrics: pd.DataFrame) -> None:
    doc.add_heading("Defence At A Glance", level=1)
    doc.add_heading("A 60-second explanation", level=2)
    _add_paragraph(
        doc,
        "This project addresses the lack of timely, data-driven evidence about how Zambians discuss "
        "electricity reliability, affordability, renewable energy, and institutional trust online. "
        "A user supplies a public Facebook post URL. The system collects public comments, cleans "
        "them, filters energy-relevant content, detects sentiment, sarcasm, and emotions, groups "
        "comments into explainable topics, and presents the findings in a Streamlit dashboard. "
        "The purpose is decision support for communication and stakeholder engagement, not automatic policy making.",
    )

    counts = frame["corrected_label"].value_counts()
    best = metrics.sort_values("f1_weighted", ascending=False).iloc[0]
    snapshot_rows = [
        ["Labeled corpus", "4,135 comments scored after preprocessing"],
        ["Relevant analysis set", f"{len(frame):,} comments (68.0% of the labeled corpus)"],
        ["Largest sentiment class", f"Negative: {int(counts.get('negative', 0)):,} comments (35.9%)"],
        ["Held-out evaluation", "563 comments (20%); training used 2,250 comments (80%)"],
        [
            "Strongest baseline",
            f"{str(best['model']).upper()}: {best['accuracy'] * 100:.1f}% accuracy; "
            f"{best['f1_weighted'] * 100:.1f}% weighted F1",
        ],
        ["End-user interface", "URL-only dashboard; research and saved datasets are not exposed"],
    ]
    _add_table(
        doc,
        ["Evidence", "Current result"],
        snapshot_rows,
        [2700, 6660],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )
    _add_callout(
        doc,
        "Defence position",
        "This is a transparent prototype. Its contribution is the Zambia-aware workflow, auditable rules, local-language adaptation, and usable evidence views. Production deployment requires a human-labeled gold-standard dataset and further model validation.",
    )

    doc.add_heading("Research objectives answered", level=2)
    objectives = [
        "Collect and preprocess publicly available Facebook comments related to Zambian energy discourse.",
        "Apply VADER-based positive, neutral, and negative labeling with sarcasm and local-lexicon corrections.",
        "Train and evaluate Naive Bayes, Logistic Regression, and SVM classifiers using TF-IDF text features.",
        "Identify sentiment distributions, dominant topics, emotions, and time-based patterns.",
        "Translate evidence into cautious policy-communication and stakeholder-engagement considerations.",
    ]
    for item in objectives:
        _add_bullet(doc, item)
    doc.add_page_break()


def _add_system_flow(doc: Document) -> None:
    doc.add_heading("How The Prototype Works", level=1)
    _add_paragraph(
        doc,
        "The design separates the end-user journey from developer and research operations. End users "
        "see only results from the URL they submit. Saved corpora, model diagnostics, manual review, "
        "and future annotation controls remain developer-facing.",
    )
    steps = [
        "Input: the user enters one publicly accessible Facebook post URL and a comment limit.",
        "Collection: the Apify integration retrieves publicly accessible comments and saves a raw audit copy.",
        "Preprocessing: text is normalized; a machine-learning version is produced without replacing the readable original.",
        "Relevance: Zambia-energy terms and context rules separate useful discussion from off-topic comments.",
        "Language analysis: readable text is scored for VADER sentiment, sarcasm, and NRC/local emotion signals.",
        "Classification: TF-IDF features feed Naive Bayes, Logistic Regression, and SVM predictions.",
        "Interpretation: topic, trend, and policy-report modules aggregate evidence for the dashboard.",
        "Output: Analyze URL, Overview, Trends, Topics, and Report all describe the current URL only.",
    ]
    for step in steps:
        _add_numbered(doc, step)

    doc.add_heading("Why two text representations are necessary", level=2)
    _add_table(
        doc,
        ["Representation", "Purpose", "Reason"],
        [
            [
                "Readable text",
                "VADER, sarcasm, emotion, relevance",
                "Preserves punctuation, contractions, negation, and human-auditable meaning.",
            ],
            [
                "Processed text",
                "TF-IDF machine-learning features",
                "Reduces sparse surface variation for the three classical classifiers.",
            ],
        ],
        [2100, 2780, 4480],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] * 3,
    )
    _add_callout(
        doc,
        "Key design correction",
        "Rule-based sentiment must read the original sentence. Stopword-processed text is useful for TF-IDF models but can destroy expressions such as \"don't\" and \"didn't\".",
        fill=LIGHT_GOLD,
    )
    doc.add_page_break()


def _add_results(doc: Document, frame: pd.DataFrame, topics: pd.DataFrame) -> None:
    doc.add_heading("Corrected Sentiment Results", level=1)
    counts = frame["corrected_label"].value_counts()
    _add_picture(
        doc,
        _chart_sentiment(counts),
        6.2,
        "Bar chart showing 1,011 negative, 904 neutral, and 898 positive comments.",
        "Figure 1. Corrected sentiment distribution for 2,813 relevant comments.",
    )
    _add_callout(
        doc,
        "Interpretation",
        "The distribution is mixed rather than overwhelmingly negative. Negative sentiment is the largest class at 35.9%, but neutral and positive comments together account for 64.0%. The defensible conclusion is that energy discourse is contested and issue-specific.",
    )

    doc.add_heading("Topic findings", level=2)
    _add_picture(
        doc,
        _chart_topics(topics),
        6.2,
        "Horizontal bar chart showing load shedding and reliability as the dominant topic.",
        "Figure 2. Six most discussed topics after transparent keyword-based grouping.",
    )

    display_topics = topics.head(6)
    rows = []
    for _, row in display_topics.iterrows():
        rows.append(
            [
                str(row["topic"]),
                f"{int(row['comment_count']):,}",
                f"{float(row['share_percent']):.1f}%",
                f"{float(row['negative_percent']):.1f}%",
                str(row["leading_emotion"]),
            ]
        )
    _add_table(
        doc,
        ["Topic", "Comments", "Share", "Negative", "Leading emotion"],
        rows,
        [3800, 1100, 1100, 1300, 2060],
        alignments=[
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
        ],
    )
    _add_paragraph(
        doc,
        "Load shedding and reliability accounts for 63.6% of relevant comments, so service continuity "
        "dominates the public conversation. Renewable energy and solar has a smaller share (2.4%) but "
        "a comparatively positive profile (55.9% positive). Small topics must be interpreted cautiously "
        "because percentages can be unstable when comment counts are low.",
    )
    doc.add_page_break()


def _add_model_results(doc: Document, metrics: pd.DataFrame) -> None:
    doc.add_heading("Model Evaluation", level=1)
    _add_paragraph(
        doc,
        "The relevant dataset was split using stratified sampling: 2,250 comments for training and "
        "563 comments for held-out testing. The same corrected label distribution was maintained in both sets.",
    )
    _add_picture(
        doc,
        _chart_models(metrics),
        6.2,
        "Grouped bar chart comparing accuracy and weighted F1 for Naive Bayes, Logistic Regression, and SVM.",
        "Figure 3. Held-out performance of the three TF-IDF baseline classifiers.",
    )
    rows = []
    name_map = {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
    }
    for _, row in metrics.iterrows():
        rows.append(
            [
                name_map.get(str(row["model"]), str(row["model"])),
                f"{row['accuracy'] * 100:.1f}%",
                f"{row['precision_weighted'] * 100:.1f}%",
                f"{row['recall_weighted'] * 100:.1f}%",
                f"{row['f1_weighted'] * 100:.1f}%",
            ]
        )
    _add_table(
        doc,
        ["Model", "Accuracy", "Precision", "Recall", "Weighted F1"],
        rows,
        [2800, 1640, 1640, 1640, 1640],
        alignments=[
            WD_ALIGN_PARAGRAPH.LEFT,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
            WD_ALIGN_PARAGRAPH.CENTER,
        ],
    )
    _add_callout(
        doc,
        "Result",
        "SVM is the strongest current baseline, but only narrowly ahead of Logistic Regression. The scores are moderate because the task uses noisy, code-switched social-media text and weak labels rather than a fully human-annotated gold standard.",
    )
    doc.add_heading("How to defend the score", level=2)
    for item in [
        "Do not claim production-grade accuracy. Call the models transparent baselines within a prototype.",
        "Explain that VADER provides reproducible weak supervision, while human annotation is the next validation step.",
        "Use weighted F1 alongside accuracy because all three sentiment classes matter.",
        "Emphasize the complete, auditable pipeline and local adaptation as the main contribution.",
    ]:
        _add_bullet(doc, item)
    doc.add_page_break()


def _add_quality_case(doc: Document) -> None:
    doc.add_heading("Quality Correction Case Study", level=1)
    _add_paragraph(
        doc,
        "A manual review found that the following complaint had previously appeared as positive:",
    )
    _add_callout(
        doc,
        "Reviewed comment",
        "\"In case your calendars are not working properly today is first October 2024, we don't want to hear stories about the stabilization of power Ba Zesco Ltd because yesterday we didn't have power.\"",
        fill=LIGHT_GOLD,
    )
    _add_table(
        doc,
        ["Stage", "Text signal", "Result"],
        [
            [
                "Before correction",
                "Processed text changed contractions to \"don t\" and \"didn t\"; VADER mainly retained \"properly\".",
                "Compound +0.0772; positive",
            ],
            [
                "After correction",
                "Readable sentence preserves \"don't want\" and \"didn't have power\".",
                "Compound -0.0572; negative",
            ],
        ],
        [2200, 4800, 2360],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] * 3,
    )
    doc.add_heading("Root cause and fix", level=2)
    for item in [
        "Root cause: rule-based VADER scoring received stopword-removed, lemmatized text.",
        "Why it failed: cleaning split contractions and removed the grammatical context required for negation.",
        "Fix: VADER, sarcasm, emotion, and relevance now prefer text_raw/text/text_clean; processed_text remains available only for TF-IDF models.",
        "Verification: regression tests cover the exact negated outage pattern and the live URL pipeline.",
        "Data repair: all 4,135 stored comments, saved URL outputs, the train/test split, and model artifacts were regenerated.",
    ]:
        _add_bullet(doc, item)
    _add_callout(
        doc,
        "Defence framing",
        "This is evidence of validation, not a hidden failure. Manual error analysis exposed a preprocessing assumption, the cause was isolated, the rule was corrected, tests were added, and every dependent result was rebuilt.",
    )
    doc.add_page_break()


def _add_dashboard_demo(doc: Document) -> None:
    doc.add_heading("Dashboard Demonstration", level=1)
    _add_paragraph(
        doc,
        "The dashboard is deliberately focused on end users. It no longer displays the research dataset "
        "or cumulative saved analyses. Every result shown comes from the public URL analyzed in the current session.",
    )
    _add_table(
        doc,
        ["Tab", "What to show", "What to say"],
        [
            ["Analyze URL", "URL field, comment limit, analyze button", "This is the only entry point and it appears first."],
            ["Overview", "Key metrics, sentiment, emotions", "This gives a fast summary of the current discussion."],
            ["Trends", "Time grouping and movement", "Trends require valid timestamps and more than one period."],
            ["Topics", "Topic table and example comments", "The taxonomy is transparent and comments remain auditable."],
            ["Report", "On-screen and downloadable report", "Recommendations are linked to evidence and include limitations."],
        ],
        [1500, 3160, 4700],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] * 3,
    )
    doc.add_heading("A 90-second live demo", level=2)
    for step in [
        "Open the dashboard at http://localhost:8501.",
        "Paste a publicly accessible Facebook post URL and keep a modest comment limit for the demonstration.",
        "Select Analyze public comments and wait for the fetched, analyzed, and energy-relevant totals.",
        "Open Overview and state the dominant sentiment, leading topic, and leading emotion.",
        "Open Topics, select the dominant topic, and show representative comments with their labels.",
        "Open Report and download the stakeholder summary.",
        "Close by stating that another URL replaces the current results; historical/research data remains developer-side.",
    ]:
        _add_numbered(doc, step)
    _add_callout(
        doc,
        "Demo safeguard",
        "Use a known public post and test network/API credentials before the defence. Keep one screenshot or exported report as a fallback if live collection is unavailable.",
        fill=LIGHT_GOLD,
    )
    doc.add_page_break()


def _add_policy_and_limits(doc: Document, topics: pd.DataFrame) -> None:
    doc.add_heading("Policy Interpretation And Limits", level=1)
    doc.add_heading("Evidence-linked considerations", level=2)
    topic_by_name = topics.set_index("topic")
    recommendations = [
        (
            "Load shedding and reliability",
            "Publish consistent schedules, explain unavoidable changes promptly, and provide restoration updates that communities can verify.",
        ),
        (
            "Faults, connections and infrastructure",
            "Separate fault, connection, and maintenance issues in reporting; publish ownership, escalation routes, and expected resolution times.",
        ),
        (
            "Customer service and communication",
            "Use clear response standards and short, regular updates that answer the questions appearing most often.",
        ),
        (
            "Governance and public trust",
            "Support announcements with measurable progress, named responsibilities, and follow-up evidence.",
        ),
    ]
    rows = []
    for topic, recommendation in recommendations:
        row = topic_by_name.loc[topic]
        rows.append(
            [
                topic,
                f"{int(row['comment_count']):,} comments; {float(row['negative_percent']):.1f}% negative",
                recommendation,
            ]
        )
    _add_table(
        doc,
        ["Evidence area", "Signal", "Recommended communication response"],
        rows,
        [2800, 2200, 4360],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] * 3,
    )

    doc.add_heading("Interpretation guardrails", level=2)
    for item in [
        "Facebook comments are unsolicited reactions, not a representative survey of Zambia.",
        "One post can attract a highly specific audience; results should not be generalized to the entire population.",
        "Automated sentiment, emotion, sarcasm, and topic labels can be wrong and require human review for high-impact use.",
        "Low-volume topics should not drive policy conclusions without supporting operational or survey evidence.",
        "Only publicly accessible content is collected; author names are removed from project outputs.",
        "Recommendations support communication and engagement decisions; the system does not make policy decisions.",
    ]:
        _add_bullet(doc, item)

    doc.add_heading("Future developer work", level=2)
    for item in [
        "Build a separate manual review and annotation dashboard for researchers and authorized developers.",
        "Create a human-labeled Zambian gold-standard test set with annotation guidelines and agreement checks.",
        "Add model/version tracking, drift monitoring, and error slices for negation, sarcasm, and code-switching.",
        "Compare the baseline with an efficiently evaluated RoBERTa or Africa-focused language model branch.",
        "Attach recommendation provenance so each report statement can be traced to supporting comments and metrics.",
    ]:
        _add_bullet(doc, item)
    doc.add_page_break()


def _add_questions(doc: Document) -> None:
    doc.add_heading("Likely Defence Questions", level=1)
    questions = [
        (
            "Why Facebook?",
            "It is a major venue for unsolicited public discussion in Zambia and provides timely reactions that complement formal surveys and hearings.",
        ),
        (
            "Why VADER?",
            "VADER is transparent, reproducible, and designed for social-media language. It also supports explainable local lexicon and sarcasm corrections.",
        ),
        (
            "Why use machine learning if VADER creates the labels?",
            "VADER supplies weak supervision for a prototype where manual labels are limited. The classifiers test whether reusable patterns can be learned from those labels; a human gold standard is the next step.",
        ),
        (
            "Why is the best accuracy only 58.4%?",
            "The data is noisy, code-switched, and weakly labeled. The score is a realistic baseline, not a production claim. The prototype prioritizes transparency and identifies where annotation investment is needed.",
        ),
        (
            "How does the system handle Zambian language?",
            "It includes local sentiment and emotion lexicons, Zambia-energy relevance terms, code-switched variants, and sarcasm rules. These are auditable but still incomplete.",
        ),
        (
            "How do you know recommendations are trustworthy?",
            "Each recommendation is generated from topic volume, negative share, emotion signals, and trend evidence. The report also states limitations and requires human interpretation.",
        ),
        (
            "What changed after the positive-label error?",
            "Rule-based analysis now uses readable text that preserves negation. Regression tests were added and all dependent labels and models were regenerated.",
        ),
        (
            "Why hide the research dataset from users?",
            "The end-user task is to analyze a supplied URL. Research corpora and diagnostic controls would add confusion and belong in a separate developer/annotation interface.",
        ),
    ]
    for question, answer in questions:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        q_run = p.add_run(question)
        _set_run_font(q_run, size=11.5, color=DARK_BLUE, bold=True)
        answer_p = doc.add_paragraph(answer)
        answer_p.paragraph_format.left_indent = Inches(0.18)
        answer_p.paragraph_format.space_after = Pt(7)
    doc.add_page_break()


def _add_close(doc: Document) -> None:
    doc.add_heading("Closing Statement", level=1)
    _add_callout(
        doc,
        "Suggested close",
        "This prototype demonstrates that public Facebook discourse can be transformed into structured, auditable evidence about Zambian energy concerns. The strongest finding is that reliability dominates discussion, while the strongest technical lesson is that local language context and negation must be preserved. The current system is suitable for exploratory decision support; the next research milestone is human annotation and stronger external validation.",
        fill=LIGHT_GOLD,
    )
    doc.add_heading("Evidence files used for this guide", level=2)
    evidence = [
        ("Corrected relevant labels", "data/processed/labeled/final_label_relevant.csv"),
        ("Held-out model metrics", "data/results/evaluation_summary.csv"),
        ("Classification reports", "data/results/classification_reports.json"),
        ("Confusion matrices", "data/results/confusion_matrices/"),
        ("Dashboard application", "src/dashboard/app.py"),
        ("Live URL pipeline", "src/dashboard/live_analysis.py"),
        ("Regression tests", "tests/test_sentiment.py and tests/test_live_analysis.py"),
    ]
    _add_table(
        doc,
        ["Evidence", "Project path"],
        [[label, path] for label, path in evidence],
        [3000, 6360],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT],
    )
    _add_paragraph(
        doc,
        "Results frozen for this guide on 3 April 2026. Regenerate the guide after any future relabeling or model-training run.",
        size=9.5,
        color=MUTED,
        italic=True,
        after=0,
    )


def build_document() -> Path:
    frame = pd.read_csv(DATA_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    topics = _topic_summary(frame)

    doc = Document()
    _configure_document(doc)
    doc.core_properties.title = "Project Defence Results Guide"
    doc.core_properties.subject = "Zambian green energy sentiment analysis prototype"
    doc.core_properties.author = "Mulenga Musamba"
    doc.core_properties.keywords = "sentiment analysis, Zambia, green energy, Facebook, project defence"
    doc.core_properties.comments = "Generated from the corrected project evaluation outputs."

    _add_cover(doc)
    _add_at_a_glance(doc, frame, metrics)
    _add_system_flow(doc)
    _add_results(doc, frame, topics)
    _add_model_results(doc, metrics)
    _add_quality_case(doc)
    _add_dashboard_demo(doc)
    _add_policy_and_limits(doc, topics)
    _add_questions(doc)
    _add_close(doc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    result = build_document()
    print(result)
