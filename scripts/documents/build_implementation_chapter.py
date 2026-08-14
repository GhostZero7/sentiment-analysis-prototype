"""Generate the guideline-aligned Chapter 5 implementation document."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw

from build_defence_results_guide import (
    BLUE,
    DARK_BLUE,
    GOLD,
    INK,
    LIGHT_GOLD,
    MUTED,
    NAVY,
    _add_bullet,
    _add_callout,
    _add_numbered,
    _add_page_field,
    _add_paragraph,
    _add_picture,
    _add_table,
    _configure_document,
    _font,
    _image_buffer,
    _rgb,
    _set_run_font,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "Chapter_5_Implementation_Mulenga_Musamba_22110887.docx"
ASSETS = ROOT / "docs" / "assets"
METRICS = ROOT / "data" / "results" / "evaluation_summary.csv"
LABELS = ROOT / "data" / "processed" / "labeled" / "final_label_relevant.csv"


def configure_report(doc: Document) -> None:
    """Apply the standard_business_brief preset with an academic cover override."""
    _configure_document(doc)
    normal = doc.styles["Normal"]
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    tokens = {
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (12, DARK_BLUE, 8, 4),
    }
    for name, (size, color, before, after) in tokens.items():
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.color.rgb = _rgb(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)

    for name in ("List Bullet", "List Number"):
        style = doc.styles[name]
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = 1.167

    if "Code Block" not in [style.name for style in doc.styles]:
        code = doc.styles.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
        code.font.name = "Consolas"
        code._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
        code._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
        code.font.size = Pt(8.5)
        code.font.color.rgb = _rgb(INK)
        code.paragraph_format.left_indent = Inches(0.18)
        code.paragraph_format.right_indent = Inches(0.12)
        code.paragraph_format.space_before = Pt(4)
        code.paragraph_format.space_after = Pt(8)
        code.paragraph_format.line_spacing = 1.0

    section = doc.sections[0]
    header = section.header.paragraphs[0]
    header.clear()
    run = header.add_run("CHAPTER 5  |  SYSTEM IMPLEMENTATION")
    _set_run_font(run, size=8.5, color=MUTED, bold=True)
    footer = section.footer.paragraphs[0]
    footer.clear()
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Mulenga Musamba  |  ")
    _set_run_font(run, size=8.5, color=MUTED)
    _add_page_field(footer, "PAGE")


def add_cover(doc: Document) -> None:
    _add_paragraph(doc, "THE COPPERBELT UNIVERSITY", size=13, bold=True,
                   color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER, before=56, after=7)
    _add_paragraph(doc, "SCHOOL OF INFORMATION AND COMMUNICATION TECHNOLOGY",
                   size=10.5, bold=True, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
    _add_paragraph(doc, "COMPUTER SCIENCE DEPARTMENT  |  CS 400: PROJECT SEMINARS",
                   size=10, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=72)
    _add_paragraph(doc, "CHAPTER 5", size=11, bold=True, color=GOLD,
                   align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    _add_paragraph(doc, "IMPLEMENTATION", size=30, bold=True, color=NAVY,
                   align=WD_ALIGN_PARAGRAPH.CENTER, after=18)
    _add_paragraph(doc,
        "A Sentiment Analysis Prototype for Zambian Green Energy Discourse on Social Media",
        size=16, color=DARK_BLUE, align=WD_ALIGN_PARAGRAPH.CENTER, after=58)
    _add_paragraph(doc, "Submitted by", size=10, color=MUTED,
                   align=WD_ALIGN_PARAGRAPH.CENTER, after=3)
    _add_paragraph(doc, "Mulenga Musamba  |  Student ID: 22110887", size=11.5,
                   bold=True, color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    _add_paragraph(doc, "Supervisor: Dr. George Mufungulwa", size=11,
                   color=INK, align=WD_ALIGN_PARAGRAPH.CENTER, after=42)
    _add_paragraph(doc, "August 2026", size=10.5, color=MUTED,
                   align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    doc.add_page_break()


def add_contents(doc: Document) -> None:
    doc.add_heading("Chapter Contents", level=1)
    entries = [
        ("5.1", "Introduction"),
        ("5.2", "System Implementation"),
        ("5.2.1", "Development and Deployment Setup"),
        ("5.2.2", "Software and Hardware"),
        ("5.2.3", "Implemented Architecture and Components"),
        ("5.2.4", "Algorithms and Functional Implementation"),
        ("5.2.5", "Installation, Deployment and Operation"),
        ("5.2.6", "Training, Project, Risk and Configuration Management"),
        ("5.2.7", "Troubleshooting and Further Work"),
        ("5.3", "Coding"),
        ("5.4", "Results"),
        ("5.5", "Conclusion"),
    ]
    _add_table(doc, ["Section", "Title"], [[a, b] for a, b in entries], [1500, 7860],
               header_fill="F2F4F7")
    _add_callout(doc, "Scope note",
        "This chapter documents the prototype as implemented and tested. Features described only in the earlier design, such as a database-backed production deployment and formal multi-user load testing, are identified as future work rather than reported as completed.",
        fill=LIGHT_GOLD)
    doc.add_page_break()


def architecture_image() -> io.BytesIO:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    title = _font(42, bold=True)
    heading = _font(27, bold=True)
    body = _font(23)
    draw.text((750, 40), "Implemented three-layer architecture", font=title, fill="#203748", anchor="ma")
    layers = [
        (105, 140, 1395, 310, "Presentation layer", "Streamlit: Analyze URL, Overview, Trends, Topics and Report", "#E8EEF5"),
        (105, 365, 1395, 590, "Application layer", "Cache lookup | Apify connector | preprocessing | relevance | sentiment | emotion | topics | reporting", "#F2F4F7"),
        (105, 645, 1395, 815, "Data layer", "CSV audit/cache files | labeled datasets | TF-IDF vectorizer | serialized ML models | evaluation outputs", "#FFF7E1"),
    ]
    for x0, y0, x1, y1, label, detail, fill in layers:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=fill, outline="#7B8794", width=3)
        draw.text((150, y0 + 42), label, font=heading, fill="#1F4D78")
        draw.text((150, y0 + 96), detail, font=body, fill="#202124")
    for y0, y1 in ((310, 365), (590, 645)):
        draw.line((750, y0 + 5, 750, y1 - 10), fill="#2E74B5", width=7)
        draw.polygon([(737, y1 - 22), (763, y1 - 22), (750, y1 - 5)], fill="#2E74B5")
    return _image_buffer(image)


def workflow_image() -> io.BytesIO:
    image = Image.new("RGB", (1600, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((800, 35), "URL analysis workflow", font=_font(42, bold=True), fill="#203748", anchor="ma")
    boxes = [
        ("Validate and canonicalise URL", "Reject invalid or unsupported input"),
        ("Check local post cache", "Match by Facebook content identifier"),
        ("Fetch only when unseen", "Apify, maximum 100, nested comments disabled"),
        ("Harden collected rows", "Reject replies, strip structured mentions, deduplicate"),
        ("Analyse readable text", "Relevance, VADER + local rules, sarcasm, NRC emotions"),
        ("Aggregate current URL", "Topics, trends, policy report and downloadable results"),
    ]
    y = 120
    for idx, (title, detail) in enumerate(boxes):
        fill = "#FFF7E1" if idx == 1 else ("#E8EEF5" if idx in (0, 4, 5) else "#F2F4F7")
        draw.rounded_rectangle((220, y, 1380, y + 120), radius=10, fill=fill, outline="#8A949E", width=3)
        draw.text((270, y + 28), f"{idx + 1}. {title}", font=_font(27, bold=True), fill="#1F4D78")
        draw.text((270, y + 72), detail, font=_font(22), fill="#202124")
        if idx < len(boxes) - 1:
            draw.line((800, y + 120, 800, y + 155), fill="#2E74B5", width=6)
            draw.polygon([(788, y + 145), (812, y + 145), (800, y + 160)], fill="#2E74B5")
        y += 155
    return _image_buffer(image)


def gantt_image() -> io.BytesIO:
    image = Image.new("RGB", (1600, 840), "white")
    draw = ImageDraw.Draw(image)
    draw.text((800, 35), "Completed implementation timeline, January-August 2026",
              font=_font(38, bold=True), fill="#203748", anchor="ma")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    left, top, cell_w, row_h = 480, 125, 130, 82
    for i, month in enumerate(months):
        draw.text((left + i * cell_w + cell_w / 2, top - 25), month, font=_font(22, bold=True), fill="#5F6368", anchor="mm")
        draw.line((left + i * cell_w, top, left + i * cell_w, 760), fill="#E0E3E7", width=2)
    draw.line((left + len(months) * cell_w, top, left + len(months) * cell_w, 760), fill="#E0E3E7", width=2)
    tasks = [
        ("Proposal and literature", 0, 1),
        ("Methodology and design", 1, 3),
        ("Collection and preprocessing", 2, 5),
        ("Lexicons and model training", 3, 5),
        ("Dashboard implementation", 4, 6),
        ("URL cache and hardening", 5, 7),
        ("Testing and documentation", 6, 7),
    ]
    for row, (label, start, end) in enumerate(tasks):
        y = top + row * row_h + 34
        draw.text((left - 25, y), label, font=_font(21, bold=True), fill="#202124", anchor="rm")
        x0, x1 = left + start * cell_w + 10, left + (end + 1) * cell_w - 10
        color = "#2E74B5" if row not in (5, 6) else "#9A7218"
        draw.rounded_rectangle((x0, y - 18, x1, y + 18), radius=8, fill=color)
    return _image_buffer(image)


def add_intro(doc: Document) -> None:
    doc.add_heading("5.1 Introduction", level=1)
    _add_paragraph(doc,
        "This chapter explains how the approved system design was converted into a working prototype for analysing public Facebook comments about green energy and electricity services in Zambia. The implemented system accepts a public Facebook post URL, obtains or reuses its top-level comments, applies relevance, sentiment, emotion and topic analysis, and presents URL-specific findings through a Streamlit dashboard.")
    _add_paragraph(doc,
        "Implementation was iterative. Error analysis led to local language and domain corrections, preservation of readable text for negation-sensitive analysis, cache-first collection to reduce external API cost, and defensive filtering of replies and profile mentions. The chapter also records installation, testing, deployment, project controls, troubleshooting and limitations so that the prototype can be reproduced and defended without overstating its maturity.")
    _add_callout(doc, "Implemented objective",
        "Transform one user-supplied public Facebook URL into auditable sentiment, emotion, topic, trend and policy-oriented results while keeping research datasets and developer controls outside the end-user dashboard.")


def add_setup(doc: Document) -> None:
    doc.add_heading("5.2 System Implementation", level=1)
    doc.add_heading("5.2.1 Development and Deployment Setup", level=2)
    _add_paragraph(doc,
        "The prototype was developed and tested on Windows 11 using Python 3.13.1, Git 2.53.0 and a local Streamlit server. Source code is organised by responsibility under src/, while scripts/, tests/, data/ and docs/ contain repeatable operations, automated checks, data artifacts and project documentation respectively. The same Python application serves both development and local demonstration deployment.")
    _add_table(doc, ["Environment", "Verified configuration"], [
        ["Operating system", "Windows 11 Pro, version 10.0.22621"],
        ["Processor", "Intel Core i3-8145U at 2.10 GHz"],
        ["Memory", "7.4 GB usable RAM"],
        ["Programming runtime", "Python 3.13.1"],
        ["Version control", "Git 2.53.0; GitHub remote repository"],
        ["Application server", "Streamlit local web server on port 8501"],
        ["External service", "Apify Facebook comments actor for unseen public URLs"],
        ["Persistence", "CSV/JSON audit files and Joblib model artifacts"],
    ], [2700, 6660], header_fill="F2F4F7")
    _add_paragraph(doc,
        "A relational DBMS was considered in the design, but the implemented prototype deliberately uses file-based persistence. This keeps the research workflow transparent and portable; a database becomes appropriate when authentication, concurrent users, retention rules or centrally managed deployment are added.")

    doc.add_heading("5.2.2 Software and Hardware", level=2)
    _add_table(doc, ["Software/component", "Purpose in the implementation"], [
        ["Streamlit", "End-user URL input and Overview, Trends, Topics and Report views"],
        ["Apify Client", "Collection of publicly accessible comments for previously unseen URLs"],
        ["pandas", "Tabular cleaning, deduplication, persistence and aggregation"],
        ["NLTK / VADER", "Transparent social-media polarity scoring"],
        ["NRCLex and local lexicons", "Emotion scoring and Zambia-specific language adaptation"],
        ["scikit-learn / Joblib", "TF-IDF features, Naive Bayes, Logistic Regression, SVM and model storage"],
        ["Matplotlib", "Charts used in evaluation and downloadable outputs"],
        ["pytest", "Automated unit and integration regression testing"],
    ], [2900, 6460], header_fill="F2F4F7")
    _add_paragraph(doc,
        "The verified development laptop is sufficient for the classical machine-learning pipeline and local dashboard. At least 8 GB RAM, a modern dual-core processor, Python 3.11 or newer, reliable internet for first-time collection, and free disk space for model and CSV artifacts are recommended. Transformer models are optional and require substantially more memory and processing time.")
    doc.add_page_break()


def add_architecture(doc: Document) -> None:
    doc.add_heading("5.2.3 Implemented Architecture and Components", level=2)
    _add_picture(doc, architecture_image(), 6.2,
        "Three-layer architecture with Streamlit presentation, analysis modules and file/model data layer.",
        "Figure 5.1. Implemented three-layer architecture of the prototype.")
    _add_paragraph(doc,
        "The presentation layer contains the end-user dashboard. The application layer coordinates URL validation, cache lookup, collection, text preparation, relevance detection, sentiment and emotion scoring, topic aggregation, temporal analysis and report generation. The data layer stores raw post caches, processed and labeled CSV files, evaluation outputs, lexicons and serialized TF-IDF/model artifacts.")
    _add_table(doc, ["Deliverable", "Implemented component", "Status"], [
        ["URL-based collection", "src/data_collection and src/dashboard/live_analysis.py", "Implemented"],
        ["Sentiment and sarcasm", "src/sentiment with VADER and local correction rules", "Implemented"],
        ["Emotion detection", "NRC plus local emotion lexicon", "Implemented"],
        ["Classical ML models", "Naive Bayes, Logistic Regression and SVM", "Implemented"],
        ["End-user dashboard", "Analyze URL, Overview, Trends, Topics and Report", "Implemented"],
        ["Developer annotation dashboard", "Manual gold-label review workflow", "Future work"],
        ["Production database", "Central multi-user persistence and access control", "Future work"],
    ], [2650, 4710, 2000], header_fill="F2F4F7")
    doc.add_page_break()


def add_algorithms(doc: Document) -> None:
    doc.add_heading("5.2.4 Algorithms and Functional Implementation", level=2)
    _add_picture(doc, workflow_image(), 6.1,
        "Six-stage URL workflow from validation and cache lookup to URL-specific reporting.",
        "Figure 5.2. Cache-first URL analysis and reporting workflow.")
    _add_paragraph(doc,
        "The URL is canonicalised to remove non-essential query parameters and identify the Facebook post or video. A matching local cache is loaded before any external request. Consequently, repeated analysis remains available when Apify credit or network access is unavailable. For an unseen URL, collection is capped at 100 comments to control cost.")
    _add_paragraph(doc,
        "The collector requests top-level comments by setting includeNestedComments to false. Defensive schema checks also reject rows containing reply depth, parent-comment identifiers or reply flags. Structured mention names are removed and duplicate comment identifiers/text pairs are discarded before analysis.")
    _add_paragraph(doc,
        "Two text representations are retained. Readable text preserves punctuation, contractions and negation for VADER, sarcasm, emotion and relevance rules. Processed text supports TF-IDF machine-learning features. This separation fixed a class of errors in which cleaning removed the grammatical context needed to interpret complaints.")
    _add_table(doc, ["Stage", "Algorithm or rule", "Output"], [
        ["Relevance", "Zambian energy vocabulary and contextual inclusion rules", "Relevant/off-topic flag"],
        ["Sentiment", "VADER compound thresholds with local phrase correction", "Negative, neutral or positive"],
        ["Sarcasm", "Auditable cue and polarity-conflict heuristics", "Sarcasm flag and adjustment"],
        ["Emotion", "NRC categories plus Zambia-specific emotion phrases", "Leading emotion and scores"],
        ["Machine learning", "Word/character TF-IDF with NB, LR and linear SVM", "Comparable baseline predictions"],
        ["Topics", "Transparent domain keyword taxonomy", "Topic labels, counts and examples"],
        ["Reporting", "Evidence thresholds over topic, sentiment, emotion and time", "Stakeholder findings and policy considerations"],
    ], [1800, 4660, 2900], header_fill="F2F4F7")
    _add_callout(doc, "Latest quality correction",
        "The phrase 'not making sense' was added as a negative local correction. The reviewed business-impact complaint now changes from VADER raw +0.2375 (positive) to corrected -0.5625 (negative), and an exact regression test protects this behaviour.",
        fill=LIGHT_GOLD)
    doc.add_page_break()


def add_operation(doc: Document) -> None:
    doc.add_heading("5.2.5 Installation, Deployment and Operation", level=2)
    _add_paragraph(doc, "The local installation and demonstration procedure is:")
    steps = [
        "Clone the Git repository and open a terminal in the project root.",
        "Create and activate a Python virtual environment.",
        "Install dependencies with: python -m pip install -r requirements.txt.",
        "Create a .env file containing APIFY_API_TOKEN and, when required, the configured actor identifier. Secrets are excluded from version control.",
        "Run automated verification with: python -m pytest -q.",
        "Start the dashboard with: python -m streamlit run src/dashboard/app.py.",
        "Open http://localhost:8501, paste a public Facebook post URL, choose a limit of 100 or less, and select Analyze public comments.",
    ]
    for step in steps:
        _add_numbered(doc, step)
    _add_paragraph(doc,
        "No legacy organisational system is replaced by this prototype, so direct cutover and transactional data migration are not applicable. Research CSV files were normalised into the project schema, then training/test partitions and model artifacts were regenerated. Deployment is currently local and single-instance; GitHub provides code distribution and configuration history.")

    doc.add_heading("User Training", level=3)
    _add_paragraph(doc,
        "End-user training is intentionally brief because the dashboard begins with the URL task. A demonstration should cover public-URL selection, the comment cap, cache messages, the meaning of sentiment/emotion/topic summaries, report download, and the warning that Facebook comments are not a representative survey. Users should validate high-impact findings against example comments and operational evidence.")
    _add_table(doc, ["Training topic", "Expected user outcome"], [
        ["URL selection", "Recognise a publicly accessible Facebook post or video URL"],
        ["Result interpretation", "Distinguish volume, polarity, emotion, topic and trend evidence"],
        ["Cache behaviour", "Understand that saved comments can be analysed without new Apify usage"],
        ["Ethical use", "Avoid identifying authors or treating results as population-wide opinion"],
        ["Reporting", "Download and present a URL-specific report with its limitations"],
    ], [3000, 6360], header_fill="F2F4F7")
    doc.add_page_break()


def add_management(doc: Document) -> None:
    doc.add_heading("5.2.6 Project, Risk and Configuration Management", level=2)
    _add_picture(doc, gantt_image(), 6.2,
        "Gantt chart covering proposal, design, collection, models, dashboard, hardening, testing and documentation from January to August 2026.",
        "Figure 5.3. Completed implementation timeline.")
    _add_paragraph(doc,
        "The project followed incremental delivery: requirements and literature informed the methodology and design; collection and preprocessing enabled lexicon/model work; dashboard integration exposed data-quality issues; testing and documentation consolidated the prototype. Git commits and generated artifacts provide a traceable implementation history.")
    _add_table(doc, ["Risk", "Implemented control", "Residual limitation"], [
        ["Apify credit or outage", "Cache-first lookup; maximum 100 comments", "Unseen URLs still require the external service"],
        ["Replies/names entering data", "Disable nested comments; reject parent-linked rows; strip mentions", "Scraper schemas can change"],
        ["Incorrect sentiment", "Local lexicons, readable-text scoring and regression tests", "Sarcasm and code-switching remain difficult"],
        ["Privacy and misuse", "Public comments only; no profile names in analysis outputs", "Platform and research ethics still require review"],
        ["Model overstatement", "Held-out metrics and explicit prototype limitations", "No independently human-labeled gold standard yet"],
        ["Configuration leakage", ".env secrets excluded; dependencies recorded", "Deployment secret management is not centralised"],
    ], [2250, 3860, 3250], header_fill="F2F4F7")
    _add_paragraph(doc,
        "Configuration management uses Git for source history, requirements.txt for Python dependencies, .env for local secrets, Joblib for model/vectorizer versions and dated CSV/JSON files for collection and evaluation evidence. Generated raw and result files are retained as an audit trail; future production deployment should add artifact version identifiers, retention policies and automated release tagging.")

    doc.add_heading("5.2.7 Troubleshooting and Further Work", level=2)
    _add_table(doc, ["Symptom", "Likely cause and response"], [
        ["Apify usage exceeded", "Use a previously cached URL; add credit only when a genuinely unseen URL must be collected"],
        ["No comments returned", "Confirm that the post is public, the URL is supported and comments are visible"],
        ["Too few relevant comments", "The discussion may be off-topic; inspect the relevance examples rather than forcing inclusion"],
        ["Unexpected sentiment", "Inspect raw score, matched local phrase and readable text; add a tested correction only when the linguistic pattern generalises"],
        ["Dashboard does not start", "Activate the environment, install requirements and verify that port 8501 is available"],
        ["Trend view is empty", "Valid timestamps and more than one time period are required"],
    ], [2600, 6760], header_fill="F2F4F7")
    for item in [
        "Build a separate manual review and annotation dashboard for authorised researchers and developers.",
        "Create a human-labeled Zambian gold-standard dataset and report inter-annotator agreement.",
        "Add a production database, authentication, retention controls and formal concurrent-user testing.",
        "Evaluate Africa-focused or fine-tuned transformer models against the transparent baselines.",
        "Add model/version provenance and drift monitoring to every downloaded stakeholder report.",
    ]:
        _add_bullet(doc, item)
    doc.add_page_break()


def add_code(doc: Document) -> None:
    doc.add_heading("5.3 Coding", level=1)
    _add_paragraph(doc,
        "Python was selected because its data-processing and natural-language libraries support reproducible experimentation, while Streamlit provides a compact web interface without duplicating analysis logic in a second language. The modular package structure allows collection, sentiment, insights and dashboard behaviour to be tested independently.")
    snippets = [
        ("5.3.1 Cache-first URL execution", "cached = load_cached_comments(source_url, max_comments=max_comments)\nif cached:\n    comments = cached.comments\n    collection_source = \"local_cache\"\nelse:\n    comments = fetch_comments(source_url, max_comments=max_comments)\n    save_post_cache(comments, source_url=source_url)"),
        ("5.3.2 Top-level comment protection", "run_input = {\n    \"startUrls\": [{\"url\": source_url}],\n    \"maxComments\": min(max_comments, 100),\n    \"includeNestedComments\": False,\n}\nif _is_reply_item(item):\n    return None"),
        ("5.3.3 Local sentiment correction", "local = apply_local_correction(text, raw_compound, raw_label)\nfinal_compound = local[\"compound\"]\nfinal_label = local[\"label\"]\n# Example: 'not making sense' applies a negative domain correction."),
    ]
    for title, code in snippets:
        doc.add_heading(title, level=2)
        p = doc.add_paragraph(style="Code Block")
        p.paragraph_format.keep_together = True
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "F2F4F7")
        p._p.get_or_add_pPr().append(shading)
        run = p.add_run(code)
        _set_run_font(run, name="Consolas", size=8.5, color=INK)
    _add_paragraph(doc,
        "Complete source listings remain in the repository. The fragments above demonstrate the three implementation decisions most important to reliability and cost: local reuse before external collection, reply exclusion at both request and row level, and auditable local correction after a transparent base score.")
    doc.add_page_break()


def add_path_picture(doc: Document, path: Path, caption: str, alt: str, width: float = 6.2) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        buffer.seek(0)
    _add_picture(doc, buffer, width, alt, caption)


def add_results(doc: Document) -> None:
    metrics = pd.read_csv(METRICS)
    labels = pd.read_csv(LABELS)
    counts = labels["corrected_label"].value_counts()
    doc.add_heading("5.4 Results", level=1)
    _add_paragraph(doc,
        "Verification combined automated tests, held-out model evaluation, manual error analysis and an end-to-end dashboard run using a cached Facebook URL. The full automated suite completed successfully with 66 passing tests. The cached demonstration loaded 20 stored top-level comments without consuming Apify tokens; nine were classified as energy-relevant and all nine were negative for that particular post.")
    _add_table(doc, ["Evidence", "Verified result"], [
        ["Automated regression suite", "66 tests passed"],
        ["Relevant labeled dataset", f"{len(labels):,} comments: {int(counts.get('negative', 0)):,} negative, {int(counts.get('neutral', 0)):,} neutral, {int(counts.get('positive', 0)):,} positive"],
        ["Naive Bayes", f"{metrics.loc[metrics.model.eq('naive_bayes'), 'accuracy'].iloc[0] * 100:.1f}% accuracy; {metrics.loc[metrics.model.eq('naive_bayes'), 'f1_weighted'].iloc[0] * 100:.1f}% weighted F1"],
        ["Logistic Regression", f"{metrics.loc[metrics.model.eq('logistic_regression'), 'accuracy'].iloc[0] * 100:.1f}% accuracy; {metrics.loc[metrics.model.eq('logistic_regression'), 'f1_weighted'].iloc[0] * 100:.1f}% weighted F1"],
        ["Support Vector Machine", f"{metrics.loc[metrics.model.eq('svm'), 'accuracy'].iloc[0] * 100:.1f}% accuracy; {metrics.loc[metrics.model.eq('svm'), 'f1_weighted'].iloc[0] * 100:.1f}% weighted F1"],
        ["Reviewed complaint", "Raw +0.2375 positive; corrected -0.5625 negative"],
    ], [3000, 6360], header_fill="F2F4F7")
    _add_callout(doc, "Interpretation",
        "SVM is the strongest of the three classical baselines, but the 58.4% accuracy and 58.3% weighted F1 are moderate. These are prototype results on noisy, code-switched social-media text with weak supervision; they are not evidence of production-grade accuracy.", fill=LIGHT_GOLD)

    add_path_picture(doc, ASSETS / "implementation_analyze_url.png",
        "Figure 5.4. The end-user dashboard begins with the public Facebook URL workflow.",
        "Analyze URL tab showing URL entry and maximum comment controls.")
    add_path_picture(doc, ASSETS / "implementation_cache_result.png",
        "Figure 5.5. A saved post was loaded locally, avoiding new Apify usage.",
        "Dashboard confirmation showing local cache usage and fetched/relevant totals.")
    add_path_picture(doc, ASSETS / "implementation_overview.png",
        "Figure 5.6. URL-specific overview for the cached demonstration post.",
        "Overview showing nine comments, negative sentiment, frustration and load-shedding topic.")
    _add_paragraph(doc,
        "The screenshots demonstrate the implemented user journey rather than the research corpus. Results are replaced when another URL is analysed. The overview combines distribution, topic and emotion evidence, while the report tab converts the same URL-specific evidence into cautious stakeholder findings and future policy considerations.")
    doc.add_page_break()


def add_conclusion(doc: Document) -> None:
    doc.add_heading("5.5 Conclusion", level=1)
    _add_paragraph(doc,
        "The implementation converts the three-layer design into a functioning, testable prototype. It provides URL-based public comment collection, cache-first reuse, top-level-comment hardening, readable-text sentiment and emotion analysis, three classical machine-learning baselines, transparent topic aggregation, trends and an end-user stakeholder report.")
    _add_paragraph(doc,
        "The work also demonstrates the value of iterative validation. Manual review exposed misleading positive classifications caused by contextual words and lost negation. Local phrase correction, text-representation separation and regression tests improved the system while preserving an auditable raw score. The prototype is suitable for exploratory research and decision support, but policy use must remain supported by human review and independent evidence.")
    _add_paragraph(doc,
        "The next priority is a developer-only annotation dashboard and a human-labeled Zambian evaluation set. Production deployment should then add central persistence, authentication, stronger configuration management, concurrency testing and model monitoring. These extensions build on the current end-user dashboard without changing its focused URL-to-results workflow.")
    doc.add_heading("Implementation Evidence", level=2)
    _add_table(doc, ["Artifact", "Project location"], [
        ["Dashboard", "src/dashboard/app.py"],
        ["Live URL pipeline", "src/dashboard/live_analysis.py"],
        ["Collector hardening", "src/data_collection/apify_client.py"],
        ["Local sentiment lexicon", "data/lexicons/local_sentiment_lexicon.csv"],
        ["Automated tests", "tests/"],
        ["Model evaluation", "data/results/evaluation_summary.csv"],
        ["User and defence documentation", "docs/"],
    ], [3000, 6360], header_fill="F2F4F7")
    _add_paragraph(doc,
        f"Document generated from the verified project state on {date.today().strftime('%d %B %Y')}.",
        size=9.5, color=MUTED, italic=True, after=0)


def build() -> Path:
    doc = Document()
    configure_report(doc)
    doc.core_properties.title = "Chapter 5: Implementation"
    doc.core_properties.subject = "Zambian green energy sentiment analysis prototype"
    doc.core_properties.author = "Mulenga Musamba"
    doc.core_properties.keywords = "implementation, sentiment analysis, Zambia, Facebook, Streamlit"
    doc.core_properties.comments = "Generated from verified implementation artifacts and CS 400 project guidelines."
    add_cover(doc)
    add_contents(doc)
    add_intro(doc)
    add_setup(doc)
    add_architecture(doc)
    add_algorithms(doc)
    add_operation(doc)
    add_management(doc)
    add_code(doc)
    add_results(doc)
    add_conclusion(doc)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
