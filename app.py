"""Focused Streamlit interface for worldwide flag recognition."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import sys
from time import strftime

import pandas as pd
from PIL import Image
import requests
import streamlit as st
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as PDFImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.inference import load_inference_bundle, predict_image
from flag_recognition.taxonomy import country_name_from_code


DISPLAY_NAME_OVERRIDES = {
    "kr": "South Korea",
    "kp": "North Korea",
    "ir": "Iran",
    "bo": "Bolivia",
    "ve": "Venezuela",
    "tz": "Tanzania",
    "md": "Moldova",
    "la": "Laos",
    "bn": "Brunei",
    "sy": "Syria",
    "ps": "Palestine",
    "tw": "Taiwan",
    "ru": "Russia",
    "vn": "Vietnam",
    "cz": "Czechia",
}


def display_country_name(code: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(
        code.lower(),
        country_name_from_code(code),
    )


VISUAL_EQUIVALENCE_GROUPS = {
    "fr": {"fr", "bl", "mf", "re", "yt"},
    "us": {"us", "um"},
    "nl": {"nl", "bq"},
    "no": {"no", "bv", "sj"},
    "au": {"au", "hm"},
    "gb": {"gb", "sh"},
}

VISUAL_EQUIVALENCE_LOOKUP = {
    member: canonical
    for canonical, members in VISUAL_EQUIVALENCE_GROUPS.items()
    for member in members
}


def merge_visually_identical_candidates(
    candidates: tuple[tuple[str, float], ...],
) -> list[tuple[str, float]]:
    """Merge probabilities for labels that use the same visible flag."""
    merged: dict[str, float] = {}

    for code, confidence in candidates:
        canonical = VISUAL_EQUIVALENCE_LOOKUP.get(code, code)
        merged[canonical] = merged.get(canonical, 0.0) + float(confidence)

    return sorted(
        merged.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def _pdf_text(value: object) -> str:
    if value is None:
        return "Not available"

    text = str(value).strip()
    return text if text else "Not available"


def get_pdf_logo_reader():
    """Return a ReportLab-safe RGB logo, or None if loading fails."""
    logo_path = ROOT_DIR / "assets" / "flag_intelligence_logo.png"

    if not logo_path.is_file():
        return None

    try:
        with Image.open(logo_path) as source:
            source.load()

            if source.mode in ("RGBA", "LA"):
                rgba = source.convert("RGBA")
                background = Image.new(
                    "RGB",
                    rgba.size,
                    "white",
                )
                background.paste(
                    rgba,
                    mask=rgba.getchannel("A"),
                )
                safe_logo = background
            else:
                safe_logo = source.convert("RGB")

            logo_buffer = BytesIO()
            safe_logo.save(
                logo_buffer,
                format="PNG",
                optimize=False,
            )
            logo_buffer.seek(0)

        return ImageReader(logo_buffer)

    except (OSError, ValueError):
        return None


def draw_pdf_watermark(canvas, document) -> None:
    """Draw a fixed corporate header, footer and subtle signature watermark."""
    canvas.saveState()

    page_width, page_height = A4
    left = 16 * mm
    right = page_width - 16 * mm

    # --- Fixed header ---
    canvas.setFillColor(colors.HexColor("#0F172A"))
    canvas.rect(
        0,
        page_height - 18 * mm,
        page_width,
        18 * mm,
        fill=1,
        stroke=0,
    )

    logo_reader = get_pdf_logo_reader()
    logo_size = 11 * mm
    logo_y = page_height - 15.5 * mm

    if logo_reader is not None:
        try:
            canvas.drawImage(
                logo_reader,
                left,
                logo_y,
                width=logo_size,
                height=logo_size,
                preserveAspectRatio=True,
                mask=None,
            )
            brand_x = left + 14 * mm
        except (OSError, ValueError):
            brand_x = left
    else:
        brand_x = left

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(
        brand_x,
        page_height - 10.5 * mm,
        "FLAG INTELLIGENCE",
    )

    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#CBD5E1"))
    canvas.drawString(
        brand_x,
        page_height - 14 * mm,
        "Country Flag Recognition System",
    )

    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(colors.white)
    canvas.drawRightString(
        right,
        page_height - 9.5 * mm,
        "RECOGNITION REPORT",
    )

    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(colors.HexColor("#CBD5E1"))
    canvas.drawRightString(
        right,
        page_height - 13.5 * mm,
        "Generated by Flag Intelligence",
    )

    # Accent line under header.
    canvas.setFillColor(colors.HexColor("#2563EB"))
    canvas.rect(
        0,
        page_height - 19 * mm,
        page_width,
        1 * mm,
        fill=1,
        stroke=0,
    )

    # --- Subtle signature watermark ---
    canvas.setFillAlpha(0.025)
    canvas.setFillColor(colors.HexColor("#0F172A"))
    canvas.setFont("Helvetica-Oblique", 26)
    canvas.translate(page_width / 2, page_height / 2)
    canvas.rotate(32)
    canvas.drawCentredString(
        0,
        0,
        "Denos Kume",
    )
    canvas.rotate(-32)
    canvas.translate(-page_width / 2, -page_height / 2)
    canvas.setFillAlpha(1)

    # --- Fixed footer ---
    footer_y = 12 * mm

    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.5)
    canvas.line(
        left,
        footer_y + 3 * mm,
        right,
        footer_y + 3 * mm,
    )

    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(
        left,
        footer_y,
        "Prepared by Denos Kume",
    )

    canvas.drawCentredString(
        page_width / 2,
        footer_y,
        "Sources: Wikidata | World Bank | Wikipedia",
    )

    canvas.drawRightString(
        right,
        footer_y,
        f"Page {document.page}",
    )

    canvas.setFont("Helvetica-Oblique", 6.5)
    canvas.setFillColor(colors.HexColor("#94A3B8"))
    canvas.drawCentredString(
        page_width / 2,
        footer_y - 3.2 * mm,
        "Automated report - verify time-sensitive public information when required.",
    )

    canvas.restoreState()



def _pdf_value(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_pdf_text(value), style)


def _profile_card(
    title: str,
    rows: list[tuple[str, object]],
    label_style: ParagraphStyle,
    value_style: ParagraphStyle,
) -> KeepTogether:
    data = [
        [
            Paragraph(f"<b>{label}</b>", label_style),
            _pdf_value(value, value_style),
        ]
        for label, value in rows
    ]

    table = Table(
        data,
        colWidths=[43 * mm, 113 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#EEF2F6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    return KeepTogether([
        Paragraph(title, ParagraphStyle(
            f"CardTitle_{title}",
            parent=value_style,
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
        )),
        table,
    ])


def build_pdf_report(
    report: dict[str, object],
    image: Image.Image,
) -> bytes:
    """Build a polished printable A4 recognition report."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=25 * mm,
        bottomMargin=22 * mm,
        title="Flag Intelligence - Recognition Report",
        author="Denos Kume",
        subject="Country flag recognition result",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=25,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=1.5 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
    )
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8.8,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    label_style = ParagraphStyle(
        "ReportLabel",
        parent=body_style,
        fontSize=8.3,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )
    value_style = ParagraphStyle(
        "ReportValue",
        parent=body_style,
        fontSize=8.8,
        leading=11.5,
        textColor=colors.HexColor("#0F172A"),
    )
    small_style = ParagraphStyle(
        "ReportSmall",
        parent=body_style,
        fontSize=7.6,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )
    metric_label_style = ParagraphStyle(
        "MetricLabel",
        parent=small_style,
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
    )
    metric_value_style = ParagraphStyle(
        "MetricValue",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_CENTER,
    )

    generated_at = _pdf_text(report.get("generated_at"))
    decision = _pdf_text(report.get("decision"))
    country_code = _pdf_text(report.get("country_code")).upper()
    confidence = float(report.get("confidence", 0.0))
    threshold = float(report.get("deployment_threshold", 0.0))
    accepted = bool(report.get("accepted"))
    status = "ACCEPTED" if accepted else "REJECTED"

    story: list[object] = []

    # Report identification block under the fixed page header.
    report_meta = Table(
        [[
            Paragraph(
                "<b>Country Flag Recognition Report</b>",
                ParagraphStyle(
                    "ReportIdentity",
                    parent=title_style,
                    fontSize=15,
                    leading=18,
                    alignment=0,
                ),
            ),
            Paragraph(
                f"<b>Report date</b><br/>{generated_at}",
                ParagraphStyle(
                    "ReportMetaCompact",
                    parent=small_style,
                    alignment=2,
                    leading=10,
                ),
            ),
        ]],
        colWidths=[105 * mm, 55 * mm],
    )
    report_meta.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.extend([
        report_meta,
        Spacer(1, 5 * mm),
    ])

    # Image preview.
    image_buffer = BytesIO()
    preview_image = image.copy().convert("RGB")
    preview_image.thumbnail((1200, 900))
    preview_image.save(image_buffer, format="JPEG", quality=92)
    image_buffer.seek(0)

    preview = PDFImage(image_buffer)
    preview._restrictSize(78 * mm, 54 * mm)

    status_bg = "#DCFCE7" if accepted else "#FEE2E2"
    status_fg = "#166534" if accepted else "#991B1B"

    summary = Table(
        [
            [
                Paragraph(
                    f"<font color='{status_fg}'><b>{status}</b></font>",
                    ParagraphStyle(
                        "StatusBadge",
                        parent=value_style,
                        fontSize=8,
                        leading=10,
                        alignment=TA_CENTER,
                    ),
                )
            ],
            [Paragraph(decision, ParagraphStyle(
                "DecisionName",
                parent=value_style,
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#0F172A"),
                spaceBefore=2 * mm,
                spaceAfter=1 * mm,
            ))],
            [Paragraph(
                f"Country code: <b>{country_code}</b>",
                value_style,
            )],
            [Spacer(1, 1.5 * mm)],
            [Table(
                [
                    [
                        Paragraph("CONFIDENCE", metric_label_style),
                        Paragraph("THRESHOLD", metric_label_style),
                    ],
                    [
                        Paragraph(f"{confidence:.2%}", metric_value_style),
                        Paragraph(f"{threshold:.2%}", metric_value_style),
                    ],
                ],
                colWidths=[38 * mm, 38 * mm],
                style=TableStyle([
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E8F0")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]),
            )],
        ],
        colWidths=[78 * mm],
    )
    summary.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(status_bg)),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    hero = Table(
        [[preview, summary]],
        colWidths=[82 * mm, 78 * mm],
        hAlign="LEFT",
    )
    hero.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (0, 0), 0.6, colors.HexColor("#E2E8F0")),
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F8FAFC")),
            ("LEFTPADDING", (0, 0), (0, 0), 5),
            ("RIGHTPADDING", (0, 0), (0, 0), 5),
            ("TOPPADDING", (0, 0), (0, 0), 5),
            ("BOTTOMPADDING", (0, 0), (0, 0), 5),
            ("LEFTPADDING", (1, 0), (1, 0), 4),
            ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ])
    )
    story.append(hero)

    # Ranking.
    story.append(Paragraph("Top candidates", section_style))
    top_rows = [[
        Paragraph("<b>Rank</b>", small_style),
        Paragraph("<b>Country</b>", small_style),
        Paragraph("<b>Code</b>", small_style),
        Paragraph("<b>Confidence</b>", small_style),
    ]]

    for index, candidate in enumerate(report.get("top_candidates", []), start=1):
        top_rows.append([
            str(index),
            _pdf_text(candidate.get("country")),
            _pdf_text(candidate.get("code")).upper(),
            f"{float(candidate.get('confidence', 0.0)):.2%}",
        ])

    top_table = Table(
        top_rows,
        colWidths=[14 * mm, 94 * mm, 20 * mm, 32 * mm],
        repeatRows=1,
    )
    ranking_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index in range(1, len(top_rows)):
        if row_index % 2 == 0:
            ranking_styles.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8FAFC"))
            )
    top_table.setStyle(TableStyle(ranking_styles))
    story.append(top_table)

    profile = report.get("country_profile")
    if isinstance(profile, dict):
        story.append(Paragraph("Country profile", section_style))

        population_value = (
            f"{int(profile['population']):,}"
            if profile.get("population") is not None
            else "Not available"
        )
        area_value = (
            f"{float(profile['area_km2']):,.0f} km²"
            if profile.get("area_km2") is not None
            else "Not available"
        )
        coordinates = (
            f"{float(profile['latitude']):.3f}, {float(profile['longitude']):.3f}"
            if (
                profile.get("latitude") is not None
                and profile.get("longitude") is not None
            )
            else "Not available"
        )

        population_ref = population_value
        if profile.get("population_year"):
            population_ref += f" ({profile.get('population_year')})"

        identity_card = _profile_card(
            "Identity & geography",
            [
                ("Official name", profile.get("name")),
                ("Capital", profile.get("capital")),
                ("Continent", profile.get("continent")),
                ("Population", population_ref),
                ("Population source", profile.get("population_source")),
                ("Area", area_value),
                ("Coordinates", coordinates),
            ],
            label_style,
            value_style,
        )

        symbols_card = _profile_card(
            "National symbols",
            [
                ("National Day", profile.get("national_day")),
                ("Independence", profile.get("independence_day")),
                ("National motto", profile.get("national_motto")),
                ("National anthem", profile.get("national_anthem")),
            ],
            label_style,
            value_style,
        )

        government_card = _profile_card(
            "Government",
            [
                ("Government form", profile.get("government_form")),
                ("Head of State", profile.get("head_of_state")),
                ("Head of State office", profile.get("head_of_state_office")),
                ("Head of Government", profile.get("head_of_government")),
                ("Head of Government office", profile.get("head_of_government_office")),
            ],
            label_style,
            value_style,
        )

        practical_card = _profile_card(
            "Practical information",
            [
                ("Currency", profile.get("currency")),
                ("Official language(s)", profile.get("official_languages")),
                ("Calling code", profile.get("calling_code")),
                ("Internet domain", profile.get("internet_domain")),
                ("Driving side", profile.get("driving_side")),
            ],
            label_style,
            value_style,
        )

        story.extend([
            identity_card,
            symbols_card,
            government_card,
            practical_card,
        ])

        overview = _pdf_text(profile.get("overview"))
        if overview != "Not available":
            overview_box = Table(
                [[Paragraph(overview, body_style)]],
                colWidths=[156 * mm],
            )
            overview_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ])
            )
            story.extend([
                Paragraph("Country overview", section_style),
                overview_box,
            ])

    story.extend([
        Spacer(1, 5 * mm),
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=1 * mm,
            spaceAfter=2 * mm,
        ),
        Paragraph(
            "<b>Data note.</b> Profile values and the encyclopedic overview "
            "may use different reference years or geographic/statistical "
            "definitions. Population year and source are shown explicitly.",
            small_style,
        ),
        Spacer(1, 1.5 * mm),
        Paragraph(
            "<b>Sources:</b> Wikidata, World Bank and Wikipedia where available.",
            small_style,
        ),
    ])

    document.build(
        story,
        onFirstPage=draw_pdf_watermark,
        onLaterPages=draw_pdf_watermark,
    )
    buffer.seek(0)
    return buffer.getvalue()


MODEL_PATH = ROOT_DIR / "artifacts/models/worldwide_mobilenet_v3_small.pt"
CONFIG_PATH = ROOT_DIR / "configs/deployment.yaml"

BANNER_FLAG_CODES = [
    "us",
    "ca",
    "mx",
    "gt",
    "bz",
    "sv",
    "hn",
    "ni",
    "cr",
    "pa",
    "cu",
    "do",
    "ht",
    "jm",
    "bs",
    "bb",
    "tt",
    "gd",
    "dm",
    "lc",
    "br",
    "ar",
    "cl",
    "pe",
    "co",
    "ve",
    "ec",
    "bo",
    "py",
    "uy",
    "gb",
    "ie",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "ch",
    "at",
    "be",
    "nl",
    "lu",
    "dk",
    "se",
    "no",
    "fi",
    "is",
    "gr",
    "pl",
    "cz",
    "sk",
    "hu",
    "ro",
    "bg",
    "hr",
    "si",
    "ba",
    "rs",
    "me",
    "mk",
    "ma",
    "dz",
    "tn",
    "ly",
    "eg",
    "ng",
    "gh",
    "ci",
    "sn",
    "ke",
    "za",
    "et",
    "tz",
    "ug",
    "rw",
    "cm",
    "ao",
    "zm",
    "zw",
    "mz",
    "sa",
    "ae",
    "qa",
    "kw",
    "om",
    "jo",
    "il",
    "lb",
    "iq",
    "ir",
    "pk",
    "in",
    "bd",
    "lk",
    "cn",
    "jp",
    "kr",
    "id",
    "th",
    "vn",
]


st.set_page_config(
    page_title="Flag Intelligence",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_model():
    return load_inference_bundle(MODEL_PATH, device="cpu")


@st.cache_data(show_spinner=False)
def get_deployment_threshold() -> float:
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        configured = config.get("open_set", {}).get("deployment_threshold")
        if configured is not None:
            return float(configured)

    return float(get_model().unknown_threshold)


@st.cache_data(ttl=3600, show_spinner=False)
def get_country_profile(country_code: str):
    return fetch_country_profile(country_code)


def build_flag_banner_html() -> str:
    flags_html = "".join(
        f'<img src="https://flagcdn.com/w80/{code}.png" loading="lazy" alt="{code} flag">'
        for code in BANNER_FLAG_CODES
    )
    return f"""
    <div class="shell">
        <div class="flag-banner">
            <div class="flag-strip">
                {flags_html}
            </div>
        </div>
        <div class="intro">
            <div class="app-title">Flag Intelligence</div>
            <div class="app-subtitle">Select an image and run recognition.</div>
        </div>
    </div>
    """


st.markdown(
    """
    <style>
        :root {
            --text: #111827;
            --muted: #667085;
            --line: #e5e7eb;
            --blue: #2563eb;
            --blue-dark: #1d4ed8;
            --surface: #ffffff;
            --bg: #f5f7fb;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 900px;
            padding-top: 2.6rem;
            padding-bottom: 2rem;
        }

        .shell {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 20px;
            background: #fff;
            box-shadow: 0 18px 50px rgba(15,23,42,.08);
            margin-bottom: 1rem;
        }

        .flag-banner {
            position: relative;
            height: auto;
            overflow: hidden;
            background: #ffffff;
            border-bottom: 1px solid #eef2f7;
        }

        .flag-strip {
            display: grid;
            grid-template-columns: repeat(20, 1fr);
            grid-auto-rows: 28px;
            gap: 3px;
            padding: 10px 10px 8px;
            align-content: start;
            background: #ffffff;
        }

        .flag-strip img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 2px;
            display: block;
        }

        .intro {
            text-align: center;
            padding: 1rem 1.5rem 1.35rem;
        }

        .app-title {
            font-size: 1.9rem;
            font-weight: 900;
            letter-spacing: -.04em;
            color: var(--text);
            margin-bottom: .3rem;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: .94rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            background: var(--surface);
            box-shadow: 0 8px 24px rgba(16,24,40,.04);
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 118px;
            border: 1.5px dashed #c2cad6;
            border-radius: 14px;
            background: #fafbfc;
        }

        .preview-placeholder {
            min-height: 118px;
            border: 1.5px dashed #c2cad6;
            border-radius: 14px;
            background: #fafbfc;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #98a2b3;
            font-size: .78rem;
            font-weight: 700;
            text-align: center;
            padding: .8rem;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--blue);
            background: #f8fbff;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            border-radius: 10px;
            font-weight: 750;
        }

        .stButton > button[kind="primary"] {
            min-height: 48px;
            border: 0;
            border-radius: 12px;
            background: var(--blue);
            font-weight: 850;
            font-size: .95rem;
            box-shadow: 0 8px 18px rgba(37,99,235,.18);
        }

        .stButton > button[kind="primary"]:hover {
            background: var(--blue-dark);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: .85rem;
            background: #fafafa;
        }

        .result-name {
            font-size: 1.65rem;
            font-weight: 850;
            letter-spacing: -.03em;
            margin-bottom: .15rem;
        }

        .result-code {
            color: var(--muted);
            font-size: .82rem;
            margin-bottom: 1rem;
        }

        .decision-ok {
            padding: .75rem .85rem;
            border-radius: 10px;
            background: #ecfdf3;
            border: 1px solid #abefc6;
            color: #067647;
            font-weight: 750;
            margin: .7rem 0;
        }

        .decision-no {
            padding: .75rem .85rem;
            border-radius: 10px;
            background: #fffaeb;
            border: 1px solid #fedf89;
            color: #b54708;
            font-weight: 750;
            margin: .7rem 0;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1.5rem;
            }
            .flag-banner {
                height: auto;
            }
            .flag-strip {
                grid-template-columns: repeat(10, 1fr);
                grid-auto-rows: 22px;
                gap: 2px;
                padding: 8px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if not MODEL_PATH.is_file():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()


bundle = get_model()
deployment_threshold = get_deployment_threshold()

st.markdown(
    build_flag_banner_html(),
    unsafe_allow_html=True,
)

with st.container(border=True):
    upload_col, preview_col = st.columns([1.35, 0.65], gap="medium")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Select image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    image = None

    with preview_col:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        else:
            st.markdown(
                '<div class="preview-placeholder">Image preview</div>',
                unsafe_allow_html=True,
            )

    process = st.button(
        "Process image",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )


@st.dialog("Recognition result", width="large")
def show_result(image: Image.Image):
    with st.spinner("Processing image..."):
        prediction = predict_image(
            image,
            bundle,
            top_k=20,
        )

    grouped_candidates = merge_visually_identical_candidates(
        prediction.top5
    )
    decision_code, decision_confidence = grouped_candidates[0]
    display_candidates = grouped_candidates[:5]

    accepted = decision_confidence >= deployment_threshold
    country = display_country_name(decision_code)

    preview_col, result_col = st.columns([.9, 1.1], gap="large")

    with preview_col:
        st.image(image, use_container_width=True)

    with result_col:
        st.markdown(
            f'<div class="result-name">{country if accepted else "Unknown"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-code">Top candidate · {country} ({decision_code.upper()})</div>',
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Confidence", f"{decision_confidence:.1%}")
        with m2:
            st.metric("Threshold", f"{deployment_threshold:.1%}")

        if accepted:
            st.markdown(
                '<div class="decision-ok">Accepted prediction</div>',
                unsafe_allow_html=True,
            )
            if decision_code in VISUAL_EQUIVALENCE_GROUPS:
                equivalents = ", ".join(
                    code.upper()
                    for code in sorted(
                        VISUAL_EQUIVALENCE_GROUPS[decision_code]
                    )
                )
                st.caption(
                    "Equivalent official flag labels merged: "
                    + equivalents
                )
        else:
            st.markdown(
                '<div class="decision-no">Prediction rejected by confidence threshold</div>',
                unsafe_allow_html=True,
            )

    ranking = pd.DataFrame(
        [
            {
                "Rank": rank,
                "Country": display_country_name(code),
                "Code": code.upper(),
                "Confidence": confidence,
            }
            for rank, (code, confidence) in enumerate(
                display_candidates,
                start=1,
            )
        ]
    )

    with st.expander("Top candidates", expanded=False):
        st.dataframe(
            ranking.style.format({"Confidence": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )

    if accepted:
        try:
            profile = get_country_profile(decision_code)

            st.divider()
            st.markdown("### Country profile")

            overview_tab, government_tab, geography_tab = st.tabs(
                ["Overview", "Government", "Geography"]
            )

            with overview_tab:
                o1, o2, o3, o4 = st.columns(4)
                with o1:
                    st.metric("Capital", profile.capital)
                with o2:
                    population = (
                        f"{profile.population.value:,}"
                        if profile.population.value is not None
                        else "Not available"
                    )
                    st.metric("Population", population)
                    if profile.population.value is not None:
                        population_note = profile.population.source
                        if profile.population.year:
                            population_note += f" · {profile.population.year}"
                        st.caption(population_note)
                with o3:
                    st.metric("Currency", profile.currency)
                with o4:
                    area = (
                        f"{profile.area_km2:,.0f} km²"
                        if profile.area_km2 is not None
                        else "Not available"
                    )
                    st.metric("Area", area)

                d1, d2 = st.columns(2)
                with d1:
                    st.metric("National Day", profile.national_day)
                with d2:
                    st.metric("Independence", profile.independence_day)

                info_left, info_right = st.columns(2, gap="large")
                with info_left:
                    st.markdown("**Official language(s)**")
                    st.write(profile.official_languages)

                    st.markdown("**Continent**")
                    st.write(profile.continent)

                    st.markdown("**National motto**")
                    st.write(profile.national_motto)

                with info_right:
                    st.markdown("**National anthem**")
                    st.write(profile.national_anthem)

                    st.markdown("**Country overview**")
                    st.write(profile.overview)

            with government_tab:
                g1, g2 = st.columns(2, gap="large")

                with g1:
                    st.markdown("**Government form**")
                    st.write(profile.government_form)

                    st.markdown("**Head of State**")
                    st.write(profile.head_of_state)
                    if profile.head_of_state_office != "Not available":
                        st.caption(profile.head_of_state_office)

                    st.markdown("**Head of Government**")
                    st.write(profile.head_of_government)
                    if profile.head_of_government_office != "Not available":
                        st.caption(profile.head_of_government_office)

                with g2:
                    st.markdown("**Calling code**")
                    st.write(profile.calling_code)

                    st.markdown("**Internet domain**")
                    st.write(profile.internet_domain)

                    st.markdown("**Driving side**")
                    st.write(profile.driving_side)

            with geography_tab:
                geo_left, geo_right = st.columns([1.35, 0.65], gap="large")

                with geo_left:
                    if (
                        profile.latitude is not None
                        and profile.longitude is not None
                    ):
                        st.map(
                            pd.DataFrame(
                                [{
                                    "lat": profile.latitude,
                                    "lon": profile.longitude,
                                }]
                            ),
                            latitude="lat",
                            longitude="lon",
                            zoom=3,
                            use_container_width=True,
                        )
                    else:
                        st.info("Geographic coordinates are not available.")

                with geo_right:
                    st.markdown("**Continent**")
                    st.write(profile.continent)

                    st.markdown("**Area**")
                    st.write(
                        f"{profile.area_km2:,.0f} km²"
                        if profile.area_km2 is not None
                        else "Not available"
                    )

                    st.markdown("**Coordinates**")
                    if (
                        profile.latitude is not None
                        and profile.longitude is not None
                    ):
                        st.write(
                            f"{profile.latitude:.3f}, "
                            f"{profile.longitude:.3f}"
                        )
                    else:
                        st.write("Not available")

            st.caption(
                "Country metadata: Wikidata · Population: World Bank · "
                "Overview: Wikipedia"
            )

        except (requests.RequestException, LookupError, ValueError) as error:
            st.warning("Country information could not be loaded.")
            st.caption(str(error))

    report = {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "accepted": accepted,
        "decision": country if accepted else "Unknown",
        "top_candidate": country,
        "country_code": decision_code,
        "confidence": decision_confidence,
        "deployment_threshold": deployment_threshold,
        "visual_equivalence_applied": (
            decision_code
            in VISUAL_EQUIVALENCE_GROUPS
        ),
        "equivalent_flag_codes": sorted(
            VISUAL_EQUIVALENCE_GROUPS.get(
                decision_code,
                {decision_code},
            )
        ),
        "top_candidates": [
            {
                "country": display_country_name(code),
                "code": code,
                "confidence": confidence,
            }
            for code, confidence in display_candidates
        ],
    }

    if accepted:
        try:
            profile = get_country_profile(decision_code)
            report["country_profile"] = {
                "name": profile.name,
                "capital": profile.capital,
                "population": profile.population.value,
                "population_year": profile.population.year,
                "population_source": profile.population.source,
                "currency": profile.currency,
                "official_languages": profile.official_languages,
                "continent": profile.continent,
                "area_km2": profile.area_km2,
                "overview": profile.overview,
                "national_day": profile.national_day,
                "independence_day": profile.independence_day,
                "national_motto": profile.national_motto,
                "national_anthem": profile.national_anthem,
                "government_form": profile.government_form,
                "head_of_state": profile.head_of_state,
                "head_of_state_office": profile.head_of_state_office,
                "head_of_government": profile.head_of_government,
                "head_of_government_office": profile.head_of_government_office,
                "calling_code": profile.calling_code,
                "internet_domain": profile.internet_domain,
                "driving_side": profile.driving_side,
                "latitude": profile.latitude,
                "longitude": profile.longitude,
            }
        except (requests.RequestException, LookupError, ValueError):
            pass

    json_col, pdf_col = st.columns(2, gap="small")

    with json_col:
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name="flag_recognition_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with pdf_col:
        st.download_button(
            "Download PDF",
            data=build_pdf_report(report, image),
            file_name="flag_recognition_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if process and image is not None:
    show_result(image)
