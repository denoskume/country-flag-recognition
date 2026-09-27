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


PDF_LOGO_BASE64 = "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"

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


def draw_pdf_watermark(canvas, document) -> None:
    """Draw the Flag Intelligence report template on every PDF page."""
    canvas.saveState()

    page_width, page_height = A4
    left = 18 * mm
    right = page_width - 18 * mm

    # ------------------------------------------------------------------
    # Flag Intelligence logo - vector version matching the supplied template
    # ------------------------------------------------------------------
    logo_size = 22 * mm
    logo_x = left
    logo_y = page_height - 36 * mm
    cx = logo_x + logo_size / 2
    cy = logo_y + logo_size / 2
    radius = logo_size / 2

    canvas.setFillColor(colors.HexColor("#F40016"))
    canvas.circle(cx, cy, radius, fill=1, stroke=0)

    # Flag symbol.
    canvas.setStrokeColor(colors.white)
    canvas.setFillColor(colors.white)
    canvas.setLineCap(1)
    canvas.setLineWidth(1.05 * mm)

    pole_x = logo_x + 7.2 * mm
    canvas.line(
        pole_x,
        logo_y + 10.2 * mm,
        pole_x,
        logo_y + 18.0 * mm,
    )

    band_x0 = logo_x + 8.6 * mm
    band_x1 = logo_x + 15.7 * mm
    for offset in (17.0, 14.4, 11.8):
        y0 = logo_y + offset * mm
        path = canvas.beginPath()
        path.moveTo(band_x0, y0)
        path.curveTo(
            logo_x + 10.7 * mm,
            y0 + 0.9 * mm,
            logo_x + 12.8 * mm,
            y0 - 0.9 * mm,
            band_x1,
            y0 + 0.1 * mm,
        )
        canvas.drawPath(path, stroke=1, fill=0)

    # Wordmark inside the red circle.
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8.6)
    canvas.drawCentredString(
        cx,
        logo_y + 5.9 * mm,
        "FLAG",
    )
    canvas.setFont("Helvetica-Bold", 4.1)
    canvas.drawCentredString(
        cx,
        logo_y + 3.6 * mm,
        "INTELLIGENCE",
    )

    # ------------------------------------------------------------------
    # Contact block
    # ------------------------------------------------------------------
    contact_x = left + 3.5 * mm
    contact_y = logo_y - 5.2 * mm

    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont("Helvetica", 8.2)
    canvas.drawString(
        contact_x,
        contact_y,
        "Contact :",
    )

    canvas.setFont("Helvetica", 7.8)
    canvas.drawString(
        contact_x,
        contact_y - 5.0 * mm,
        "Phone  +33 (0)6 62 91 94 68",
    )
    canvas.drawString(
        contact_x,
        contact_y - 9.6 * mm,
        "Email  denoskume@yahoo.com",
    )

    # ------------------------------------------------------------------
    # Report title - centered like the supplied template
    # ------------------------------------------------------------------
    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont("Helvetica", 17)
    canvas.drawCentredString(
        page_width / 2,
        page_height - 66 * mm,
        "Report",
    )

    # ------------------------------------------------------------------
    # Subtle signature watermark requested earlier
    # ------------------------------------------------------------------
    canvas.setFillAlpha(0.022)
    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont("Helvetica-Oblique", 24)
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

    # ------------------------------------------------------------------
    # Footer - centered copyright from the supplied template
    # ------------------------------------------------------------------
    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont("Helvetica", 8.2)
    canvas.drawCentredString(
        page_width / 2,
        15 * mm,
        "© 2026 Flag Intelligence, all right reserved.",
    )

    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.setFont("Helvetica", 6.5)
    canvas.drawRightString(
        right,
        9 * mm,
        f"Page {document.page}",
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
        topMargin=76 * mm,
        bottomMargin=24 * mm,
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

    # Compact report metadata under the template title.
    report_meta = Table(
        [[
            Paragraph(
                f"<b>Country Flag Recognition Report</b>",
                ParagraphStyle(
                    "ReportIdentity",
                    parent=title_style,
                    fontSize=12.5,
                    leading=15,
                    alignment=0,
                ),
            ),
            Paragraph(
                f"<b>Date</b><br/>{generated_at}",
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
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.extend([
        report_meta,
        Spacer(1, 4 * mm),
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
