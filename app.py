"""Focused Streamlit interface for worldwide flag recognition."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from time import strftime

import pandas as pd
import pydeck as pdk
from PIL import Image, ImageDraw
import requests
import streamlit as st
import yaml
from playwright.sync_api import sync_playwright
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
    PageBreak,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PDF_LOGO_BASE64 = "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"

PDF_MARGIN_MM = 14.0
PDF_FRAME_PADDING_PT = 6.0
PDF_FRAME_PADDING_MM = PDF_FRAME_PADDING_PT * 25.4 / 72.0
PDF_CONTENT_WIDTH_MM = (
    210.0
    - (2 * PDF_MARGIN_MM)
    - (2 * PDF_FRAME_PADDING_MM)
)
PDF_CONTENT_LEFT_MM = PDF_MARGIN_MM + PDF_FRAME_PADDING_MM

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
import flag_recognition.country_info as country_info_module

# Streamlit can rerun app.py without restarting the Python interpreter.
# Reload country data logic so deployments never keep stale historical rules.
country_info_module = importlib.reload(country_info_module)

fetch_country_profile = country_info_module.fetch_country_profile
fetch_emergency_numbers = country_info_module.fetch_emergency_numbers
fetch_emergency_numbers_fallback = (
    country_info_module.fetch_emergency_numbers_fallback
)
format_emergency_numbers = country_info_module.format_emergency_numbers
from flag_recognition.inference import (
    load_inference_bundle,
    predict_image,
    predict_robust,
)
from flag_recognition.taxonomy import country_code_from_text, country_name_from_code
from flag_recognition.country_intelligence import (
    build_from_legacy_profile,
    section_completion,
    validate_country_intelligence,
)
from flag_recognition.country_knowledge import (
    canonical_overview_text,
    enrich_from_encyclopedia,
)
from flag_recognition.flag_knowledge import enrich_flag_profile
from flag_recognition.learning import answer_country_question
from flag_recognition.report_manifest import (
    build_report_manifest,
    missing_required_sections,
)


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
    """Draw the final Flag Intelligence PDF template on every page."""
    canvas.saveState()

    page_width, page_height = A4
    left = PDF_CONTENT_LEFT_MM * mm
    right = page_width - (PDF_CONTENT_LEFT_MM * mm)

    # ------------------------------------------------------------------
    # Final template logo - top left
    # ------------------------------------------------------------------
    logo_size = 24 * mm
    logo_x = left
    logo_y = page_height - 38 * mm
    cx = logo_x + logo_size / 2
    cy = logo_y + logo_size / 2
    radius = logo_size / 2

    canvas.setFillColor(colors.HexColor("#F40016"))
    canvas.circle(cx, cy, radius, fill=1, stroke=0)

    canvas.setStrokeColor(colors.white)
    canvas.setFillColor(colors.white)
    canvas.setLineCap(1)
    canvas.setLineWidth(1.15 * mm)

    pole_x = logo_x + 7.7 * mm
    canvas.line(
        pole_x,
        logo_y + 11.2 * mm,
        pole_x,
        logo_y + 19.7 * mm,
    )

    band_x0 = logo_x + 9.2 * mm
    band_x1 = logo_x + 17.2 * mm
    for offset in (18.7, 15.9, 13.1):
        y0 = logo_y + offset * mm
        path = canvas.beginPath()
        path.moveTo(band_x0, y0)
        path.curveTo(
            logo_x + 11.5 * mm,
            y0 + 1.0 * mm,
            logo_x + 13.8 * mm,
            y0 - 1.0 * mm,
            band_x1,
            y0 + 0.1 * mm,
        )
        canvas.drawPath(path, stroke=1, fill=0)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 9.6)
    canvas.drawCentredString(
        cx,
        logo_y + 6.4 * mm,
        "FLAG",
    )
    canvas.setFont("Helvetica-Bold", 4.5)
    canvas.drawCentredString(
        cx,
        logo_y + 3.8 * mm,
        "INTELLIGENCE",
    )

    # ------------------------------------------------------------------
    # Contact is shown only on the first page; later pages use a compact
    # document marker so educational content gets more visual space.
    # ------------------------------------------------------------------
    contact_x = right - 68 * mm
    contact_y = page_height - 23 * mm

    if getattr(document, "page", 1) == 1:
        canvas.setFillColor(colors.HexColor("#111111"))
        canvas.setFont("Helvetica", 9.3)
        canvas.drawString(contact_x, contact_y, "Contact")

        canvas.setFont("Helvetica", 8.6)
        canvas.drawString(
            contact_x,
            contact_y - 5.0 * mm,
            "+33 (0)6 62 91 94 68",
        )
        canvas.drawString(
            contact_x,
            contact_y - 10.2 * mm,
            "denoskume@yahoo.com",
        )
    else:
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.setFont("Helvetica-Bold", 8.2)
        canvas.drawRightString(
            right,
            contact_y,
            "FLAG INTELLIGENCE · COUNTRY INTELLIGENCE",
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont("Helvetica", 6.8)
    canvas.drawCentredString(
        (left + right) / 2,
        18 * mm,
        "Sources: World Bank · Wikidata · REST Countries · Wikipedia · EmergencyNumberAPI",
    )
    canvas.setFont("Helvetica", 7.2)
    canvas.drawCentredString(
        (left + right) / 2,
        13.5 * mm,
        f"© 2026 Flag Intelligence. All rights reserved. · Page {document.page}",
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


def _report_filename_country(country_name: str) -> str:
    """Return a filesystem-safe country slug for report downloads."""
    normalized = (
        str(country_name)
        .strip()
        .lower()
        .replace("&", "and")
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "country"


def _build_geography_deck(
    latitude: float,
    longitude: float,
    area_km2: float | None = None,
) -> pdk.Deck:
    """Build the single CARTO map used by both the app and the PDF."""
    if area_km2 is None:
        zoom = 4.2
    elif area_km2 < 2_000:
        zoom = 7.0
    elif area_km2 < 20_000:
        zoom = 6.2
    elif area_km2 < 100_000:
        zoom = 5.3
    elif area_km2 < 300_000:
        zoom = 4.6
    elif area_km2 < 800_000:
        zoom = 4.0
    elif area_km2 < 2_000_000:
        zoom = 3.6
    else:
        zoom = 3.2

    data = pd.DataFrame(
        [{
            "lat": float(latitude),
            "lon": float(longitude),
        }]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_fill_color=[200, 30, 0, 160],
        get_line_color=[200, 30, 0, 255],
        get_radius=55000,
        radius_min_pixels=5,
        radius_max_pixels=8,
        stroked=True,
        filled=True,
        pickable=False,
    )

    view_state = pdk.ViewState(
        latitude=float(latitude),
        longitude=float(longitude),
        zoom=zoom,
        pitch=0,
        bearing=0,
    )

    return pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=[layer],
    )


def _render_geography_deck_png(
    latitude: float | None,
    longitude: float | None,
    area_km2: float | None = None,
) -> bytes | None:
    """Render the exact app CARTO/PyDeck map to PNG for the PDF."""
    if latitude is None or longitude is None:
        return None

    chromium = (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )
    if chromium is None:
        return None

    deck = _build_geography_deck(
        float(latitude),
        float(longitude),
        area_km2,
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            html_path = tmp_path / "geography_map.html"
            png_path = tmp_path / "geography_map.png"

            deck.to_html(
                str(html_path),
                open_browser=False,
                notebook_display=False,
            )

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=chromium,
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                page = browser.new_page(
                    viewport={
                        "width": 1100,
                        "height": 650,
                    }
                )
                page.goto(
                    html_path.as_uri(),
                    wait_until="domcontentloaded",
                )
                page.wait_for_timeout(4500)

                map_element = page.locator(".deckgl-wrapper").first
                if map_element.count() == 0:
                    map_element = page.locator("canvas").first

                map_element.screenshot(
                    path=str(png_path),
                )
                browser.close()

            return png_path.read_bytes()

    except Exception:
        return None


def _build_pdf_location_map(
    latitude: float | None,
    longitude: float | None,
    area_km2: float | None = None,
    country_name: str = "Country",
    capital: str = "Not available",
    country_code: str = "",
) -> PDFImage | None:
    """Convert the exact app geography map into a ReportLab image."""
    map_bytes = _render_geography_deck_png(
        latitude,
        longitude,
        area_km2,
    )
    if map_bytes is None:
        return None

    pdf_map = PDFImage(BytesIO(map_bytes))
    pdf_map.drawWidth = (PDF_CONTENT_WIDTH_MM - 4.0) * mm
    pdf_map.drawHeight = 82 * mm
    return pdf_map


def build_pdf_report(
    report: dict[str, object],
    image: Image.Image | None,
) -> bytes:
    """Build a compact institutional country knowledge report."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=PDF_MARGIN_MM * mm,
        leftMargin=PDF_MARGIN_MM * mm,
        topMargin=44 * mm,
        bottomMargin=18 * mm,
        title="Flag Intelligence - Country Knowledge Report",
        author="Denos Kume",
        subject="Country knowledge report generated from flag recognition",
    )

    REPORT_WIDTH_MM = PDF_CONTENT_WIDTH_MM

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CountryReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111111"),
        spaceAfter=2.5 * mm,
    )

    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.5,
        textColor=colors.HexColor("#444444"),
    )

    section_title_style = ParagraphStyle(
        "BoxSectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=12,
        textColor=colors.HexColor("#111111"),
        spaceAfter=0,
    )

    label_style = ParagraphStyle(
        "CompactLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9.2,
        textColor=colors.HexColor("#333333"),
    )

    value_style = ParagraphStyle(
        "CompactValue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=9.4,
        textColor=colors.HexColor("#222222"),
    )

    body_style = ParagraphStyle(
        "CompactBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#222222"),
    )

    small_style = ParagraphStyle(
        "CompactSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.7,
        leading=8.2,
        textColor=colors.HexColor("#555555"),
    )

    def clean(value: object) -> str:
        text = _pdf_text(value)
        return text if text not in {"", "None"} else "Not available"

    def compact_list(value: object, limit: int = 12) -> str:
        text = clean(value)
        if text == "Not available":
            return text

        items = [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

        if len(items) <= limit:
            return ", ".join(items)

        shown = ", ".join(items[:limit])
        return f"{shown} (+{len(items) - limit} more)"

    def info_grid(
        rows: list[tuple[str, object]],
        *,
        two_pairs: bool = True,
        width_mm: float = REPORT_WIDTH_MM,
    ) -> Table:
        filtered = [
            (label, clean(value))
            for label, value in rows
            if clean(value) != "Not available"
        ]

        # ReportLab cannot build an empty Table. Keep the section valid
        # even when an external source provides no usable values.
        if not filtered:
            filtered = [
                ("Information", "Not available"),
            ]

        data: list[list[object]] = []

        if two_pairs:
            for index in range(0, len(filtered), 2):
                row = filtered[index:index + 2]
                cells: list[object] = []

                for label, value in row:
                    cells.extend([
                        Paragraph(label, label_style),
                        Paragraph(value, value_style),
                    ])

                if len(row) == 1:
                    cells.extend(["", ""])

                data.append(cells)

            label_width_mm = 28.0
            value_width_mm = (
                width_mm - (2 * label_width_mm)
            ) / 2

            table = Table(
                data,
                colWidths=[
                    label_width_mm * mm,
                    value_width_mm * mm,
                    label_width_mm * mm,
                    value_width_mm * mm,
                ],
                hAlign="LEFT",
            )
            label_columns = [(0, 0), (2, 0)]
        else:
            data = [
                [
                    Paragraph(label, label_style),
                    Paragraph(value, value_style),
                ]
                for label, value in filtered
            ]
            label_width_mm = 43.0
            table = Table(
                data,
                colWidths=[
                    label_width_mm * mm,
                    (width_mm - label_width_mm) * mm,
                ],
                hAlign="LEFT",
            )
            label_columns = [(0, 0)]

        style = [
            ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#666666")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D6D6D6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]

        if two_pairs:
            style.extend([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F4F4")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F4F4F4")),
            ])
        else:
            style.append(
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F4F4"))
            )

        table.setStyle(TableStyle(style))
        return table

    def section_title_bar(title: str) -> Table:
        """Standalone title bar that can precede page-splittable content."""
        title_bar = Table(
            [[Paragraph(title, section_title_style)]],
            colWidths=[REPORT_WIDTH_MM * mm],
            hAlign="LEFT",
        )
        title_bar.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#111111")),
                ("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#111111")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ])
        )
        title_bar.keepWithNext = True
        return title_bar

    def split_section(
        title: str,
        content: object,
    ) -> list[object]:
        """Return independent flowables so long content may split across pages."""
        return [
            section_title_bar(title),
            content,
            Spacer(1, 3 * mm),
        ]

    def section_box(
        title: str,
        content: object,
    ) -> Table:
        """Compact non-splittable box for short sections only."""
        title_bar = section_title_bar(title)

        wrapper = Table(
            [[title_bar], [content]],
            colWidths=[REPORT_WIDTH_MM * mm],
            hAlign="LEFT",
        )
        wrapper.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#111111")),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ])
        )
        return wrapper

    generated_at = clean(report.get("generated_at"))
    decision = clean(report.get("decision"))
    country_code = clean(report.get("country_code")).upper()

    def optional_float(value: object) -> float | None:
        """Preserve missing recognition metrics for text-selected countries."""
        if value in (None, "", "Not available"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    confidence = optional_float(report.get("confidence"))
    threshold = optional_float(report.get("deployment_threshold"))
    decision_margin = optional_float(report.get("decision_margin"))
    decision_reason = clean(report.get("decision_reason"))
    accepted = bool(report.get("accepted"))
    status = "Accepted" if accepted else "Rejected"

    if " " in generated_at:
        report_date, report_time = generated_at.split(" ", 1)
    else:
        report_date, report_time = generated_at, "Not available"

    story: list[object] = []

    # Compact document heading inspired by the institutional reference.
    heading = Table(
        [[
            Paragraph(
                f"<b>Generated:</b> {report_date} · {report_time}",
                meta_style,
            ),
            Paragraph(
                f"Report ({decision})",
                title_style,
            ),
            Paragraph(
                f"<b>Country code:</b> {country_code}",
                ParagraphStyle(
                    "ReportStatusMeta",
                    parent=meta_style,
                    alignment=2,
                ),
            ),
        ]],
        colWidths=[
            52 * mm,
            (REPORT_WIDTH_MM - 104.0) * mm,
            52 * mm,
        ],
        hAlign="LEFT",
    )
    heading.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
        ])
    )
    story.append(heading)

    profile = report.get("country_profile")

    if accepted and isinstance(profile, dict):
        population_value = (
            f"{int(profile['population']):,}"
            if profile.get("population") is not None
            else "Not available"
        )
        if (
            profile.get("population_year")
            and population_value != "Not available"
        ):
            population_value += f" ({profile.get('population_year')})"

        area_value = (
            f"{float(profile['area_km2']):,.0f} km²"
            if profile.get("area_km2") is not None
            else "Not available"
        )

        coordinates = (
            f"{float(profile['latitude']):.3f}, "
            f"{float(profile['longitude']):.3f}"
            if (
                profile.get("latitude") is not None
                and profile.get("longitude") is not None
            )
            else "Not available"
        )

        gdp_value = (
            f"$ {float(profile['gdp_usd']):,.0f}"
            if profile.get("gdp_usd") is not None
            else "Not available"
        )
        if (
            profile.get("gdp_year")
            and gdp_value != "Not available"
        ):
            gdp_value += f" ({profile.get('gdp_year')})"

        intelligence = report.get("country_intelligence_v2")
        completion = report.get("country_intelligence_completion")
        report_manifest = report.get("official_report_manifest")
        if not isinstance(report_manifest, dict):
            report_manifest = build_report_manifest(
                intelligence if isinstance(intelligence, dict) else {},
                profile,
            )

        def context_value(
            section_name: str,
            key: str = "context",
        ) -> str:
            if not isinstance(intelligence, dict):
                return "Not available"
            section = intelligence.get(section_name)
            if not isinstance(section, dict):
                return "Not available"
            context = section.get(key)
            if isinstance(context, dict):
                return clean(context.get("value"))
            return "Not available"

        def learning_table(text: object) -> Table | None:
            blocks = _learning_blocks(text)
            if not blocks:
                return None

            rows = [
                [
                    Paragraph(heading, label_style),
                    Paragraph(summary, body_style),
                ]
                for heading, summary in blocks[:6]
            ]
            table = Table(
                rows,
                colWidths=[38 * mm, (REPORT_WIDTH_MM - 38.0) * mm],
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle([
                    ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#D6D6D6")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#E5E7EB")),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F6F7F9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ])
            )
            return table

        def add_learning_section(title: str, text: object) -> None:
            table = learning_table(text)
            if table is not None:
                story.extend(split_section(title, table))

        # 1. Country at a Glance
        glance_rows = [
            ("Capital", profile.get("capital")),
            ("Country code", country_code),
            ("ISO alpha-3", profile.get("iso_alpha3")),
            ("Region", profile.get("subregion") or profile.get("region")),
            ("Population", population_value),
            ("Area", area_value),
            ("Language(s)", profile.get("official_languages")),
            ("Currency", profile.get("currency")),
        ]

        if image is not None:
            image_buffer = BytesIO()
            preview_image = image.copy().convert("RGB")
            preview_image.thumbnail((1000, 700))
            preview_image.save(image_buffer, format="JPEG", quality=90)
            image_buffer.seek(0)

            preview = PDFImage(image_buffer)
            preview._restrictSize(40 * mm, 28 * mm)
            identity_table = info_grid(
                glance_rows,
                width_mm=REPORT_WIDTH_MM - 46.0,
            )
            identity_content = Table(
                [[preview, identity_table]],
                colWidths=[46 * mm, (REPORT_WIDTH_MM - 46.0) * mm],
            )
        else:
            identity_content = info_grid(
                glance_rows,
                width_mm=REPORT_WIDTH_MM,
            )
        identity_content.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.extend([
            section_box("Country at a Glance", identity_content),
            Spacer(1, 3 * mm),
        ])

        overview = clean(profile.get("overview"))
        if overview != "Not available":
            overview_paragraph = Paragraph(
                overview,
                ParagraphStyle(
                    "CountryOverviewV3",
                    parent=body_style,
                    leftIndent=6,
                    rightIndent=6,
                    spaceBefore=4,
                    spaceAfter=4,
                ),
            )
            story.extend(split_section("Country Overview", overview_paragraph))

        # 2. Geography
        location_map = _build_pdf_location_map(
            profile.get("latitude"),
            profile.get("longitude"),
            profile.get("area_km2"),
            country_name=decision,
            capital=clean(profile.get("capital")),
            country_code=country_code,
        )
        if location_map is not None:
            location_content = Table(
                [[location_map]],
                colWidths=[REPORT_WIDTH_MM * mm],
                hAlign="CENTER",
            )
            location_content.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            story.extend(split_section("Geographic Location", location_content))

        geography = info_grid(
            [
                ("Largest cities", profile.get("largest_cities")),
                ("Bordering countries", profile.get("borders")),
                ("Time zones", profile.get("timezones")),
                ("Highest point", profile.get("highest_point")),
                ("Lowest point", profile.get("lowest_point")),
                ("Country reference coordinates", coordinates),
            ]
        )
        story.extend([
            section_box("Geography — Key Facts", geography),
            Spacer(1, 3 * mm),
        ])

        climate_text = context_value("environment", "climate_seasons")
        rivers_text = context_value("geography", "rivers_lakes")
        relief_text = context_value("geography", "mountains_relief")
        resources_text = context_value("environment", "natural_resources")

        add_learning_section("Climate & Seasons", climate_text)
        add_learning_section("Rivers, Lakes & Waterways", rivers_text)
        add_learning_section("Mountains & Relief", relief_text)
        add_learning_section("Natural Resources & Raw Materials", resources_text)

        if all(
            value == "Not available"
            for value in (
                climate_text,
                rivers_text,
                relief_text,
                resources_text,
            )
        ):
            add_learning_section(
                "Physical Geography & Environment",
                context_value("environment"),
            )

        if any(
            report_manifest.get(key, False)
            for key in ("flag", "origins", "history")
        ):
            story.append(PageBreak())

        # 3. Flag Intelligence
        if isinstance(intelligence, dict):
            flag_info = intelligence.get("flag")
            if isinstance(flag_info, dict):
                flag_rows: list[tuple[str, object]] = []

                adoption = flag_info.get("adoption_date")
                if isinstance(adoption, dict):
                    flag_rows.append(("Adoption", adoption.get("value")))

                proportion = flag_info.get("proportion")
                if isinstance(proportion, dict):
                    flag_rows.append(("Proportion", proportion.get("value")))

                similar = flag_info.get("similar_flags")
                if isinstance(similar, list) and similar:
                    flag_rows.append(
                        ("Recognition alternatives", ", ".join(similar[:4]))
                    )

                if flag_rows:
                    story.extend([
                        section_box(
                            "Flag Intelligence — Key Facts",
                            info_grid(flag_rows, two_pairs=False),
                        ),
                        Spacer(1, 3 * mm),
                    ])

                for title, field in (
                    ("Flag Design & Construction", "design_origin"),
                    ("Flag Meaning & Symbolism", "symbolism"),
                ):
                    items = flag_info.get(field)
                    if isinstance(items, (list, tuple)):
                        values = [
                            clean(item.get("value"))
                            for item in items
                            if isinstance(item, dict)
                            and clean(item.get("value")) != "Not available"
                        ]
                        if values:
                            add_learning_section(title, "\n\n".join(values))

                flag_history = flag_info.get("historical_flags")
                if isinstance(flag_history, (list, tuple)) and flag_history:
                    rows = [
                        (
                            clean(item.get("period")),
                            clean(item.get("summary")),
                        )
                        for item in flag_history[:8]
                        if isinstance(item, dict)
                    ]
                    if rows:
                        story.extend(
                            split_section(
                                "Flag History",
                                info_grid(rows, two_pairs=False),
                            )
                        )

            # 4. Origins and Historical Journey
            origins = intelligence.get("origins")
            if isinstance(origins, (list, tuple)) and origins:
                origin_rows = [
                    (
                        clean(event.get("label")),
                        clean(event.get("summary")),
                    )
                    for event in origins[:6]
                    if isinstance(event, dict)
                ]
                if origin_rows:
                    story.extend(
                        split_section(
                            "Origins & Early History",
                            info_grid(origin_rows, two_pairs=False),
                        )
                    )

            timeline = intelligence.get("historical_timeline")
            if isinstance(timeline, (list, tuple)) and timeline:
                timeline_rows = [
                    (
                        clean(event.get("period")),
                        clean(event.get("summary")),
                    )
                    for event in timeline[:18]
                    if isinstance(event, dict)
                ]
                if timeline_rows:
                    story.extend(
                        split_section(
                            "Historical Journey",
                            info_grid(timeline_rows, two_pairs=False),
                        )
                    )

        if any(
            report_manifest.get(key, False)
            for key in ("flag", "origins", "history")
        ):
            story.append(PageBreak())

        # 5. State Formation, identity and institutions
        sovereignty = info_grid(
            [
                ("Former colonial power(s)", profile.get("former_colonial_powers")),
                ("Colonial / sovereignty status", profile.get("colonial_period")),
                ("Independence / sovereignty date", profile.get("independence_day")),
                ("Key independence figure", profile.get("independence_leader")),
            ],
            two_pairs=False,
        )
        story.extend([
            section_box("State Formation & Sovereignty", sovereignty),
            Spacer(1, 3 * mm),
        ])

        national_identity = info_grid(
            [
                ("National Day", profile.get("national_day")),
                ("National motto", profile.get("national_motto")),
                ("National anthem", profile.get("national_anthem")),
                ("Demonym", profile.get("demonym")),
            ]
        )
        story.extend([
            section_box("National Identity", national_identity),
            Spacer(1, 3 * mm),
        ])

        government = info_grid(
            [
                ("Government form", profile.get("government_form")),
                ("Head of State", profile.get("head_of_state")),
                ("Head of State office", profile.get("head_of_state_office")),
                ("Head of Government", profile.get("head_of_government")),
                ("Head of Government office", profile.get("head_of_government_office")),
            ],
            two_pairs=False,
        )
        story.extend([
            section_box("Government & Institutions", government),
            Spacer(1, 3 * mm),
        ])
        add_learning_section(
            "Administrative Divisions",
            context_value("government", "administrative_divisions"),
        )

        if any(
            report_manifest.get(key, False)
            for key in (
                "society", "languages_religion", "health",
                "culture", "festivals", "heritage",
            )
        ):
            story.append(PageBreak())

        # 6. People, society and culture
        add_learning_section(
            "People & Society",
            context_value("people_society"),
        )
        add_learning_section(
            "Languages & Religion",
            context_value("people_society", "languages_religion"),
        )
        add_learning_section(
            "Health System & Public Health",
            context_value("people_society", "health_system"),
        )
        add_learning_section(
            "Culture, Cuisine, Music & Sport",
            context_value("culture"),
        )
        add_learning_section(
            "Festivals, Holidays & Traditions",
            context_value("culture", "festivals_holidays"),
        )
        add_learning_section(
            "Heritage, UNESCO & Major Landmarks",
            context_value("culture", "heritage_landmarks"),
        )

        if any(
            report_manifest.get(key, False)
            for key in (
                "economy", "economic_drivers", "infrastructure",
                "transport", "energy", "education", "environment",
            )
        ):
            story.append(PageBreak())

        economy_summary = info_grid(
            [
                ("GDP (current US$)", gdp_value),
                ("GDP source", profile.get("gdp_source")),
            ]
        )
        story.extend([
            section_box("Economy — Key Metric", economy_summary),
            Spacer(1, 3 * mm),
        ])
        add_learning_section(
            "Economic Structure & Trade",
            context_value("economy"),
        )
        add_learning_section(
            "Economic Drivers, Industries & Exports",
            context_value("economy", "economic_drivers"),
        )
        add_learning_section(
            "Infrastructure Overview",
            context_value("infrastructure"),
        )
        add_learning_section(
            "Transport Network, Ports & Airports",
            context_value("infrastructure", "transport_network"),
        )
        add_learning_section(
            "Energy & Connectivity",
            context_value("infrastructure", "energy_connectivity"),
        )
        add_learning_section(
            "Education, Science & Innovation",
            context_value("education_science"),
        )
        add_learning_section(
            "Environment & Biodiversity",
            context_value("environment"),
        )

        if any(
            report_manifest.get(key, False)
            for key in ("practical", "international", "notable_people")
        ):
            story.append(PageBreak())

        practical_rows: list[tuple[str, object]] = [
            ("Calling code", profile.get("calling_code")),
            ("Emergency numbers", profile.get("emergency_numbers")),
            ("Driving side", profile.get("driving_side")),
            ("Internet domain", profile.get("internet_domain")),
            ("Time zones", profile.get("timezones")),
        ]
        story.extend([
            section_box(
                "Practical & Emergency Information",
                info_grid(practical_rows, two_pairs=False),
            ),
            Spacer(1, 3 * mm),
        ])

        add_learning_section(
            "International Relations",
            context_value("international_relations"),
        )
        add_learning_section(
            "Notable Public Figures",
            context_value("culture", "notable_people"),
        )

        # 7. Did You Know? — only from already sourced profile facts
        did_you_know_rows: list[tuple[str, object]] = []
        if clean(profile.get("highest_point")) != "Not available":
            did_you_know_rows.append(
                ("Geography", f"Highest point: {clean(profile.get('highest_point'))}.")
            )
        if clean(profile.get("national_anthem")) != "Not available":
            did_you_know_rows.append(
                ("National identity", f"National anthem: {clean(profile.get('national_anthem'))}.")
            )
        if clean(profile.get("largest_cities")) != "Not available":
            did_you_know_rows.append(
                ("Urban geography", f"Major cities include {clean(profile.get('largest_cities'))}.")
            )
        if clean(profile.get("national_motto")) != "Not available":
            did_you_know_rows.append(
                ("National motto", clean(profile.get("national_motto")))
            )
        if clean(profile.get("timezones")) != "Not available":
            did_you_know_rows.append(
                ("Time zone", clean(profile.get("timezones")))
            )

        if did_you_know_rows:
            story.extend([
                section_box(
                    "Did You Know?",
                    info_grid(did_you_know_rows[:5], two_pairs=False),
                ),
                Spacer(1, 3 * mm),
            ])



    else:
        rejection = info_grid(
            [
                ("Status", status),
                ("Top candidate", decision),
                ("Country code", country_code),
                ("Confidence", f"{confidence:.2%}"),
                ("Top-1 margin", f"{decision_margin:.2%}"),
                ("Decision mode", decision_reason.replace("_", " ").title()),
            ],
            two_pairs=False,
        )
        story.extend([
            section_box("Recognition Result", rejection),
            Spacer(1, 3 * mm),
        ])

    source_note = Table(
        [[
            Paragraph(
                "<b>Sources:</b> Wikidata, World Bank and Wikipedia where "
                "available. Values may use different reference years.",
                small_style,
            )
        ]],
        colWidths=[REPORT_WIDTH_MM * mm],
    )
    source_note.setStyle(
        TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#777777")),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    if not accepted or not isinstance(
        report.get("country_intelligence_v2"),
        dict,
    ):
        story.append(source_note)

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
    layout="wide",
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


def resolve_emergency_numbers(
    country_code: str,
    profile_value: str | None = None,
) -> str:
    """Return emergency numbers even if a stale profile/cache is incomplete."""
    current = str(profile_value or "").strip()
    if current and current != "Not available":
        try:
            profile = get_country_profile_v2(
                country_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
            calling_code = profile.calling_code
        except Exception:
            calling_code = ""
        return format_emergency_numbers(
            current,
            calling_code,
        )

    try:
        value = fetch_emergency_numbers(country_code)
    except Exception:
        value = "Not available"

    if value != "Not available":
        try:
            profile = get_country_profile_v2(
                country_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
            calling_code = profile.calling_code
        except Exception:
            calling_code = ""
        return format_emergency_numbers(
            value,
            calling_code,
        )

    try:
        value = fetch_emergency_numbers_fallback(country_code)
    except Exception:
        value = "No national emergency number documented"

    try:
        profile = get_country_profile_v2(
                country_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
        calling_code = profile.calling_code
    except Exception:
        calling_code = ""

    return format_emergency_numbers(
        value,
        calling_code,
    )


COUNTRY_PROFILE_SCHEMA_VERSION = "2026-09-28-v18"
COUNTRY_INTELLIGENCE_SCHEMA_VERSION = "2026-09-28-v21"

@st.cache_data(ttl=1800, show_spinner=False)
def get_country_profile_v2(
    country_code: str,
    schema_version: str = COUNTRY_PROFILE_SCHEMA_VERSION,
):
    # schema_version is intentionally part of the cache key.
    _ = schema_version
    return fetch_country_profile(country_code)


def get_fresh_historical_profile(country_code: str):
    """Bypass Streamlit profile cache for historical facts."""
    return country_info_module.fetch_country_profile(
        country_code
    )


def _learning_blocks(value: object) -> list[tuple[str, str]]:
    """Parse concise 'Heading: summary' encyclopedia context into learning blocks."""
    text = str(value or "").strip()
    if not text or text == "Not available":
        return []

    blocks: list[tuple[str, str]] = []
    for raw in re.split(r"\n\s*\n", text):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            heading, summary = raw.split(":", 1)
        else:
            heading, summary = "Overview", raw
        heading = heading.strip()
        summary = summary.strip()
        if summary:
            blocks.append((heading, summary))
    return blocks


def _country_profile_payload(
    country_code: str,
    profile,
    historical_profile,
) -> dict[str, object]:
    """Normalize the legacy profile once for UI, JSON and PDF."""
    return {
        "code": country_code.upper(),
        "name": profile.name,
        "capital": profile.capital,
        "population": profile.population.value,
        "population_year": profile.population.year,
        "population_source": profile.population.source,
        "currency": profile.currency,
        "official_languages": profile.official_languages,
        "continent": profile.continent,
        "area_km2": profile.area_km2,
        "overview": canonical_overview_text(profile.overview),
        "national_day": historical_profile.national_day,
        "independence_day": historical_profile.independence_day,
        "colonial_history": getattr(
            historical_profile,
            "colonial_history",
            "Not applicable",
        ),
        "former_colonial_powers": getattr(
            historical_profile,
            "former_colonial_powers",
            "Not applicable",
        ),
        "colonial_period": getattr(
            historical_profile,
            "colonial_period",
            "Not applicable",
        ),
        "independence_leader": getattr(
            historical_profile,
            "independence_leader",
            "Not applicable",
        ),
        "historical_context": getattr(
            historical_profile,
            "historical_context",
            "No classical colonial-independence transition is documented "
            "in the available country overview.",
        ),
        "national_motto": profile.national_motto,
        "national_anthem": profile.national_anthem,
        "region": getattr(profile, "region", "Not available"),
        "subregion": getattr(profile, "subregion", "Not available"),
        "demonym": getattr(profile, "demonym", "Not available"),
        "iso_alpha3": getattr(profile, "iso_alpha3", "Not available"),
        "timezones": getattr(profile, "timezones", "Not available"),
        "borders": getattr(profile, "borders", "Not available"),
        "largest_cities": getattr(profile, "largest_cities", "Not available"),
        "international_organizations": getattr(
            profile,
            "international_organizations",
            "Not available",
        ),
        "official_religion": getattr(
            profile,
            "official_religion",
            "Not available",
        ),
        "highest_point": getattr(profile, "highest_point", "Not available"),
        "lowest_point": getattr(profile, "lowest_point", "Not available"),
        "gdp_usd": getattr(getattr(profile, "gdp", None), "value_usd", None),
        "gdp_year": getattr(getattr(profile, "gdp", None), "year", None),
        "gdp_source": getattr(
            getattr(profile, "gdp", None),
            "source",
            "World Bank",
        ),
        "government_form": profile.government_form,
        "head_of_state": profile.head_of_state,
        "head_of_state_office": profile.head_of_state_office,
        "head_of_government": profile.head_of_government,
        "head_of_government_office": profile.head_of_government_office,
        "calling_code": profile.calling_code,
        "emergency_numbers": resolve_emergency_numbers(
            country_code,
            getattr(profile, "emergency_numbers", None),
        ),
        "internet_domain": profile.internet_domain,
        "driving_side": profile.driving_side,
        "latitude": profile.latitude,
        "longitude": profile.longitude,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_country_intelligence_v2(
    country_code: str,
    similar_flags: tuple[str, ...] = (),
    schema_version: str = COUNTRY_INTELLIGENCE_SCHEMA_VERSION,
):
    """Build and enrich a reusable source-aware country knowledge payload."""
    _ = schema_version
    profile = get_country_profile_v2(
        country_code,
        schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
    )
    try:
        historical_profile = get_fresh_historical_profile(country_code)
    except Exception:
        historical_profile = profile

    payload = _country_profile_payload(
        country_code,
        profile,
        historical_profile,
    )
    intelligence = build_from_legacy_profile(payload)

    try:
        intelligence = enrich_from_encyclopedia(
            intelligence,
            title=profile.name,
        )
    except (requests.RequestException, LookupError, ValueError):
        # Structured facts remain available even if narrative enrichment fails.
        pass

    try:
        canonical_fact = intelligence.identity.get(
            "encyclopedia_title"
        )
        flag_country_name = (
            str(canonical_fact.value)
            if canonical_fact is not None
            else profile.name
        )
        intelligence.flag = enrich_flag_profile(
            intelligence.flag,
            flag_country_name,
            similar_flags=similar_flags,
        )
    except (requests.RequestException, LookupError, ValueError):
        # A missing dedicated flag article must never hide country knowledge.
        pass

    intelligence_payload = intelligence.to_dict()
    report_manifest = build_report_manifest(
        intelligence_payload,
        payload,
    )

    return {
        "profile": payload,
        "intelligence": intelligence_payload,
        "completion": section_completion(intelligence),
        "validation": validate_country_intelligence(intelligence),
        "report_manifest": report_manifest,
        "missing_required_report_sections": missing_required_sections(
            report_manifest
        ),
    }


def build_product_header_html() -> str:
    """Render the restrained product brand header."""
    return """
    <div class="product-header">
        <div class="brand-lockup">
            <div class="brand-mark">FI</div>
            <div class="brand-copy">
                <div class="brand-name">Flag Intelligence</div>
                <div class="brand-tagline">Worldwide country intelligence</div>
            </div>
        </div>
        <div class="header-meta">
            250-country taxonomy · Source-aware knowledge
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
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        .product-header {
            min-height: 86px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0 1.25rem;
            border: 1px solid var(--line);
            border-bottom: 0;
            background: #ffffff;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: .8rem;
        }

        .brand-mark {
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0f2f57;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: .02em;
        }

        .brand-name {
            color: #0f2f57;
            font-size: 1.1rem;
            font-weight: 850;
            line-height: 1.15;
        }

        .brand-tagline {
            color: #667085;
            font-size: .76rem;
            margin-top: .12rem;
        }

        .header-meta {
            color: #667085;
            font-size: .78rem;
            white-space: nowrap;
        }

        .breadcrumb {
            color: #52606d;
            font-size: .76rem;
            text-transform: uppercase;
            letter-spacing: .035em;
            margin: .75rem 0 1.4rem;
        }

        .breadcrumb strong {
            color: #0f2f57;
        }

        .section-rail-title {
            color: #0f2f57;
            font-size: 1.15rem;
            font-weight: 800;
            border-top: 1px solid #cfd7e3;
            padding-top: .85rem;
            margin-bottom: .45rem;
        }

        .section-rail-copy {
            color: #667085;
            font-size: .8rem;
            line-height: 1.45;
            margin-bottom: .8rem;
        }

        .content-title {
            color: #0f2f57;
            font-size: 2rem;
            font-weight: 450;
            letter-spacing: -.025em;
            margin: 0 0 .85rem;
        }

        .content-subtitle {
            color: #667085;
            font-size: .96rem;
            line-height: 1.55;
            max-width: 780px;
            margin-bottom: 1.25rem;
        }

        .page-kicker {
            color: var(--blue);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .25rem;
        }

        .page-title {
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 850;
            letter-spacing: -.025em;
            margin-bottom: .35rem;
        }

        .page-copy {
            color: var(--muted);
            font-size: .96rem;
            line-height: 1.55;
            max-width: 760px;
            margin-bottom: 1rem;
        }

        div[data-testid="stRadio"] > div {
            gap: .35rem;
        }

        div[data-testid="stRadio"] label {
            border-radius: 999px;
        }

        div[data-testid="stSelectbox"] > div > div {
            border-radius: 12px;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 12px;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 2px;
            font-weight: 700;
            min-height: 42px;
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            border: 0;
            box-shadow: none;
            background: transparent;
            color: #0f2f57;
        }

        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: #f3f6fa;
            color: #0f2f57;
        }

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 2px !important;
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

        .preview-fixed {
            width: 189px;
            height: 151px;
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
            padding: 8px;
            box-sizing: border-box;
            overflow: hidden;
            margin: 0 auto;
        }

        .preview-fixed img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .result-preview-fixed {
            width: 320px;
            height: 240px;
            border-radius: 14px;
            background: #f7f7f8;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin: 0 auto;
            padding: 10px;
            box-sizing: border-box;
        }

        .result-preview-fixed img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
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
    
        .site-shell {
            margin: 0 0 0.75rem 0;
            border-left: 1px solid #d9e0e8;
            border-right: 1px solid #d9e0e8;
            background: #fff;
        }

        .utility-bar {
            min-height: 40px;
            display: flex;
            align-items: center;
            background: #0b2e55;
            border-bottom: 1px solid #0b2e55;
        }

        .utility-spacer { flex: 1; }

        .utility-nav {
            display: flex;
            align-items: stretch;
            min-height: 40px;
        }

        .utility-link {
            display: flex;
            align-items: center;
            padding: 0 14px;
            color: #fff !important;
            font-size: .74rem;
            font-weight: 600;
            text-decoration: none !important;
            border-left: 1px solid rgba(255,255,255,.18);
        }

        .utility-link:hover,
        .utility-link.is-active {
            background: rgba(255,255,255,.1);
        }

        .main-nav {
            min-height: 72px;
            display: grid;
            grid-template-columns: 280px 1fr;
            align-items: stretch;
            border-bottom: 1px solid #d7dee8;
        }

        .site-brand {
            display: flex;
            align-items: center;
            gap: .8rem;
            padding: 0 22px;
            text-decoration: none !important;
            border-right: 1px solid #e3e8ef;
        }

        .primary-nav {
            display: flex;
            align-items: stretch;
        }

        .primary-link {
            display: flex;
            align-items: center;
            padding: 0 24px;
            color: #0b2e55 !important;
            font-size: .95rem;
            font-weight: 760;
            text-decoration: none !important;
            border-bottom: 3px solid transparent;
        }

        .primary-link:hover {
            background: #f7f9fc;
            border-bottom-color: #f1a400;
        }

        .primary-link.is-active {
            border-bottom-color: #f1a400;
            background: #fbfcfe;
        }

        .breadcrumb a {
            color: #0b2e55 !important;
            text-decoration: underline;
            text-underline-offset: 2px;
        }

        .local-nav {
            border-top: 1px solid #cfd7e3;
            margin-top: .15rem;
        }

        .local-nav a {
            position: relative;
            display: block;
            padding: .72rem .65rem .72rem 1rem;
            border-bottom: 1px solid #e7ebf0;
            color: #143659 !important;
            text-decoration: none !important;
            font-size: .86rem;
            font-weight: 650;
        }

        .local-nav a::before {
            content: "";
            position: absolute;
            left: 0;
            top: 16px;
            bottom: 16px;
            width: 2px;
            background: #f1a400;
        }

        .local-nav a:hover,
        .local-nav a.is-active {
            background: #f7f9fc;
            color: #001f3f !important;
        }

        .local-nav a.is-active {
            font-weight: 800;
        }

        .page-frame {
            border: 1px solid #d9e0e8;
            background: #fff;
            padding: 1.8rem 2rem 2rem;
        }

        .content-title {
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 430;
        }

        .content-subtitle {
            max-width: 900px;
        }

        .section-heading {
            color: #0b2e55;
            font-size: 1.35rem;
            font-weight: 700;
            padding-bottom: .45rem;
            border-bottom: 2px solid #0b2e55;
            margin: .2rem 0 1rem;
        }

        .section-heading span {
            color: #f1a400;
        }

        @media (max-width: 850px) {
            .utility-nav {
                overflow-x: auto;
            }

            .main-nav {
                grid-template-columns: 1fr;
            }

            .site-brand {
                min-height: 68px;
                border-right: 0;
                border-bottom: 1px solid #e3e8ef;
            }

            .primary-nav {
                overflow-x: auto;
            }

            .primary-link {
                padding: 0 16px;
                min-height: 52px;
                white-space: nowrap;
            }

            .header-meta {
                display: none;
            }
        }
</style>
    """,
    unsafe_allow_html=True,
)


def render_fixed_upload_preview(
    image: Image.Image | None,
) -> None:
    """Render a fixed 5 cm × 4 cm preview area without layout shift."""
    if image is None:
        inner = "Image preview"
    else:
        buffer = BytesIO()
        preview = image.copy().convert("RGB")
        preview.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        inner = (
            f'<img src="data:image/jpeg;base64,{encoded}" '
            'alt="Image preview">'
        )

    st.markdown(
        f'<div class="preview-fixed">{inner}</div>',
        unsafe_allow_html=True,
    )


def render_fixed_result_preview(
    image: Image.Image,
) -> None:
    """Render the dialog image in a fixed display area."""
    buffer = BytesIO()
    preview = image.copy().convert("RGB")
    preview.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    st.markdown(
        (
            '<div class="result-preview-fixed">'
            f'<img src="data:image/jpeg;base64,{encoded}" '
            'alt="Recognition input">'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def evaluate_production_decision(
    candidates: list[tuple[str, float]],
    hard_threshold: float,
) -> tuple[bool, float, str]:
    """Adaptive open-set decision using confidence and class separation."""
    if not candidates:
        return False, 0.0, "no_candidates"

    top1 = float(candidates[0][1])
    top2 = (
        float(candidates[1][1])
        if len(candidates) > 1
        else 0.0
    )
    margin = top1 - top2

    # High-confidence path preserves the calibrated open-set behavior.
    if top1 >= hard_threshold:
        return True, margin, "high_confidence"

    # Strong class separation: useful for real-world flags whose confidence
    # is depressed by folds, perspective, lighting or compression.
    if top1 >= 0.55 and margin >= 0.30:
        return True, margin, "dominant_candidate"

    # Slightly higher confidence permits a smaller but still clear margin.
    if top1 >= 0.72 and margin >= 0.18:
        return True, margin, "strong_candidate"

    return False, margin, "ambiguous"


def _query_value(name: str, default: str = "") -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value or default)


def _nav_href(page: str, **params: str) -> str:
    from urllib.parse import urlencode

    payload = {"page": page}
    payload.update({
        key: value
        for key, value in params.items()
        if value not in (None, "")
    })
    return "?" + urlencode(payload)


def _active_class(current: str, target: str) -> str:
    return "is-active" if current == target else ""


def render_site_navigation(current_page: str) -> None:
    """Render a two-level public-site navigation bar."""
    top_links = (
        ("Country Explorer", "explore"),
        ("Methodology", "about"),
        ("Worldwide Audit", "audit"),
        ("About", "about"),
    )
    primary_links = (
        ("Recognize", "recognize"),
        ("Countries", "explore"),
        ("Learn", "learn"),
        ("Reports", "reports"),
    )

    utility = "".join(
        f'<a class="utility-link {_active_class(current_page, page)}" '
        f'href="{_nav_href(page)}">{label}</a>'
        for label, page in top_links
    )
    primary = "".join(
        f'<a class="primary-link {_active_class(current_page, page)}" '
        f'href="{_nav_href(page)}">{label}</a>'
        for label, page in primary_links
    )

    st.markdown(
        f"""
        <div class="site-shell">
          <div class="utility-bar">
            <div class="utility-spacer"></div>
            <nav class="utility-nav">{utility}</nav>
          </div>
          <div class="main-nav">
            <a class="site-brand" href="{_nav_href('home')}">
              <div class="brand-mark">FI</div>
              <div class="brand-copy">
                <div class="brand-name">Flag Intelligence</div>
                <div class="brand-tagline">Worldwide country intelligence</div>
              </div>
            </a>
            <nav class="primary-nav">{primary}</nav>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_breadcrumb(items: list[tuple[str, str | None]]) -> None:
    chunks = []
    for label, href in items:
        if href:
            chunks.append(f'<a href="{href}">{label}</a>')
        else:
            chunks.append(f"<strong>{label}</strong>")
    st.markdown(
        '<div class="breadcrumb">' + " / ".join(chunks) + "</div>",
        unsafe_allow_html=True,
    )


bundle = None
deployment_threshold = None
image = None
process = False
text_process = False
typed_country = ""
direct_selected_code = None

page_aliases = {
    "home": "home",
    "recognize": "recognize",
    "explore": "explore",
    "countries": "explore",
    "about": "about",
    "audit": "audit",
    "learn": "learn",
    "reports": "reports",
    "country": "country",
}
active_page = page_aliases.get(
    _query_value("page", "home").lower(),
    "home",
)

render_site_navigation(active_page)

if active_page == "home":
    render_breadcrumb([("Flag Intelligence", None)])
elif active_page == "country":
    render_breadcrumb([
        ("Flag Intelligence", _nav_href("home")),
        ("Countries", _nav_href("explore")),
        (_query_value("country", "Country"), None),
    ])
else:
    render_breadcrumb([
        ("Flag Intelligence", _nav_href("home")),
        (active_page.replace("_", " ").title(), None),
    ])

left_rail, main_area = st.columns([0.22, 0.78], gap="large")

with left_rail:
    st.markdown(
        '<div class="section-rail-title">Flag Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-rail-copy">'
        'Recognize flags, explore countries and study structured public knowledge.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <nav class="local-nav">
          <a class="{_active_class(active_page, 'home')}" href="{_nav_href('home')}">Home</a>
          <a class="{_active_class(active_page, 'recognize')}" href="{_nav_href('recognize')}">Recognize a flag</a>
          <a class="{_active_class(active_page, 'explore')}" href="{_nav_href('explore')}">Explore countries</a>
          <a class="{_active_class(active_page, 'learn')}" href="{_nav_href('learn')}">Learning guide</a>
          <a class="{_active_class(active_page, 'reports')}" href="{_nav_href('reports')}">Reports & exports</a>
          <a class="{_active_class(active_page, 'audit')}" href="{_nav_href('audit')}">Worldwide audit</a>
          <a class="{_active_class(active_page, 'about')}" href="{_nav_href('about')}">About</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )

with main_area:
    st.markdown('<div class="page-frame">', unsafe_allow_html=True)

    if active_page == "home":
        st.markdown(
            '<div class="content-title">Understand a country from its flag or name</div>'
            '<div class="content-subtitle">'
            'Flag Intelligence brings recognition and country knowledge into one '
            'workflow. Start with an image when you need identification, or search '
            'the country directly when you already know its name.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-heading"><span>01</span> Choose your entry point</div>',
            unsafe_allow_html=True,
        )
        action = st.selectbox(
            "Start with",
            (
                "Select an action…",
                "Recognize a flag image",
                "Explore a country by name",
            ),
            label_visibility="collapsed",
        )
        if action == "Recognize a flag image":
            st.markdown(
                f'<a href="{_nav_href("recognize")}">Continue to flag recognition →</a>',
                unsafe_allow_html=True,
            )
        elif action == "Explore a country by name":
            st.markdown(
                f'<a href="{_nav_href("explore")}">Continue to country explorer →</a>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-heading"><span>02</span> What the system covers</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Flag intelligence, identity, geography, historical journey, "
            "government, society, culture, economy, infrastructure, education, "
            "international relations and practical information where current "
            "source coverage supports them."
        )

    elif active_page == "recognize":
        st.markdown(
            '<div class="content-title">Recognize a flag</div>'
            '<div class="content-subtitle">'
            'Upload a clear flag image. The result opens the same country profile '
            'used by direct search.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-heading"><span>01</span> Upload image</div>',
            unsafe_allow_html=True,
        )
        if not MODEL_PATH.is_file():
            st.error("The visual recognition model is temporarily unavailable.")
        else:
            bundle = get_model()
            deployment_threshold = get_deployment_threshold()

        upload_col, preview_col = st.columns([1.35, 0.65], gap="large")
        with upload_col:
            uploaded_file = st.file_uploader(
                "Choose flag image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            st.caption(
                "Supported: JPG, PNG and WEBP. Use an image where the flag is clearly visible."
            )
            process = st.button(
                "Recognize flag",
                type="primary",
                use_container_width=True,
                disabled=(uploaded_file is None or bundle is None),
            )
        with preview_col:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
            render_fixed_upload_preview(image)

    elif active_page == "explore":
        st.markdown(
            '<div class="content-title">Country explorer</div>'
            '<div class="content-subtitle">'
            'Search the worldwide catalogue by country name. No flag image or '
            'recognition score is required.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-heading"><span>01</span> Select country</div>',
            unsafe_allow_html=True,
        )

        country_options = sorted(
            {
                display_country_name(str(country.alpha_2).lower()):
                str(country.alpha_2).lower()
                for country in __import__("pycountry").countries
            }.items(),
            key=lambda item: item[0],
        )
        if not any(code == "xk" for _, code in country_options):
            country_options.append(("Kosovo", "xk"))
            country_options.sort(key=lambda item: item[0])

        label_to_code = {label: code for label, code in country_options}
        labels = ["Select a country…"] + [label for label, _ in country_options]
        selected_label = st.selectbox(
            "Country",
            labels,
            help="Start typing to filter the list.",
        )
        with st.expander("Use an alias or ISO code"):
            typed_country = st.text_input(
                "Country name or ISO code",
                placeholder="Examples: Ivory Coast, Côte d’Ivoire, FRA, CI, XK",
            ).strip()

        direct_selected_code = (
            label_to_code.get(selected_label)
            if selected_label != "Select a country…"
            else None
        )
        text_process = st.button(
            "Open country profile",
            type="primary",
            use_container_width=True,
            disabled=not bool(direct_selected_code or typed_country),
        )

    elif active_page == "learn":
        st.markdown(
            '<div class="content-title">Learning guide</div>'
            '<div class="content-subtitle">'
            'Use Flag Intelligence as a structured way to move from recognition '
            'to deeper country understanding.'
            '</div>',
            unsafe_allow_html=True,
        )
        for number, title, body in (
            ("01", "Identify", "Recognize a flag or select the country by name."),
            ("02", "Orient", "Read identity, capital, population, map and geographic context."),
            ("03", "Understand", "Follow the historical journey, institutions and society."),
            ("04", "Connect", "Study culture, economy, infrastructure and international context."),
            ("05", "Verify", "Use the attached source references and coverage notes."),
        ):
            st.markdown(
                f'<div class="section-heading"><span>{number}</span> {title}</div>',
                unsafe_allow_html=True,
            )
            st.write(body)

    elif active_page == "reports":
        st.markdown(
            '<div class="content-title">Reports & exports</div>'
            '<div class="content-subtitle">'
            'Country profiles can be exported as structured JSON or reader-friendly PDF '
            'when required knowledge sections are available.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-heading"><span>01</span> JSON</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Preserves the structured Country Intelligence record, provenance and "
            "machine-readable values."
        )
        st.markdown(
            '<div class="section-heading"><span>02</span> PDF</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Builds a reader-oriented A4 country report from the same validated record."
        )

    elif active_page == "audit":
        st.markdown(
            '<div class="content-title">Worldwide quality audit</div>'
            '<div class="content-subtitle">'
            'The project audits the complete 250-class taxonomy to expose structural '
            'and coverage gaps rather than hide them.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-heading"><span>01</span> Coverage</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "249 ISO 3166-1 country/territory codes plus Kosovo (XK), validated in "
            "sharded GitHub Actions runs."
        )
        st.markdown(
            '<div class="section-heading"><span>02</span> Checks</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Required report sections, source failures, timeline contamination, "
            "raw MediaWiki markup, sovereignty coherence and section coverage."
        )

    else:
        st.markdown(
            '<div class="content-title">About Flag Intelligence</div>'
            '<div class="content-subtitle">'
            'A country-learning system that combines computer vision with structured '
            'public knowledge and explicit source-aware validation.'
            '</div>',
            unsafe_allow_html=True,
        )
        about_topic = st.selectbox(
            "Topic",
            (
                "Product overview",
                "Data & sources",
                "Recognition model",
                "Privacy & safety",
                "Limitations",
            ),
        )
        about_copy = {
            "Product overview": (
                "Flag-image recognition and direct country lookup converge on one "
                "Country Intelligence experience."
            ),
            "Data & sources": (
                "Current sources include Wikidata, World Bank, REST Countries and "
                "Wikipedia/MediaWiki, with coverage varying by country and domain."
            ),
            "Recognition model": (
                "The visual baseline uses MobileNetV3-Small with Top-K alternatives, "
                "visually equivalent flag handling and open-set decision logic."
            ),
            "Privacy & safety": (
                "The product focuses on public country-level information and is not "
                "designed to collect or expose sensitive personal information."
            ),
            "Limitations": (
                "Recognition remains sensitive to uncontrolled imagery, while deep "
                "country coverage depends on public-source quality and structure."
            ),
        }
        st.write(about_copy[about_topic])

    st.markdown("</div>", unsafe_allow_html=True)



def show_result(
    image: Image.Image | None = None,
    direct_code: str | None = None,
):
    if direct_code is None:
        with st.spinner("Processing image..."):
            prediction = predict_robust(
                image,
                bundle,
                top_k=20,
            )

        grouped_candidates = merge_visually_identical_candidates(
            prediction.top5
        )
        decision_code, decision_confidence = grouped_candidates[0]
        display_candidates = grouped_candidates[:5]
        accepted, decision_margin, decision_reason = (
            evaluate_production_decision(
                grouped_candidates,
                deployment_threshold,
            )
        )
        input_mode_used = "image"
    else:
        decision_code = direct_code.lower()
        decision_confidence = None
        display_candidates = [(decision_code, 1.0)]
        accepted = True
        decision_margin = None
        decision_reason = "country_name_input"
        input_mode_used = "text"

    country = display_country_name(decision_code)

    preview_col, result_col = st.columns([.9, 1.1], gap="large")

    with preview_col:
        if image is not None:
            render_fixed_result_preview(image)
        else:
            st.markdown(
                f"<div class='result-name'>{country}</div>"
                f"<div class='result-code'>{decision_code.upper()}</div>",
                unsafe_allow_html=True,
            )
            st.caption("Country selected by name")

    with result_col:
        st.markdown(
            f'<div class="result-name">{country if accepted else "Unknown"}</div>',
            unsafe_allow_html=True,
        )
        if input_mode_used == "image":
            st.markdown(
                f'<div class="result-code">Top candidate · {country} ({decision_code.upper()})</div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Confidence", f"{decision_confidence:.1%}")
            with m2:
                st.metric("Top-1 margin", f"{decision_margin:.1%}")
        else:
            st.markdown(
                f'<div class="result-code">Selected country · {decision_code.upper()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="decision-ok">Country selected by name</div>',
                unsafe_allow_html=True,
            )

        if input_mode_used == "image":
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
                    '<div class="decision-no">Prediction remains ambiguous</div>',
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

    if input_mode_used == "image":
        with st.expander("Top candidates", expanded=False):
            st.dataframe(
                ranking.style.format({"Confidence": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )

    if accepted:
        try:
            profile = get_country_profile_v2(
                decision_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
            try:
                historical_profile = get_fresh_historical_profile(
                    decision_code
                )
            except Exception:
                historical_profile = profile

            knowledge = get_country_intelligence_v2(
                decision_code,
                tuple(
                    display_country_name(code)
                    for code, _ in display_candidates[1:5]
                ),
                schema_version=COUNTRY_INTELLIGENCE_SCHEMA_VERSION,
            )
            intelligence = knowledge["intelligence"]
            completion = knowledge["completion"]

            st.markdown(
                f'<div class="breadcrumb">FLAG INTELLIGENCE &nbsp;/&nbsp; '
                f'COUNTRIES &nbsp;/&nbsp; <strong>{country}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="content-title">{country}</div>'
                f'<div class="content-subtitle">'
                f'Country profile and source-aware intelligence · '
                f'{decision_code.upper()}'
                f'</div>',
                unsafe_allow_html=True,
            )

            country_rail, country_main = st.columns(
                [0.24, 0.76],
                gap="large",
            )

            section_key = _query_value("section", "overview").lower()
            section_map = {
                "overview": "Overview",
                "history": "Flag & History",
                "geography": "Geography",
                "society": "Government & Society",
                "economy": "Economy & Infrastructure",
                "ask": "Ask & Export",
            }
            country_section = section_map.get(section_key, "Overview")

            with country_rail:
                st.markdown(
                    '<div class="section-rail-title">Country sections</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <nav class="local-nav">
                      <a class="{_active_class(section_key, 'overview')}" href="{_nav_href('country', code=decision_code, country=country, section='overview')}">Overview</a>
                      <a class="{_active_class(section_key, 'history')}" href="{_nav_href('country', code=decision_code, country=country, section='history')}">Flag & History</a>
                      <a class="{_active_class(section_key, 'geography')}" href="{_nav_href('country', code=decision_code, country=country, section='geography')}">Geography</a>
                      <a class="{_active_class(section_key, 'society')}" href="{_nav_href('country', code=decision_code, country=country, section='society')}">Government & Society</a>
                      <a class="{_active_class(section_key, 'economy')}" href="{_nav_href('country', code=decision_code, country=country, section='economy')}">Economy & Infrastructure</a>
                      <a class="{_active_class(section_key, 'ask')}" href="{_nav_href('country', code=decision_code, country=country, section='ask')}">Ask & Export</a>
                    </nav>
                    """,
                    unsafe_allow_html=True,
                )

                complete_count = sum(bool(value) for value in completion.values())
                total_count = max(len(completion), 1)
                st.progress(complete_count / total_count)
                st.caption(
                    f"{complete_count}/{total_count} knowledge domains supported"
                )

                with st.expander("Coverage notes", expanded=False):
                    validation = knowledge.get("validation") or []
                    if validation:
                        for issue in validation:
                            st.write(f"• {issue}")
                    else:
                        st.caption("No validation warnings for the current record.")

            with country_main:
                if country_section == "Overview":
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

                    profile_gdp = getattr(profile, "gdp", None)
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        if (
                            profile_gdp is not None
                            and getattr(profile_gdp, "value_usd", None) is not None
                        ):
                            st.metric(
                                "GDP",
                                f"$ {profile_gdp.value_usd / 1_000_000_000:,.1f}B",
                            )
                            gdp_note = getattr(
                                profile_gdp,
                                "source",
                                "World Bank",
                            )
                            if getattr(profile_gdp, "year", None):
                                gdp_note += f" · {profile_gdp.year}"
                            st.caption(gdp_note)
                    with metric_cols[1]:
                        st.metric("National Day", profile.national_day)

                    st.markdown("#### Identity")
                    identity_left, identity_right = st.columns(2, gap="large")
                    with identity_left:
                        st.markdown("**Official language(s)**")
                        st.write(profile.official_languages)
                        st.markdown("**Region / Subregion**")
                        st.write(
                            f"{getattr(profile, 'region', 'Not available')} · "
                            f"{getattr(profile, 'subregion', 'Not available')}"
                        )
                        st.markdown("**Demonym**")
                        st.write(getattr(profile, "demonym", "Not available"))
                        st.markdown("**National motto**")
                        st.write(profile.national_motto)
                    with identity_right:
                        st.markdown("**National anthem**")
                        st.write(profile.national_anthem)
                        st.markdown("**Country overview**")
                        st.write(profile.overview)

                elif country_section == "Flag & History":
                    flag_info = intelligence.get("flag") or {}
                    st.markdown("#### Flag Intelligence")

                    adoption = flag_info.get("adoption_date")
                    proportion = flag_info.get("proportion")
                    flag_meta = st.columns(2)
                    with flag_meta[0]:
                        if isinstance(adoption, dict) and adoption.get("value"):
                            st.metric("Adoption", adoption["value"])
                    with flag_meta[1]:
                        if isinstance(proportion, dict) and proportion.get("value"):
                            st.metric("Proportion", proportion["value"])

                    for heading, key in (
                        ("Design & construction", "design_origin"),
                        ("Meaning & symbolism", "symbolism"),
                    ):
                        items = flag_info.get(key) or []
                        if items:
                            st.markdown(f"**{heading}**")
                            for item in items:
                                if isinstance(item, dict):
                                    blocks = _learning_blocks(item.get("value"))
                                    if blocks:
                                        for block_heading, summary in blocks:
                                            st.markdown(f"**{block_heading}**")
                                            st.write(summary)

                    timeline = intelligence.get("historical_timeline") or []
                    st.markdown("#### Historical Journey")
                    if timeline:
                        for event in timeline:
                            period = event.get("period") or "Historical period"
                            label = event.get("label") or "Event"
                            with st.expander(f"{period} — {label}", expanded=False):
                                st.write(event.get("summary") or "Not available")
                    else:
                        st.info("No sufficiently supported historical timeline is available.")

                    origins = intelligence.get("origins") or []
                    if origins:
                        st.markdown("#### Origins & Early History")
                        for event in origins:
                            st.markdown(
                                f"**{event.get('label', 'Early history')}**"
                            )
                            st.write(event.get("summary") or "Not available")

                    st.markdown("#### State formation & sovereignty")
                    sovereignty_left, sovereignty_right = st.columns(2)
                    with sovereignty_left:
                        st.markdown("**Former colonial power(s)**")
                        st.write(
                            getattr(
                                historical_profile,
                                "former_colonial_powers",
                                "Not applicable",
                            )
                        )
                        st.markdown("**Colonial period / status**")
                        st.write(
                            getattr(
                                historical_profile,
                                "colonial_period",
                                "Not applicable",
                            )
                        )
                    with sovereignty_right:
                        st.markdown("**Independence / sovereignty date**")
                        st.write(historical_profile.independence_day)
                        st.markdown("**Key independence figure**")
                        st.write(
                            getattr(
                                historical_profile,
                                "independence_leader",
                                "Not applicable",
                            )
                        )

                elif country_section == "Geography":
                    geo_left, geo_right = st.columns([1.35, 0.65], gap="large")
                    with geo_left:
                        if (
                            profile.latitude is not None
                            and profile.longitude is not None
                        ):
                            geography_deck = _build_geography_deck(
                                profile.latitude,
                                profile.longitude,
                                profile.area_km2,
                            )
                            st.pydeck_chart(
                                geography_deck,
                                use_container_width=True,
                                height=500,
                            )
                        else:
                            st.info("Geographic coordinates are not available.")

                    with geo_right:
                        for label, value in (
                            ("Continent", profile.continent),
                            (
                                "Largest cities",
                                getattr(profile, "largest_cities", "Not available"),
                            ),
                            (
                                "Borders",
                                getattr(profile, "borders", "Not available"),
                            ),
                            (
                                "Highest point",
                                getattr(profile, "highest_point", "Not available"),
                            ),
                            (
                                "Lowest point",
                                getattr(profile, "lowest_point", "Not available"),
                            ),
                        ):
                            st.markdown(f"**{label}**")
                            st.write(value)

                    geography_domains = (
                        ("geography", "rivers_lakes", "Rivers, Lakes & Waterways"),
                        ("geography", "mountains_relief", "Mountains & Relief"),
                        ("environment", "climate_seasons", "Climate & Seasons"),
                        (
                            "environment",
                            "natural_resources",
                            "Natural Resources & Raw Materials",
                        ),
                        (
                            "environment",
                            "context",
                            "Environment & Biodiversity",
                        ),
                    )
                    for domain_key, context_key, domain_label in geography_domains:
                        section = intelligence.get(domain_key) or {}
                        context = (
                            section.get(context_key)
                            if isinstance(section, dict)
                            else None
                        )
                        value = (
                            context.get("value")
                            if isinstance(context, dict)
                            else None
                        )
                        if value:
                            st.markdown(f"#### {domain_label}")
                            blocks = _learning_blocks(value)
                            if blocks:
                                for block_heading, summary in blocks:
                                    st.markdown(f"**{block_heading}**")
                                    st.write(summary)
                            else:
                                st.write(value)

                elif country_section == "Government & Society":
                    st.markdown("#### Government & institutions")
                    gov_left, gov_right = st.columns(2, gap="large")
                    with gov_left:
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
                    with gov_right:
                        st.markdown("**International organizations**")
                        st.write(
                            getattr(
                                profile,
                                "international_organizations",
                                "Not available",
                            )
                        )
                        st.markdown("**Emergency number(s)**")
                        st.write(
                            resolve_emergency_numbers(
                                decision_code,
                                getattr(profile, "emergency_numbers", None),
                            )
                        )
                        st.markdown("**Driving side**")
                        st.write(profile.driving_side)

                    society_domains = (
                        ("government", "administrative_divisions", "Administrative Divisions"),
                        ("people_society", "context", "People & Society"),
                        ("people_society", "languages_religion", "Languages & Religion"),
                        ("people_society", "health_system", "Health System & Public Health"),
                        ("culture", "context", "Culture, Cuisine, Music & Sport"),
                        ("culture", "festivals_holidays", "Festivals, Holidays & Traditions"),
                        ("culture", "heritage_landmarks", "Heritage, UNESCO & Major Landmarks"),
                        ("culture", "notable_people", "Notable Public Figures"),
                        ("education_science", "context", "Education, Science & Innovation"),
                        ("international_relations", "context", "International Relations"),
                    )
                    for domain_key, context_key, domain_label in society_domains:
                        section = intelligence.get(domain_key) or {}
                        context = section.get(context_key) if isinstance(section, dict) else None
                        value = context.get("value") if isinstance(context, dict) else None
                        if value:
                            with st.expander(domain_label, expanded=False):
                                blocks = _learning_blocks(value)
                                if blocks:
                                    for block_heading, summary in blocks:
                                        st.markdown(f"**{block_heading}**")
                                        st.write(summary)
                                else:
                                    st.write(value)

                elif country_section == "Economy & Infrastructure":
                    economy_domains = (
                        ("economy", "context", "Economic Structure & Trade"),
                        (
                            "economy",
                            "economic_drivers",
                            "Economic Drivers, Industries & Exports",
                        ),
                        ("infrastructure", "context", "Infrastructure Overview"),
                        (
                            "infrastructure",
                            "transport_network",
                            "Transport Network, Ports & Airports",
                        ),
                        (
                            "infrastructure",
                            "energy_connectivity",
                            "Energy & Connectivity",
                        ),
                    )
                    for domain_key, context_key, domain_label in economy_domains:
                        section = intelligence.get(domain_key) or {}
                        context = section.get(context_key) if isinstance(section, dict) else None
                        value = context.get("value") if isinstance(context, dict) else None
                        if value:
                            st.markdown(f"#### {domain_label}")
                            blocks = _learning_blocks(value)
                            if blocks:
                                for block_heading, summary in blocks:
                                    st.markdown(f"**{block_heading}**")
                                    st.write(summary)
                            else:
                                st.write(value)

                    profile_gdp = getattr(profile, "gdp", None)
                    if (
                        profile_gdp is not None
                        and getattr(profile_gdp, "value_usd", None) is not None
                    ):
                        st.metric(
                            "GDP",
                            f"$ {profile_gdp.value_usd / 1_000_000_000:,.1f}B",
                        )
                        gdp_note = getattr(profile_gdp, "source", "World Bank")
                        if getattr(profile_gdp, "year", None):
                            gdp_note += f" · {profile_gdp.year}"
                        st.caption(gdp_note)

                else:
                    st.markdown("#### Ask Flag Intelligence")
                    question = st.text_input(
                        "Ask a question about this country",
                        placeholder=(
                            "Example: What is the capital? What happened in 1960? "
                            "What does the flag mean?"
                        ),
                        key=f"country-question-{decision_code}",
                    )
                    if question:
                        answers = answer_country_question(
                            question,
                            intelligence,
                            max_results=3,
                        )
                        if answers:
                            for answer in answers:
                                st.markdown(f"**{answer['label']}**")
                                st.write(answer["value"])
                                source_note = (
                                    answer.get("source") or "Source unavailable"
                                )
                                if answer.get("reference_year"):
                                    source_note += f" · {answer['reference_year']}"
                                if answer.get("source_url"):
                                    st.markdown(
                                        f"[{source_note}]({answer['source_url']})"
                                    )
                                else:
                                    st.caption(source_note)
                        else:
                            st.info(
                                "The current verified country record does not "
                                "contain enough information to answer that question."
                            )

                    st.markdown("#### Export")
                    st.caption(
                        "JSON preserves the structured record. PDF produces a "
                        "reader-friendly country report."
                    )

            st.caption(
                "Structured facts: Wikidata / World Bank / REST Countries · "
                "Educational context: Wikipedia/MediaWiki where supported."
            )

        except (requests.RequestException, LookupError, ValueError) as error:
            st.warning("Country information could not be loaded.")
            st.caption(str(error))

    report = {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S UTC"),
        "accepted": accepted,
        "decision": country if accepted else "Unknown",
        "top_candidate": country,
        "country_code": decision_code,
        "input_mode": input_mode_used,
        "confidence": decision_confidence,
        "deployment_threshold": (
            deployment_threshold if input_mode_used == "image" else None
        ),
        "decision_margin": decision_margin,
        "decision_reason": decision_reason,
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
            profile = get_country_profile_v2(
                decision_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
            try:
                historical_profile = get_fresh_historical_profile(
                    decision_code
                )
            except Exception:
                historical_profile = profile

            knowledge = get_country_intelligence_v2(
                decision_code,
                tuple(
                display_country_name(code)
                for code, _ in display_candidates[1:5]
            ),
                schema_version=COUNTRY_INTELLIGENCE_SCHEMA_VERSION,
            )
            report["country_profile"] = knowledge["profile"]
            report["country_intelligence_v2"] = knowledge["intelligence"]
            report["country_intelligence_completion"] = knowledge["completion"]
            report["country_intelligence_validation"] = knowledge["validation"]
            report["official_report_manifest"] = knowledge["report_manifest"]
            report["official_report_missing_required"] = (
                knowledge["missing_required_report_sections"]
            )
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
        missing_required = report.get(
            "official_report_missing_required"
        )
        report_ready = (
            accepted
            and isinstance(missing_required, list)
            and not missing_required
        )

        if accepted and not report_ready:
            missing_text = (
                ", ".join(missing_required)
                if isinstance(missing_required, list)
                and missing_required
                else "Country Intelligence enrichment"
            )
            st.warning(
                "Official PDF withheld: required country knowledge is "
                f"incomplete ({missing_text})."
            )
            st.button(
                "Official PDF unavailable",
                disabled=True,
                use_container_width=True,
            )
        else:
            st.download_button(
                "Download PDF",
                data=build_pdf_report(report, image),
                file_name=(
                    f"{_report_filename_country(country)}_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )


if process and image is not None:
    show_result(image=image)

if text_process:
    resolved_code = direct_selected_code
    if resolved_code is None and typed_country:
        resolved_code = country_code_from_text(typed_country)

    if resolved_code is None:
        st.error(
            "Country not recognized. Choose a country from the list or enter "
            "a valid country name / ISO alpha-2 / ISO alpha-3 code."
        )
    else:
        resolved_name = display_country_name(resolved_code)
        st.query_params.clear()
        st.query_params.update({
            "page": "country",
            "code": resolved_code,
            "country": resolved_name,
            "section": "overview",
        })
        st.rerun()

if active_page == "country":
    route_code = _query_value("code", "").lower()
    if route_code:
        show_result(direct_code=route_code)
    else:
        st.error("Country route is missing a country code.")
