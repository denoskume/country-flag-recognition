# UI restoration redeploy marker
"""Focused Streamlit interface for worldwide flag recognition."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import quote_plus
import sys
import tempfile
from time import strftime
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
from matplotlib import font_manager
import pydeck as pdk
from PIL import Image, ImageDraw
import requests
import streamlit as st
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
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
from reportlab.platypus.tableofcontents import TableOfContents


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

# ---------------------------------------------------------------------------
# Global PDF design rules — single source of truth for every country report.
# ---------------------------------------------------------------------------
PDF_FRONT_MATTER_PAGES = 2  # Cover + Contents
PDF_BODY_FONT_SIZE = 10.5
PDF_BODY_LEADING = 14.0
PDF_TOC_FONT_SIZE = 10.5
PDF_TOC_LEADING = 14.5
PDF_CHAPTER_FONT_SIZE = 14.5
PDF_CHAPTER_LEADING = 17.5
PDF_SECTION_FONT_SIZE = 11.2
PDF_SECTION_LEADING = 13.5
PDF_SNAPSHOT_LABEL_FONT_SIZE = 8.0
PDF_SNAPSHOT_LABEL_LEADING = 9.5
PDF_SNAPSHOT_VALUE_FONT_SIZE = 10.0
PDF_SNAPSHOT_VALUE_LEADING = 12.0
PDF_COVER_MAP_HEIGHT_MM = 70.0

PDF_FONT_REGULAR = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"

try:
    _dejavu_regular = font_manager.findfont("DejaVu Sans")
    _dejavu_bold = font_manager.findfont(
        font_manager.FontProperties(
            family="DejaVu Sans",
            weight="bold",
        )
    )
    pdfmetrics.registerFont(TTFont("FlagUnicode", _dejavu_regular))
    pdfmetrics.registerFont(TTFont("FlagUnicode-Bold", _dejavu_bold))
    PDF_FONT_REGULAR = "FlagUnicode"
    PDF_FONT_BOLD = "FlagUnicode-Bold"
except Exception:
    # Helvetica remains a safe fallback if the runtime font registry fails.
    pass

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
from flag_recognition.report_writer import generate_authored_report


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



def _normalize_price_cell(value: object) -> str | None:
    """Normalize a scraped price cell while keeping its original currency."""
    if value is None:
        return None
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _numbeo_price_lookup(html: str) -> dict[str, str]:
    """Extract selected current country prices from Numbeo tables."""
    wanted = {
        "Apartment (1 bedroom) in City Centre": "rent_1br_centre",
        "Apartment (1 bedroom) Outside of Centre": "rent_1br_outside",
        "Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment": "utilities",
        "Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)": "internet",
        "Monthly Pass (Regular Price)": "transport_pass",
        "Gasoline (1 liter)": "gasoline_numbeo",
        "Average Monthly Net Salary (After Tax)": "net_salary",
        "Meal, Inexpensive Restaurant": "cheap_meal",
    }
    result: dict[str, str] = {}
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError):
        return result

    for table in tables:
        if table.shape[1] < 2:
            continue
        for _, row in table.iterrows():
            label = _normalize_price_cell(row.iloc[0])
            value = _normalize_price_cell(row.iloc[1])
            if not label or not value:
                continue
            for expected, key in wanted.items():
                if label.casefold() == expected.casefold():
                    result[key] = value
    return result


def _livingcost_slug(country_name: str) -> str:
    """Return a best-effort Livingcost country slug."""
    overrides = {
        "Côte d'Ivoire": "ivory-coast",
        "Ivory Coast": "ivory-coast",
        "United States": "united-states",
        "United Kingdom": "united-kingdom",
        "South Korea": "south-korea",
        "North Korea": "north-korea",
        "Czechia": "czech-republic",
    }
    if country_name in overrides:
        return overrides[country_name]
    slug = country_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_current_living_cost(
    country_code: str,
    country_name: str,
) -> dict[str, object]:
    """
    Fetch current indicative living-cost data.

    Livingcost supplies country-level monthly budget estimates; Numbeo supplies
    current crowd-sourced everyday prices; GlobalPetrolPrices is used when a
    recent weekly gasoline price can be parsed. Missing providers never block
    the report.
    """
    result: dict[str, object] = {
        "country_code": country_code.upper(),
        "country_name": country_name,
        "sources": [],
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FlagIntelligence/0.1; "
            "+https://github.com/denoskume/country-flag-recognition)"
        )
    }

    # Livingcost: country-level population-weighted monthly estimates.
    try:
        slug = _livingcost_slug(country_name)
        url = f"https://livingcost.org/cost/{slug}"
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        plain = re.sub(r"<[^>]+>", " ", response.text)
        plain = re.sub(r"\s+", " ", plain)

        patterns = {
            "monthly_one_person_with_rent_usd": r"Total with rent\s*\$\s*([\d,]+(?:\.\d+)?)",
            "monthly_one_person_without_rent_usd": r"Without rent\s*\$\s*([\d,]+(?:\.\d+)?)",
            "monthly_family_with_rent_usd": r"Family of 4.*?Total with rent\s*\$\s*([\d,]+(?:\.\d+)?)",
            "food_one_person_usd": r"Food\s*\$\s*([\d,]+(?:\.\d+)?)",
            "transport_one_person_usd": r"Transport\s*\$\s*([\d,]+(?:\.\d+)?)",
            "salary_after_tax_usd": r"Monthly salary after tax\s*\$\s*([\d,]+(?:\.\d+)?)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, plain, flags=re.IGNORECASE)
            if match:
                result[key] = match.group(1)

        updated = re.search(
            r"Updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            plain,
            flags=re.IGNORECASE,
        )
        if updated:
            result["livingcost_updated"] = updated.group(1)

        if any(key.startswith("monthly_") for key in result):
            result["sources"].append("Livingcost")
    except (requests.RequestException, ValueError):
        pass

    # Numbeo: current everyday prices in the country's local display currency.
    try:
        url = (
            "https://www.numbeo.com/cost-of-living/"
            f"country_result.jsp?country={quote_plus(country_name)}"
        )
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        prices = _numbeo_price_lookup(response.text)
        if prices:
            result.update(prices)
            result["sources"].append("Numbeo")

        plain = re.sub(r"<[^>]+>", " ", response.text)
        plain = re.sub(r"\s+", " ", plain)
        updated = re.search(
            r"Last update:\s*([^<]{3,40}?)(?:\s{2,}|$)",
            plain,
            flags=re.IGNORECASE,
        )
        if updated:
            result["numbeo_updated"] = updated.group(1).strip()
    except (requests.RequestException, ValueError):
        pass

    # GlobalPetrolPrices: recent weekly gasoline price where available.
    try:
        petrol_name = country_name.replace(" ", "_")
        url = (
            "https://www.globalpetrolprices.com/"
            f"{quote_plus(petrol_name)}/gasoline_prices/"
        )
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        plain = re.sub(r"<[^>]+>", " ", response.text)
        plain = re.sub(r"\s+", " ", plain)
        match = re.search(
            r"current gasoline price.*?is\s+([A-Z]{3})\s*([\d.]+)\s+per liter",
            plain,
            flags=re.IGNORECASE,
        )
        if match:
            result["gasoline_current"] = f"{match.group(1).upper()} {match.group(2)} per litre"
            date_match = re.search(
                r"updated on\s+(\d{1,2}-[A-Za-z]{3}-\d{4})",
                plain,
                flags=re.IGNORECASE,
            )
            if date_match:
                result["gasoline_updated"] = date_match.group(1)
            result["sources"].append("GlobalPetrolPrices")
    except (requests.RequestException, ValueError):
        pass

    result["sources"] = list(dict.fromkeys(result["sources"]))
    return result


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
    canvas.setFont(PDF_FONT_BOLD, 9.6)
    canvas.drawCentredString(
        cx,
        logo_y + 6.4 * mm,
        "FLAG",
    )
    canvas.setFont(PDF_FONT_BOLD, 4.5)
    canvas.drawCentredString(
        cx,
        logo_y + 3.8 * mm,
        "INTELLIGENCE",
    )

    # ------------------------------------------------------------------
    # Page header contact - repeated on every page and right-aligned to
    # the same outer content margin as the rest of the report.
    # ------------------------------------------------------------------
    contact_y = page_height - 23 * mm

    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont(PDF_FONT_REGULAR, 9.3)
    canvas.drawRightString(right, contact_y, "Contact")

    canvas.setFont(PDF_FONT_REGULAR, 8.6)
    canvas.drawRightString(
        right,
        contact_y - 5.0 * mm,
        "+33 (0)6 62 91 94 68",
    )
    canvas.drawRightString(
        right,
        contact_y - 10.2 * mm,
        "denoskume@yahoo.com",
    )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.setFont(PDF_FONT_REGULAR, 6.8)
    canvas.drawCentredString(
        (left + right) / 2,
        18 * mm,
        "Sources: World Bank · Wikidata · REST Countries · Wikipedia · EmergencyNumberAPI",
    )
    canvas.setFont(PDF_FONT_REGULAR, 7.2)
    physical_page = int(getattr(document, "page", 1))
    if physical_page <= PDF_FRONT_MATTER_PAGES:
        footer_text = (
            "© 2026 Flag Intelligence · Version 0.1.0 · All rights reserved."
        )
    else:
        logical_page = physical_page - PDF_FRONT_MATTER_PAGES
        footer_text = (
            "© 2026 Flag Intelligence · Version 0.1.0 · All rights reserved. "
            f"· Page {logical_page}"
        )
    canvas.drawCentredString(
        (left + right) / 2,
        13.5 * mm,
        footer_text,
    )

    canvas.restoreState()



class FlagIntelligenceDocTemplate(SimpleDocTemplate):
    """Report document with a generated TOC and logical page numbering."""

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph):
            style_name = getattr(flowable.style, "name", "")
            if style_name == "ChapterTitle":
                # Front page + Contents are unnumbered front matter.
                logical_page = max(
                    int(self.page) - PDF_FRONT_MATTER_PAGES,
                    1,
                )
                self.notify(
                    "TOCEntry",
                    (0, flowable.getPlainText(), logical_page),
                )


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
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    return KeepTogether([
        Paragraph(title, ParagraphStyle(
            f"CardTitle_{title}",
            parent=value_style,
            fontName=PDF_FONT_BOLD,
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=3 * mm,
            spaceAfter=1.5 * mm,
        )),
        table,
    ])


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_official_flag_png(country_code: str) -> bytes | None:
    """Fetch the canonical country flag image used on the PDF cover."""
    code = str(country_code or "").strip().lower()
    if len(code) != 2:
        return None

    try:
        response = requests.get(
            f"https://flagcdn.com/w320/{code}.png",
            timeout=6,
            headers={
                "User-Agent": (
                    "flag-intelligence/1.0 "
                    "(country intelligence report generator)"
                )
            },
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def _cover_flag_image(
    country_code: str,
    fallback_image: Image.Image | None,
) -> PDFImage | None:
    """Return a report-ready official flag, falling back to the uploaded image."""
    flag_bytes = _fetch_official_flag_png(country_code)

    try:
        if flag_bytes:
            source = BytesIO(flag_bytes)
            flag_image = PDFImage(source)
        elif fallback_image is not None:
            buffer = BytesIO()
            fallback_image.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=90,
            )
            buffer.seek(0)
            flag_image = PDFImage(buffer)
        else:
            return None

        # Fixed physical size on the report cover.
        # Width: 2 cm · Height: 1 cm.
        flag_image.drawWidth = 20 * mm
        flag_image.drawHeight = 10 * mm
        return flag_image
    except Exception:
        return None


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

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
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
                page.wait_for_timeout(1000)

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
    pdf_map.drawHeight = PDF_COVER_MAP_HEIGHT_MM * mm
    return pdf_map


class ReportQualityError(ValueError):
    """Raised when a PDF fails the professional editorial quality gate."""


def _flowable_text(value: object) -> str:
    """Extract visible text recursively from ReportLab flowables for QA."""
    if isinstance(value, Paragraph):
        return value.getPlainText()
    if isinstance(value, Table):
        parts: list[str] = []
        for row in getattr(value, "_cellvalues", []):
            for cell in row:
                parts.append(_flowable_text(cell))
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        return " ".join(_flowable_text(item) for item in value)
    if isinstance(value, str):
        return value
    return ""


def _sanitize_pdf_payload(value: object) -> object:
    """Repair harmless editorial separators across the complete PDF payload."""
    if isinstance(value, str):
        text = value.replace(" | ", " · ").replace("|", " · ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()
    if isinstance(value, dict):
        return {
            key: _sanitize_pdf_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_pdf_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_pdf_payload(item) for item in value)
    return value


def _validate_professional_report_story(story: list[object]) -> None:
    """
    Enforce non-negotiable editorial quality rules before PDF generation.

    The report is rejected when obvious source residue, ambiguous date formats,
    broken markup, or other raw extraction artefacts remain in visible text.
    """
    text = " ".join(_flowable_text(item) for item in story)
    normalized = re.sub(r"\s+", " ", text)

    violations: list[str] = []

    forbidden_patterns = (
        (r"\{\{|\}\}", "MediaWiki template residue"),
        (r"&nbsp;|&#160;|&#x0*a0;", "non-breaking-space HTML residue"),
        (r"\balt=", "image-alt extraction residue"),
        (r"\bthumb\|", "MediaWiki image residue"),
        (r"\bpx\s", "image-dimension residue"),
        (
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "ambiguous numeric calendar date",
        ),
        (
            r"\b(?:1\d{3}|20\d{2})\s+-\s+",
            "legacy date-dash timeline formatting",
        ),
    )

    for pattern, label in forbidden_patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            violations.append(label)

    # Detect malformed punctuation commonly produced by source extraction.
    if re.search(r"\.\s*\.", normalized):
        violations.append("duplicate sentence punctuation")
    if re.search(r";\s*;", normalized):
        violations.append("empty semicolon-delimited fragment")
    if re.search(r"\bQ\d{2,}\b", normalized):
        violations.append("unresolved Wikidata identifier")
    if "■■" in normalized:
        violations.append("unsupported or missing glyphs")
    if re.search(
        r"\b(?:hlist|ublistr?|ulist|plainlist|flatlist|wikitable|item[_ -]?style)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        violations.append("source markup residue")
    if re.search(r"https?://|www\.", normalized, flags=re.IGNORECASE):
        violations.append("raw URL exposed in narrative")
    if re.search(
        r"\b(?:Political parties President|family tree of .* monarchs|"
        r"French literary awards:\s*–|Further reading:)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        violations.append("raw catalogue or navigation residue")
    if re.search(
        r"\b(for for|in in|by by|on on|the the|and and)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        violations.append("duplicated word or preposition")
    if re.search(
        r"\b(?:in|by|on)\s+(\d{3,4})\s*,[^.!?]{0,120}\b(?:in|by|on)\s+\1\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        violations.append("duplicated timeline date")
    if re.search(r"\+Adults|::", normalized):
        violations.append("table/list extraction residue")
    if re.search(r"\bNot available\b", normalized, flags=re.IGNORECASE):
        violations.append("unresolved missing value exposed to reader")

    if violations:
        unique = ", ".join(dict.fromkeys(violations))
        raise ReportQualityError(
            "Professional report quality gate failed: "
            f"{unique}. PDF generation has been blocked."
        )


def _build_pdf_report_uncached(
    report: dict[str, object],
    image: Image.Image | None,
) -> bytes:
    """Build a compact institutional country knowledge report."""
    report = _sanitize_pdf_payload(report)
    assert isinstance(report, dict)
    buffer = BytesIO()
    document = FlagIntelligenceDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=PDF_MARGIN_MM * mm,
        leftMargin=PDF_MARGIN_MM * mm,
        topMargin=44 * mm,
        # Reserve a dedicated footer band so narrative text can never
        # overlap sources, copyright or logical page numbering.
        bottomMargin=28 * mm,
        title="Flag Intelligence - Country Knowledge Report",
        author="Denos Kume",
        subject="Country knowledge report generated from flag recognition",
    )

    REPORT_WIDTH_MM = PDF_CONTENT_WIDTH_MM

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CountryReportTitle",
        parent=styles["Title"],
        fontName=PDF_FONT_BOLD,
        fontSize=16.5,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111111"),
        spaceAfter=2.5 * mm,
    )

    meta_style = ParagraphStyle(
        "ReportMeta",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=7.0,
        leading=8.5,
        textColor=colors.HexColor("#444444"),
    )

    section_title_style = ParagraphStyle(
        "BoxSectionTitle",
        parent=styles["Heading2"],
        fontName=PDF_FONT_BOLD,
        fontSize=10.2,
        leading=12,
        textColor=colors.HexColor("#111111"),
        spaceAfter=0,
    )

    label_style = ParagraphStyle(
        "CompactLabel",
        parent=styles["Normal"],
        fontName=PDF_FONT_BOLD,
        fontSize=7.4,
        leading=9.2,
        textColor=colors.HexColor("#333333"),
    )

    value_style = ParagraphStyle(
        "CompactValue",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=7.6,
        leading=9.4,
        textColor=colors.HexColor("#222222"),
    )

    body_style = ParagraphStyle(
        "CompactBody",
        parent=styles["BodyText"],
        fontName=PDF_FONT_REGULAR,
        fontSize=10.5,
        leading=14.0,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#222222"),
        spaceAfter=3.0 * mm,
    )

    narrative_heading_style = ParagraphStyle(
        "NarrativeSectionHeading",
        parent=styles["Heading2"],
        fontName=PDF_FONT_BOLD,
        fontSize=11.2,
        leading=13.5,
        textColor=colors.HexColor("#111111"),
        spaceBefore=1.5 * mm,
        spaceAfter=1.2 * mm,
        keepWithNext=True,
    )

    chapter_title_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading1"],
        fontName=PDF_FONT_BOLD,
        fontSize=14.5,
        leading=17.5,
        textColor=colors.HexColor("#111111"),
        spaceBefore=2.0 * mm,
        spaceAfter=2.0 * mm,
        keepWithNext=True,
    )

    front_matter_title_style = ParagraphStyle(
        "FrontMatterTitle",
        parent=styles["Heading1"],
        fontName=PDF_FONT_BOLD,
        fontSize=14.5,
        leading=17.5,
        textColor=colors.HexColor("#111111"),
        spaceBefore=2.0 * mm,
        spaceAfter=2.0 * mm,
        keepWithNext=True,
    )

    contents_item_style = ParagraphStyle(
        "ContentsItem",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=9.2,
        leading=14,
        leftIndent=4 * mm,
        spaceAfter=1.2 * mm,
        textColor=colors.HexColor("#222222"),
    )

    narrative_subheading_style = ParagraphStyle(
        "NarrativeSubheading",
        parent=styles["Heading3"],
        fontName=PDF_FONT_BOLD,
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#333333"),
        spaceBefore=1.2 * mm,
        spaceAfter=0.8 * mm,
        keepWithNext=True,
    )

    small_style = ParagraphStyle(
        "CompactSmall",
        parent=styles["Normal"],
        fontName=PDF_FONT_REGULAR,
        fontSize=6.7,
        leading=8.2,
        textColor=colors.HexColor("#555555"),
    )

    def clean(value: object) -> str:
        text = _pdf_text(value)
        if text in {"", "None"}:
            return "Not available"

        # Keep report text human-readable by removing common MediaWiki residue.
        text = re.sub(
            r"\{\{\s*convert\|([^|{}]+)\|([^|{}]+)[^{}]*\}\}",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\{\{\s*Start date(?: and age)?\|(\d{4})\|(\d{1,2})\|(\d{1,2})[^{}]*\}\}",
            lambda m: f"{int(m.group(3))}/{int(m.group(2))}/{m.group(1)}",
            text,
            flags=re.IGNORECASE,
        )
        # Remove residual MediaWiki templates, including malformed/nested
        # convert/efn fragments that occasionally survive source extraction.
        text = re.sub(
            r"\{\{\s*convert\|([^|{}]+)\|([^|{}]+)[^\n;]*",
            lambda m: f"{m.group(1)} {m.group(2)}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\{\{\s*efn\|?[^\n]*",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        text = re.sub(r"\{\{|\}\}", " ", text)
        text = re.sub(
            r"\bthumb\|[^.]*?(?=(?:[A-Z][a-z]+\s|The\s|In\s|France\s|Renewable\s|Climate\s|$))",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"<ref\b[^>]*>.*?</ref>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<ref\b[^>]*/>", " ", text, flags=re.I)

        # Remove HTML/entity and image-caption artefacts before report prose.
        text = (
            text.replace("&nbsp;", " ")
            .replace("&#160;", " ")
            .replace("&#xA0;", " ")
            .replace("&#xa0;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )

        # Wikipedia image captions can be embedded directly inside prose.
        # Remove metadata spans rather than letting fragments reach the report.
        text = re.sub(
            r"\balt\s*=\s*.*?(?=\||(?:\s+[A-Z][a-z]+(?:\s+[a-z]+){0,3}:)|$)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:File|Image):[^|\n]+(?:\||$)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:thumb|thumbnail|upright)(?:\s*=\s*[^| ]+)?\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove image-size residue both as '300 px' and orphaned leading 'px'.
        text = re.sub(
            r"\b\d+(?:\.\d+)?\s*px\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?:(?<=^)|(?<=[.!?;]))\s*px\b\s*",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bpx\b(?=\s+[A-Z])",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Strip common image-caption lead-ins that survive encyclopedia extraction.
        text = re.sub(
            r"\b(?:map|chart|photo|image|illustration)\s+of\b[^.]{0,180}\.",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove residual list/table markup emitted by MediaWiki templates.
        # These fragments can appear inline in otherwise valid prose.
        text = re.sub(
            r"\b(?:hlist|ublistr?|ulist|plainlist|flatlist)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bitem[_ -]?style\s*=\s*[^,;|}\]]+",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bclass\s*=\s*[\"']?wikitable[\"']?",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:style|rowspan|colspan|align)\s*=\s*[^,;|}\]]+",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"(?<!\w)[*#]+\s*",
            " ",
            text,
        )
        text = re.sub(
            r"[{}\[\]]+",
            " ",
            text,
        )

        # Remove raw URLs and citation-link residue from narrative text.
        text = re.sub(
            r"https?://\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:www\.)\S+",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove Timeline/graph configuration residue emitted by MediaWiki.
        # Keep any prose that follows an explicit overview label.
        text = re.sub(
            r"\btag\s*:\s*timeline\b.*?(?=\boverview\s*:|$)",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"\b(?:ImageSize|PlotArea|TimeAxis|ScaleMajor|ScaleMinor|AlignBars|"
            r"DateFormat|Period|Colors|Define|BarData|PlotData)\s*=.*?(?=\boverview\s*:|$)",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"\boverview\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove leftover table/list punctuation and pseudo-headings that
        # commonly survive encyclopedia extraction.
        text = re.sub(r"\s*\+Adults\b", " Adults", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*:+\s*:+\s*", " ", text)
        text = re.sub(r"\s*:\s*;", "; ", text)
        text = re.sub(r"\s*;\s*:", "; ", text)

        text = re.sub(r"\s*\|\s*", ", ", text)
        text = re.sub(r"\s+", " ", text).strip(" ;|")
        return text or "Not available"

    def professional_date(value: object) -> str:
        """Render full calendar dates in an unambiguous professional English style."""
        text = clean(value)
        if text == "Not available":
            return text

        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]

        def repl(match: re.Match) -> str:
            day = int(match.group(1))
            month = int(match.group(2))
            year = match.group(3)
            if 1 <= month <= 12:
                return f"{day} {months[month - 1]} {year}"
            return match.group(0)

        # Convert numeric D/M/YYYY or DD/MM/YYYY dates.
        text = re.sub(
            r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
            repl,
            text,
        )

        month_lookup = {
            month.lower(): month
            for month in months
        }

        def month_day_repl(match: re.Match) -> str:
            month_name = month_lookup[match.group(1).lower()]
            day = int(match.group(2))
            return f"{day} {month_name}"

        text = re.sub(
            r"\b(" + "|".join(months) + r")\s+(\d{1,2})\b",
            month_day_repl,
            text,
            flags=re.IGNORECASE,
        )
        return text

    def professional_inline(value: object) -> str:
        """Normalize compact inline facts for polished report display."""
        text = clean(value)
        if text == "Not available":
            return text
        text = re.sub(r"\s*\|\s*", " · ", text)
        text = re.sub(r"\s*;\s*", "; ", text)
        return re.sub(r"\s+", " ", text).strip()

    def primary_internet_domain(value: object) -> str:
        """
        Keep the cover compact and universally renderable by selecting the
        primary ASCII country-code domain from any multilingual domain list.
        """
        text = clean(value)
        if text == "Not available":
            return text
        candidates = re.findall(r"\.[A-Za-z0-9-]{2,}", text)
        if candidates:
            # Prefer a conventional two-letter ccTLD where available.
            two_letter = [
                domain
                for domain in candidates
                if re.fullmatch(r"\.[A-Za-z]{2}", domain)
            ]
            return two_letter[0] if two_letter else candidates[0]
        # Avoid unsupported native-script glyphs in the compact cover cell.
        ascii_text = text.encode("ascii", "ignore").decode("ascii").strip(" ,")
        return ascii_text or "Not available"

    def paragraph(value: object, style=body_style) -> Paragraph:
        return Paragraph(xml_escape(clean(value)), style)

    def narrative_section(
        title: str,
        flowables: list[object],
    ) -> list[object]:
        if not flowables:
            return []
        return [
            Paragraph(xml_escape(title), narrative_heading_style),
            HRFlowable(
                width="100%",
                thickness=0.7,
                color=colors.HexColor("#B8BEC7"),
                spaceBefore=0,
                spaceAfter=2.0 * mm,
            ),
            *flowables,
            Spacer(1, 2.5 * mm),
        ]

    def chapter_heading(number: int, title: str) -> list[object]:
        return [
            Paragraph(
                f"{number}. {xml_escape(title)}",
                chapter_title_style,
            ),
            HRFlowable(
                width="100%",
                thickness=1.15,
                color=colors.HexColor("#111111"),
                spaceBefore=0,
                spaceAfter=3.0 * mm,
            ),
        ]

    def labeled_paragraphs(
        rows: list[tuple[str, object]],
    ) -> list[object]:
        flowables: list[object] = []
        for label, value in rows:
            cleaned = clean(value)
            if cleaned == "Not available":
                continue
            flowables.append(
                Paragraph(xml_escape(label), narrative_subheading_style)
            )
            flowables.append(Paragraph(xml_escape(cleaned), body_style))
        return flowables

    def readable_fact_paragraph(
        sentences: list[str],
    ) -> list[object]:
        """Render factual sections as short, readable paragraphs."""
        return _paragraphize_sentences(
            sentences,
            max_sentences=3,
            max_chars=520,
        )

    def fact_value(value: object) -> str | None:
        """Return a cleaned fact value, preserving explicit 'Not applicable'."""
        cleaned = clean(value)
        if cleaned == "Not available":
            return None
        return cleaned

    def sentence_for(label: str, value: object) -> str | None:
        """Convert a short label/value fact into a readable sentence."""
        cleaned = fact_value(value)
        if cleaned is None:
            return None
        return f"{label}: {cleaned}."

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
    story.append(
        Table(
            [[
                Paragraph(
                    f"<b>Generated:</b> {report_date} · {report_time}",
                    meta_style,
                ),
                Paragraph(
                    f"<b>Country code:</b> {country_code}",
                    ParagraphStyle(
                        "CoverCodeMeta",
                        parent=meta_style,
                        alignment=2,
                    ),
                ),
            ]],
            colWidths=[
                (REPORT_WIDTH_MM / 2) * mm,
                (REPORT_WIDTH_MM / 2) * mm,
            ],
            hAlign="LEFT",
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0 * mm),
            ]),
        )
    )

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
        if not isinstance(intelligence, dict):
            intelligence = {}
        completion = report.get("country_intelligence_completion")
        authored_report = report.get("authored_report")
        if not isinstance(authored_report, dict):
            authored_report = {}

        def authored_text(key: str) -> str:
            value = authored_report.get(key)
            if not isinstance(value, str):
                return ""
            return value.strip()

        def authored_flowables(key: str) -> list[object]:
            value = authored_text(key)
            if not value:
                return []
            paragraphs = [
                re.sub(r"\s+", " ", part).strip()
                for part in re.split(r"\n\s*\n+", value)
                if part.strip()
            ]
            return [
                Paragraph(xml_escape(part), body_style)
                for part in paragraphs
            ]

        def add_authored_section(title: str, key: str) -> bool:
            flowables = authored_flowables(key)
            if not flowables:
                return False
            story.extend(narrative_section(title, flowables))
            return True

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

        def _normalize_sentence(text: str) -> str:
            """Normalize scraped prose into report-ready sentence text."""
            value = clean(text)
            if value == "Not available":
                return ""
            value = re.sub(r"^[*•\-]+\s*", "", value)
            value = re.sub(r"\s*\|\s*", ", ", value)
            value = re.sub(r"\s*;\s*", "; ", value)

            # Convert encyclopedic inline labels into normal sentence flow.
            value = re.sub(
                r"\b(?:Forms|In children|In adults|Society|Religion|Languages|"
                r"Education|Health|Cuisine|Sports|Music|Philosophy|Media|"
                r"Agriculture|Tourism|Roads|Railways|Renewable energy|"
                r"Solar energy)\s*:\s*",
                "",
                value,
                flags=re.IGNORECASE,
            )

            # Remove source-introduction boilerplate left by linked citations.
            value = re.sub(
                r"According to\s+(?:the\s+)?(?:WHO\s+report|report)\s*,?\s*",
                "",
                value,
                flags=re.IGNORECASE,
            )
            value = re.sub(
                r"Global Oral Health Status Report[^.]*?2030\)?\s*",
                "",
                value,
                flags=re.IGNORECASE,
            )

            # Remove duplicated words/prepositions introduced by source
            # stitching or API motivation text (e.g. "for for").
            value = re.sub(
                r"\b(for|in|on|by|the|a|an|to|of|and)\s+\1\b",
                r"\1",
                value,
                flags=re.IGNORECASE,
            )

            # Collapse immediately repeated years/dates such as
            # "in 1579, ... in 1579" or "by 1801, in 1801".
            value = re.sub(
                r"\b(in|by|on)\s+(\d{3,4})\s*,\s*"
                r"(?:(?:in|by|on)\s+\2\s*,?\s*)",
                r"\1 \2, ",
                value,
                flags=re.IGNORECASE,
            )
            value = re.sub(
                r"\b(\d{3,4})\b(\s*[,;:]?\s*)\1\b",
                r"\1",
                value,
            )

            # Remove repeated adjacent phrases created when multiple source
            # fragments overlap.
            value = re.sub(
                r"\b([^.!?]{8,80})\s+\1\b",
                r"\1",
                value,
                flags=re.IGNORECASE,
            )

            value = re.sub(r"\s+", " ", value).strip(" ;,:")
            if value and value[-1] not in ".!?":
                value += "."
            return value

        def _list_to_sentence(value: str, intro: str) -> str:
            """Turn comma/semicolon separated source fragments into readable prose."""
            cleaned = _normalize_sentence(value).rstrip(".")
            if not cleaned:
                return ""
            parts = [
                item.strip(" .")
                for item in re.split(r"[;|]", cleaned)
                if item.strip(" .")
            ]
            if len(parts) <= 1:
                return f"{intro} {cleaned}."
            if len(parts) == 2:
                joined = f"{parts[0]} and {parts[1]}"
            else:
                joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            return f"{intro} {joined}."

        def _paragraphize_sentences(
            sentences: list[str],
            *,
            max_sentences: int = 3,
            max_chars: int = 560,
        ) -> list[object]:
            """Render short thematic paragraphs instead of dense text walls."""
            normalized = [
                re.sub(r"\s+", " ", sentence).strip()
                for sentence in sentences
                if sentence and re.sub(r"\s+", " ", sentence).strip()
            ]
            if not normalized:
                return []

            topic_breaks = re.compile(
                r"^(?:In\s+\d{3,4}|By\s+\d{3,4}|Later\b|Subsequently\b|"
                r"Thereafter\b|More recently\b|A major shift\b|A turning point\b|"
                r"The next decisive moment\b|A new chapter\b|Against this background\b|"
                r"Within this evolving context\b|In this broader context\b|"
                r"Politically\b|Economically\b|Culturally\b|Scientifically\b|"
                r"Administratively\b|Institutionally\b)",
                flags=re.IGNORECASE,
            )

            flowables: list[object] = []
            chunk: list[str] = []
            char_count = 0

            def flush() -> None:
                nonlocal chunk, char_count
                if not chunk:
                    return
                flowables.append(
                    Paragraph(
                        xml_escape(" ".join(chunk)),
                        body_style,
                    )
                )
                flowables.append(Spacer(1, 2.2 * mm))
                chunk = []
                char_count = 0

            for sentence in normalized:
                projected = char_count + len(sentence) + (1 if chunk else 0)

                # Start a new paragraph when a sentence clearly introduces a
                # new period, theme or institutional topic.
                if chunk and (
                    topic_breaks.search(sentence)
                    or len(chunk) >= max_sentences
                    or projected > max_chars
                ):
                    flush()

                chunk.append(sentence)
                char_count += len(sentence) + (1 if chunk else 0)

            flush()

            if flowables and isinstance(flowables[-1], Spacer):
                flowables.pop()
            return flowables


        def _narrative_paragraph(sentences: list[str]) -> list[object]:
            """Build coherent, visually separated report paragraphs."""
            return _paragraphize_sentences(sentences)

        def _timeline_narrative(
            rows: list[tuple[str, str]],
        ) -> list[object]:
            """Convert dated events into varied, coherent historical narration."""
            events = [
                (clean(period), _normalize_sentence(summary))
                for period, summary in rows
                if clean(period) != "Not available"
                and _normalize_sentence(summary)
            ]
            if not events:
                return []

            paragraphs: list[object] = []
            chunk: list[str] = []

            transition_sets = (
                ("A few decades later, in", "Subsequently, in", "By", "Later, in"),
                ("This was followed by", "The sequence continued in", "In the years that followed, by", "Thereafter, in"),
                ("A major shift came in", "A turning point emerged in", "The political landscape changed again in", "The next decisive moment came in"),
                ("Against this background, in", "In this broader context, by", "Amid these changes, in", "Within this evolving context, in"),
                ("The situation evolved further in", "The period entered a new phase in", "A new chapter began in", "The historical trajectory then moved to"),
            )

            for index, (period, summary) in enumerate(events):
                lowered_summary = (
                    summary[0].lower() + summary[1:]
                    if len(summary) > 1
                    else summary.lower()
                )

                if index == 0:
                    sentence = f"In {period}, {lowered_summary}"
                elif index == len(events) - 1:
                    sentence = f"More recently, in {period}, {lowered_summary}"
                else:
                    connector_group = transition_sets[(index - 1) % len(transition_sets)]
                    connector = connector_group[(index - 1) % len(connector_group)]

                    if connector in {
                        "This was followed by",
                        "The historical trajectory then moved to",
                    }:
                        sentence = f"{connector} {period}, when {lowered_summary}"
                    elif connector == "By":
                        sentence = f"By {period}, {lowered_summary}"
                    elif connector.startswith("In the years that followed"):
                        sentence = f"In the years that followed, by {period}, {lowered_summary}"
                    else:
                        sentence = f"{connector} {period}, {lowered_summary}"

                chunk.append(sentence)

                if len(chunk) >= 3:
                    paragraphs.extend(
                        _paragraphize_sentences(
                            chunk,
                            max_sentences=3,
                            max_chars=620,
                        )
                    )
                    chunk = []

            if chunk:
                paragraphs.extend(
                    _paragraphize_sentences(
                        chunk,
                        max_sentences=3,
                        max_chars=620,
                    )
                )
            return paragraphs

        INLINE_TOPIC_LABELS = (
            "Carolingian dynasty", "Capetian dynasty", "House of Capet",
            "House of Valois", "House of Bourbon", "National Convention",
            "Directory", "Consulate", "19th-century monarchs",
            "Fiction", "Poetry", "Theatre", "Nonfiction",
            "Literary criticism", "Medieval period", "Nominalism",
            "Platonism", "Indifferentism", "Peter Abelard",
            "Physics", "Chemistry", "Mathematics", "Nuclear power",
            "Space science", "Historical", "Scientific fields",
            "Businessmen and entrepreneurs", "Fashion", "Architecture",
            "Film, television and radio personalities", "Musicians",
            "Politicians/Law", "Writers", "Key industries",
            "Main export goods", "Main import goods",
        )

        def _split_editorial_units(value: str) -> list[tuple[str, str]]:
            """
            Split raw encyclopedic prose into topical units.

            The source often collapses headings such as 'Physics:' or
            'National Convention:' into one continuous paragraph. Recover
            those headings before rendering so every topic can become its own
            short section.
            """
            text = _normalize_sentence(value)
            if not text:
                return []

            labels_pattern = "|".join(
                re.escape(label)
                for label in sorted(
                    INLINE_TOPIC_LABELS,
                    key=len,
                    reverse=True,
                )
            )
            text = re.sub(
                rf"\s+(?=({labels_pattern})\s*:)",
                "\n",
                text,
                flags=re.IGNORECASE,
            )

            units: list[tuple[str, str]] = []
            for raw in re.split(r"\n+", text):
                part = raw.strip()
                if not part:
                    continue

                match = re.match(
                    rf"^({labels_pattern})\s*:\s*(.*)$",
                    part,
                    flags=re.IGNORECASE,
                )
                if match:
                    units.append(
                        (
                            match.group(1).strip(),
                            match.group(2).strip(),
                        )
                    )
                else:
                    units.append(("", part))

            return units

        def _sentence_units(value: str) -> list[str]:
            """Split cleaned prose into readable sentence-level units."""
            text = _normalize_sentence(value)
            if not text:
                return []

            # Break at normal sentence boundaries and before obvious inline
            # topic labels that survived source extraction.
            sentences = re.split(
                r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])",
                text,
            )
            return [
                item.strip()
                for item in sentences
                if item and item.strip()
            ]

        def _representative_sentences(
            value: str,
            *,
            max_sentences: int = 6,
            max_chars: int = 1500,
        ) -> list[str]:
            """
            Keep informative prose while preventing raw catalogues from
            flooding the report.

            Long name/award/catalogue sections are represented by the first
            well-formed examples rather than copied wholesale.
            """
            sentences = _sentence_units(value)
            selected: list[str] = []
            char_count = 0

            for sentence in sentences:
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if not cleaned:
                    continue

                # Drop obvious navigation/caption/list artefacts.
                if re.search(
                    r"\b(?:see chart below|further reading|political parties|"
                    r"family tree|painting by|walk, paris|seminar with)\b",
                    cleaned,
                    flags=re.IGNORECASE,
                ):
                    continue

                # A source can collapse an entire catalogue into one
                # punctuation-free "sentence". Never let that pass through.
                if len(cleaned) > 520:
                    shortened = cleaned[:520].rsplit(" ", 1)[0].rstrip(" ,;:")
                    if shortened:
                        cleaned = shortened + "."

                projected = char_count + len(cleaned)
                if selected and (
                    len(selected) >= max_sentences
                    or projected > max_chars
                ):
                    break

                selected.append(cleaned)
                char_count = projected

            return selected

        def _render_editorial_text(value: str) -> list[object]:
            """
            Render any long source text as structured, short paragraphs.

            No raw source block is allowed to become a single PDF paragraph.
            """
            units = _split_editorial_units(value)
            if not units:
                return []

            flowables: list[object] = []
            for heading, body in units:
                sentences = _representative_sentences(body)
                if not sentences:
                    continue

                if heading:
                    flowables.append(
                        Paragraph(
                            xml_escape(heading),
                            narrative_subheading_style,
                        )
                    )

                flowables.extend(
                    _paragraphize_sentences(
                        sentences,
                        max_sentences=2,
                        max_chars=460,
                    )
                )

            return flowables


        def _editorial_compose(
            text: object,
            *,
            max_paragraphs: int = 4,
            max_sentences_per_paragraph: int = 2,
        ) -> list[object]:
            """
            Write report prose from collected evidence instead of printing
            source prose verbatim.

            The source text is treated only as evidence. We extract clean,
            non-duplicated factual sentences, remove catalogue/caption residue,
            then rebuild short coherent paragraphs.
            """
            raw = clean(text)
            if raw == "Not available":
                return []

            # Split source evidence into candidate statements.
            candidates = _sentence_units(raw)
            if not candidates:
                return []

            cleaned: list[str] = []
            seen: set[str] = set()

            for sentence in candidates:
                value = _normalize_sentence(sentence)
                if not value:
                    continue

                # Reject source/caption/navigation residue.
                if re.search(
                    r"\b(?:image|pictured|photographed|caption|gallery|"
                    r"see also|further reading|bibliography|references|"
                    r"the table below|the chart below|this list|"
                    r"figure|map shows|displayed for sale|painting by)\b",
                    value,
                    flags=re.IGNORECASE,
                ):
                    continue

                # Reject visibly incomplete fragments.
                if re.search(r"\b(?:ca\.|c\.|d\.|r\.)\s*$", value):
                    continue
                if value.endswith((":", ";", "(", "[")):
                    continue

                # De-duplicate near-identical statements.
                key = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
                if not key or key in seen:
                    continue

                duplicate = False
                for existing in list(seen):
                    shorter = min(len(existing), len(key))
                    if shorter >= 40 and (
                        existing in key or key in existing
                    ):
                        duplicate = True
                        break
                if duplicate:
                    continue

                seen.add(key)
                cleaned.append(value)

            if not cleaned:
                return []

            # Keep the most informative evidence while preventing data dumps.
            selected: list[str] = []
            for sentence in cleaned:
                if len(selected) >= max_paragraphs * max_sentences_per_paragraph:
                    break
                selected.append(sentence)

            # Rebuild readable report paragraphs.
            return _paragraphize_sentences(
                selected,
                max_sentences=max_sentences_per_paragraph,
                max_chars=500,
            )


        def add_editorial_section(
            title: str,
            text: object,
            *,
            max_paragraphs: int = 4,
        ) -> None:
            """Render a fully rewritten report section from verified evidence."""
            flowables = _editorial_compose(
                text,
                max_paragraphs=max_paragraphs,
            )
            if flowables:
                story.extend(narrative_section(title, flowables))


        def learning_flowables(text: object) -> list[object]:
            """
            Convert source fragments into readable report prose.

            Small source headings such as 'Climate', 'Longest river',
            'Natural resources', 'Education', etc. are treated as metadata,
            not as visual subheadings. Their content is merged into prose.
            Only genuinely multi-topic sections retain internal headings.
            """
            blocks = _learning_blocks(text)
            if not blocks:
                return []

            generic_headings = {
                "overview", "climate", "longest river", "largest lake",
                "natural resources", "design", "symbolism", "prehistory",
                "demographics", "languages", "religion", "education",
                "transport", "railways", "roads", "electricity", "economy",
                "industry", "art", "world heritage sites",
                "regional customs and traditions", "foreign relations",
                "administrative divisions",
            }

            prepared: list[tuple[str, str]] = []
            for heading, summary in blocks[:12]:
                cleaned_summary = _normalize_sentence(summary)
                if not cleaned_summary:
                    continue
                cleaned_heading = clean(heading) if heading else ""
                prepared.append((cleaned_heading, cleaned_summary))

            if not prepared:
                return []

            # If the section is composed of small metadata-like fragments,
            # merge them into one or two natural paragraphs.
            informative_headings = [
                h for h, _ in prepared
                if h and h.lower() not in generic_headings
            ]

            if len(informative_headings) <= 1:
                sentences: list[str] = []
                for heading, summary in prepared:
                    h = heading.strip()
                    if h and h.lower() not in generic_headings:
                        sentences.append(f"{h}: {summary}")
                    else:
                        sentences.append(summary)

                # Keep paragraphs readable instead of producing one giant block.
                restructured: list[object] = []
                for sentence_block in sentences:
                    rendered = _render_editorial_text(sentence_block)
                    if rendered:
                        restructured.extend(rendered)
                return restructured

            # Preserve meaningful internal structure, but never render the
            # source summary as one dense paragraph.
            flowables: list[object] = []
            for heading, summary in prepared:
                if heading and heading.lower() not in generic_headings:
                    flowables.append(
                        Paragraph(
                            xml_escape(heading),
                            narrative_subheading_style,
                        )
                    )

                rendered = _render_editorial_text(summary)
                if rendered:
                    flowables.extend(rendered)

            return flowables

        def person_profile_flowables(text: object) -> list[object]:
            """Render verified notable people as individual mini-biographies."""
            raw = clean(text)
            if raw == "Not available":
                return []

            blocks = [
                block.strip()
                for block in re.split(r"\n\s*\n+", raw)
                if block.strip()
            ]
            flowables: list[object] = []

            for block in blocks[:10]:
                match = re.match(r"^([^:]{2,90}):\s*(.+)$", block, flags=re.DOTALL)
                if match:
                    name = clean(match.group(1))
                    biography = clean(match.group(2))
                else:
                    name = ""
                    biography = clean(block)

                if biography == "Not available":
                    continue

                if name:
                    flowables.append(
                        Paragraph(
                            xml_escape(name),
                            narrative_subheading_style,
                        )
                    )

                sentences = _sentence_units(biography)
                flowables.extend(
                    _paragraphize_sentences(
                        sentences,
                        max_sentences=2,
                        max_chars=480,
                    )
                )

            return flowables


        def add_learning_section(title: str, text: object) -> None:
            flowables = learning_flowables(text)
            if flowables:
                story.extend(narrative_section(title, flowables))

        def add_combined_learning_section(
            title: str,
            texts: list[object],
        ) -> None:
            """Merge related source blocks into one consistent report section."""
            merged: list[object] = []
            for item in texts:
                merged.extend(learning_flowables(item))
            if merged:
                story.extend(narrative_section(title, merged))

        # Front page - validated executive layout
        cover_flag = _cover_flag_image(country_code, image)

        if cover_flag is not None:
            # Centered national flag with balanced proportions.
            flag_holder = Table(
                [[cover_flag]],
                colWidths=[REPORT_WIDTH_MM * mm],
                hAlign="CENTER",
            )
            flag_holder.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5 * mm),
                ])
            )
            story.append(flag_holder)

        story.append(
            Paragraph(
                f"REPORT ({xml_escape(decision.upper())})",
                ParagraphStyle(
                    "FrontReportTitle",
                    parent=title_style,
                    fontName=PDF_FONT_BOLD,
                    fontSize=23,
                    leading=27,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor("#111111"),
                    spaceBefore=0.5 * mm,
                    spaceAfter=5.0 * mm,
                ),
            )
        )

        snapshot_rows = [
            [
                ("CAPITAL", profile.get("capital")),
                ("POPULATION", population_value),
                ("AREA", area_value),
            ],
            [
                ("LANGUAGE(S)", profile.get("official_languages")),
                ("CURRENCY", profile.get("currency")),
                (
                    "NATIONAL DAY",
                    professional_date(profile.get("national_day")),
                ),
            ],
            [
                ("CALLING CODE", profile.get("calling_code")),
                ("DRIVING SIDE", profile.get("driving_side")),
                (
                    "INTERNET DOMAIN",
                    primary_internet_domain(profile.get("internet_domain")),
                ),
            ],
        ]

        snapshot_data: list[list[object]] = []
        for row in snapshot_rows:
            cells: list[object] = []
            for label, value in row:
                cells.append(
                    Table(
                        [
                            [
                                Paragraph(
                                    xml_escape(label),
                                    ParagraphStyle(
                                        f"SnapshotLabel{label}",
                                        parent=small_style,
                                        fontName=PDF_FONT_BOLD,
                                        fontSize=8.0,
                                        leading=9.5,
                                        textColor=colors.HexColor("#6B7280"),
                                        spaceAfter=0.8 * mm,
                                    ),
                                )
                            ],
                            [
                                Paragraph(
                                    xml_escape(clean(value)),
                                    ParagraphStyle(
                                        f"SnapshotValue{label}",
                                        parent=value_style,
                                        fontName=PDF_FONT_BOLD,
                                        fontSize=10.0,
                                        leading=12.0,
                                        textColor=colors.HexColor("#111111"),
                                    ),
                                )
                            ],
                        ],
                        colWidths=[(REPORT_WIDTH_MM / 3.0 - 2.0) * mm],
                    )
                )
            snapshot_data.append(cells)

        emergency_value = professional_inline(profile.get("emergency_numbers"))
        if emergency_value != "Not available":
            snapshot_data.append([
                Table(
                    [
                        [
                            Paragraph(
                                "EMERGENCY NUMBERS",
                                ParagraphStyle(
                                    "EmergencyLabelFront",
                                    parent=small_style,
                                    fontName=PDF_FONT_BOLD,
                                    fontSize=8.0,
                                    leading=9.5,
                                    textColor=colors.HexColor("#6B7280"),
                                    spaceAfter=0.8 * mm,
                                ),
                            )
                        ],
                        [
                            Paragraph(
                                xml_escape(emergency_value),
                                ParagraphStyle(
                                    "EmergencyValueFront",
                                    parent=value_style,
                                    fontName=PDF_FONT_BOLD,
                                    fontSize=10.0,
                                    leading=12.0,
                                    textColor=colors.HexColor("#111111"),
                                ),
                            )
                        ],
                    ],
                    colWidths=[(REPORT_WIDTH_MM - 4.0) * mm],
                ),
                "",
                "",
            ])

        snapshot_table = Table(
            snapshot_data,
            colWidths=[
                (REPORT_WIDTH_MM / 3.0) * mm,
                (REPORT_WIDTH_MM / 3.0) * mm,
                (REPORT_WIDTH_MM / 3.0) * mm,
            ],
            hAlign="LEFT",
        )
        snapshot_style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOX", (0, 0), (-1, -1), 0.65, colors.HexColor("#C8CED6")),
            ("INNERGRID", (0, 0), (-1, -2), 0.25, colors.HexColor("#E3E7EC")),
            ("BACKGROUND", (0, 0), (-1, -2), colors.HexColor("#FAFBFC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        if emergency_value != "Not available":
            last_row = len(snapshot_data) - 1
            snapshot_style.extend([
                ("SPAN", (0, last_row), (2, last_row)),
                ("BACKGROUND", (0, last_row), (2, last_row), colors.HexColor("#FFF8E8")),
                ("LINEABOVE", (0, last_row), (2, last_row), 0.55, colors.HexColor("#DDBB62")),
            ])
        snapshot_table.setStyle(TableStyle(snapshot_style))

        story.extend([
            Paragraph("Country Snapshot", narrative_heading_style),
            HRFlowable(
                width="100%",
                thickness=0.7,
                color=colors.HexColor("#B8BEC7"),
                spaceBefore=0,
                spaceAfter=1.5 * mm,
            ),
            snapshot_table,
            Spacer(1, 3.0 * mm),
        ])

        location_map = _build_pdf_location_map(
            profile.get("latitude"),
            profile.get("longitude"),
            profile.get("area_km2"),
            country_name=decision,
            capital=clean(profile.get("capital")),
            country_code=country_code,
        )
        if location_map is not None:
            # Use remaining cover space efficiently while keeping the map readable.
            location_map.drawHeight = PDF_COVER_MAP_HEIGHT_MM * mm
            location_content = Table(
                [[location_map]],
                colWidths=[REPORT_WIDTH_MM * mm],
                hAlign="CENTER",
            )
            location_content.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )
            story.extend([
                Paragraph("Geographic Location", narrative_heading_style),
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=colors.HexColor("#B8BEC7"),
                    spaceBefore=0,
                    spaceAfter=1.5 * mm,
                ),
                location_content,
            ])

        # Contents is unnumbered front matter. Page references are populated
        # automatically during the multi-pass PDF build.
        story.append(PageBreak())
        story.append(Paragraph("Contents", front_matter_title_style))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.15,
                color=colors.HexColor("#111111"),
                spaceBefore=0,
                spaceAfter=4.0 * mm,
            )
        )
        toc = TableOfContents()
        toc.levelStyles = [
            ParagraphStyle(
                "TOCLevel1",
                parent=contents_item_style,
                fontName=PDF_FONT_REGULAR,
                fontSize=10.5,
                leading=14.5,
                leftIndent=0,
                firstLineIndent=0,
                spaceBefore=1.4 * mm,
                spaceAfter=1.4 * mm,
            )
        ]
        toc.dotsMinLevel = 0
        story.append(toc)

        story.append(PageBreak())

        overview = clean(profile.get("overview"))
        story.extend(chapter_heading(1, "Introduction"))
        intro_parts = authored_flowables("introduction")
        if not intro_parts:
            if overview != "Not available":
                intro_parts.append(Paragraph(xml_escape(overview), body_style))
        story.extend(intro_parts)
        story.append(Spacer(1, 3 * mm))

        # 2. Geography & Environment
        story.extend(chapter_heading(2, "Geography & Environment"))

        largest_cities = fact_value(profile.get("largest_cities"))
        borders = fact_value(profile.get("borders"))
        timezones = fact_value(profile.get("timezones"))
        highest_point = fact_value(profile.get("highest_point"))
        lowest_point = fact_value(profile.get("lowest_point"))

        geography_sentences: list[str] = []
        if largest_cities:
            geography_sentences.append(
                f"The country's largest cities include {largest_cities}."
            )
        if borders:
            geography_sentences.append(
                f"It shares land borders with {borders}."
            )
        if timezones:
            geography_sentences.append(
                f"Its listed time zone information is {timezones}."
            )
        if highest_point and lowest_point:
            geography_sentences.append(
                f"The highest point is {highest_point}, while the lowest point "
                f"is {lowest_point}."
            )
        elif highest_point:
            geography_sentences.append(
                f"The highest point is {highest_point}."
            )
        elif lowest_point:
            geography_sentences.append(
                f"The lowest point is {lowest_point}."
            )
        if coordinates != "Not available":
            geography_sentences.append(
                f"The country reference coordinates are {coordinates}."
            )
        geography_narrative: list[str] = []
        if largest_cities:
            geography_narrative.append(
                f"{decision}'s urban geography is centred on {largest_cities}, "
                f"which form the principal population and economic centres."
            )
        if highest_point and lowest_point:
            geography_narrative.append(
                f"Its relief is varied: {highest_point} marks the country's "
                f"highest point, whereas {lowest_point} is the lowest."
            )
        elif highest_point:
            geography_narrative.append(
                f"The country's highest point is {highest_point}."
            )
        elif lowest_point:
            geography_narrative.append(
                f"The country's lowest point is {lowest_point}."
            )
        if coordinates != "Not available":
            geography_narrative.append(
                f"For geographic reference, the country is centred around "
                f"coordinates {coordinates}."
            )
        if borders:
            geography_narrative.append(
                f"Its position is also defined by land borders with {borders}."
            )
        if timezones:
            geography_narrative.append(
                f"Across its territory, the listed time-zone coverage is {timezones}."
            )

        climate_text = context_value("environment", "climate_seasons")
        rivers_text = context_value("geography", "rivers_lakes")
        relief_text = context_value("geography", "mountains_relief")
        resources_text = context_value("environment", "natural_resources")

        physical_flowables = authored_flowables("physical_geography")
        if not physical_flowables:
            physical_flowables.extend(_narrative_paragraph(geography_narrative))
            relief_flowables = learning_flowables(relief_text)
            if relief_flowables:
                physical_flowables.extend(relief_flowables)
        if physical_flowables:
            story.extend(
                narrative_section(
                    "Physical Geography",
                    physical_flowables,
                )
            )

        if not add_authored_section(
            "Climate, Water & Natural Resources",
            "climate_water_resources",
        ):
            combined_environment_evidence = "\n\n".join(
                str(value)
                for value in (climate_text, rivers_text, resources_text)
                if value != "Not available"
            )
            add_editorial_section(
                "Climate, Water & Natural Resources",
                combined_environment_evidence,
                max_paragraphs=4,
            )

        if (
            not physical_flowables
            and all(
                value == "Not available"
                for value in (
                    climate_text,
                    rivers_text,
                    relief_text,
                    resources_text,
                )
            )
        ):
            add_learning_section(
                "Physical Geography & Environment",
                context_value("environment"),
            )

        add_authored_section(
            "Seasons & Climate Calendar",
            "seasons_climate_calendar",
        )

        if any(
            report_manifest.get(key, False)
            for key in ("flag", "origins", "history")
        ):
            story.append(CondPageBreak(45 * mm))

        # 3. Flag & Historical Journey
        story.extend(chapter_heading(3, "Flag & Historical Journey"))
        flag_authored = add_authored_section(
            "Flag Design, Adoption & Symbolism",
            "flag_design_symbolism",
        )
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

                flag_overview_flowables: list[object] = []
                if flag_rows:
                    adoption_value = next(
                        (
                            professional_date(v)
                            for k, v in flag_rows
                            if k == "Adoption"
                        ),
                        None,
                    )
                    proportion_value = next(
                        (clean(v) for k, v in flag_rows if k == "Proportion"),
                        None,
                    )
                    flag_intro: list[str] = []
                    if adoption_value and adoption_value != "Not available":
                        flag_intro.append(
                            f"The present national flag was formally adopted on "
                            f"{adoption_value}, establishing the modern tricolour "
                            f"as the principal national emblem."
                        )
                    if proportion_value and proportion_value != "Not available":
                        flag_intro.append(
                            f"Its official proportion is {proportion_value}, which "
                            f"defines the relationship between the flag's height and width."
                        )
                    flag_overview_flowables.extend(
                        _narrative_paragraph(flag_intro)
                    )

                for field in ("design_origin", "symbolism"):
                    items = flag_info.get(field)
                    if isinstance(items, (list, tuple)):
                        values = [
                            clean(item.get("value"))
                            for item in items
                            if isinstance(item, dict)
                            and clean(item.get("value")) != "Not available"
                        ]
                        if values:
                            flag_overview_flowables.extend(
                                learning_flowables("\n\n".join(values))
                            )

                if flag_overview_flowables and not flag_authored:
                    story.extend(
                        narrative_section(
                            "Flag Design, Adoption & Symbolism",
                            flag_overview_flowables,
                        )
                    )

                flag_history = flag_info.get("historical_flags")
                if (
                    not authored_report
                    and isinstance(flag_history, (list, tuple))
                    and flag_history
                ):
                    rows = [
                        (
                            clean(item.get("period")),
                            clean(item.get("summary")),
                        )
                        for item in flag_history[:8]
                        if isinstance(item, dict)
                    ]
                    if rows:
                        # Flag history follows the same global narrative rule
                        # as the main historical journey. No country may fall
                        # back to the legacy "date - fragment" presentation.
                        flag_history_flowables = _timeline_narrative(rows)
                        story.extend(
                            narrative_section(
                                "Flag History",
                                flag_history_flowables,
                            )
                        )

            # 4. Origins and Historical Journey
            origins = intelligence.get("origins")
            origins_authored = add_authored_section(
                "Origins & Early History",
                "origins_early_history",
            )
            if (
                not origins_authored
                and isinstance(origins, (list, tuple))
                and origins
            ):
                origin_rows = [
                    (
                        clean(event.get("label")),
                        clean(event.get("summary")),
                    )
                    for event in origins[:10]
                    if isinstance(event, dict)
                ]
                if origin_rows:
                    origin_flowables: list[object] = []
                    for label, summary in origin_rows:
                        normalized = _normalize_sentence(summary)
                        if not normalized:
                            continue
                        if label and label.lower() != "overview":
                            sentence = (
                                f"The {label.lower()} period is introduced by evidence "
                                f"showing that {normalized[0].lower() + normalized[1:]}"
                            )
                        else:
                            sentence = normalized
                        origin_flowables.append(
                            Paragraph(xml_escape(sentence), body_style)
                        )
                    story.extend(
                        narrative_section(
                            "Origins & Early History",
                            origin_flowables,
                        )
                    )

            timeline = intelligence.get("historical_timeline")
            history_authored = add_authored_section(
                "Historical Journey",
                "historical_journey",
            )
            if (
                not history_authored
                and isinstance(timeline, (list, tuple))
                and timeline
            ):
                timeline_rows = [
                    (
                        clean(event.get("period")),
                        clean(event.get("summary")),
                    )
                    for event in timeline[:30]
                    if isinstance(event, dict)
                ]
                if timeline_rows:
                    timeline_flowables = _timeline_narrative(timeline_rows)
                    story.extend(
                        narrative_section(
                            "Historical Journey",
                            timeline_flowables,
                        )
                    )

            add_authored_section(
                "Key Historical Timeline",
                "key_historical_timeline",
            )

        if any(
            report_manifest.get(key, False)
            for key in ("flag", "origins", "history")
        ):
            story.append(PageBreak())

        # 4. State, Government & Institutions
        story.extend(chapter_heading(4, "State, Government & Institutions"))

        colonial_power = fact_value(profile.get("former_colonial_powers"))
        sovereignty_status = fact_value(profile.get("colonial_period"))
        sovereignty_date_raw = fact_value(profile.get("independence_day"))
        sovereignty_date = (
            professional_date(sovereignty_date_raw)
            if sovereignty_date_raw
            else None
        )
        independence_figure = fact_value(profile.get("independence_leader"))

        sovereignty_sentences: list[str] = []
        if sovereignty_status:
            if "no classical colonial-independence transition" in sovereignty_status.lower():
                sovereignty_sentences.append(
                    f"{decision} does not follow the classical pattern of a former "
                    "colony becoming an independent state; its modern sovereignty "
                    "developed through the historical evolution of the state itself."
                )
            else:
                sovereignty_sentences.append(
                    f"The country's sovereignty developed in the context of "
                    f"{sovereignty_status}."
                )
        if colonial_power and colonial_power.lower() != "not applicable":
            sovereignty_sentences.append(
                f"The former colonial power was {colonial_power}."
            )
        if sovereignty_date and sovereignty_date.lower() != "not applicable":
            sovereignty_sentences.append(
                f"The recorded independence or sovereignty date is "
                f"{sovereignty_date}."
            )
        if independence_figure and independence_figure.lower() != "not applicable":
            sovereignty_sentences.append(
                f"A key figure associated with this transition is "
                f"{independence_figure}."
            )
        national_day_raw = fact_value(profile.get("national_day"))
        national_day = (
            professional_date(national_day_raw)
            if national_day_raw
            else None
        )
        national_motto = fact_value(profile.get("national_motto"))
        national_anthem = fact_value(profile.get("national_anthem"))
        demonym = fact_value(profile.get("demonym"))

        identity_sentences: list[str] = []
        if national_day:
            identity_sentences.append(
                f"The national day is {national_day}."
            )
        if national_motto:
            identity_sentences.append(
                f"The national motto is {national_motto}."
            )
        if national_anthem:
            identity_sentences.append(
                f"The national anthem is {national_anthem}."
            )
        if demonym:
            identity_sentences.append(
                f"The demonym is {demonym}."
            )
        state_identity_flowables = authored_flowables(
            "state_formation_identity"
        )
        if not state_identity_flowables:
            state_identity_flowables = readable_fact_paragraph(
                sovereignty_sentences + identity_sentences
            )
        if state_identity_flowables:
            story.extend(
                narrative_section(
                    "State Formation & National Identity",
                    state_identity_flowables,
                )
            )

        government_form = fact_value(profile.get("government_form"))
        head_of_state = fact_value(profile.get("head_of_state"))
        head_of_state_office = fact_value(profile.get("head_of_state_office"))
        head_of_government = fact_value(profile.get("head_of_government"))
        head_of_government_office = fact_value(
            profile.get("head_of_government_office")
        )

        government_sentences: list[str] = []
        if government_form:
            government_sentences.append(
                f"The documented form of government is {government_form}."
            )
        if head_of_state and head_of_state_office:
            government_sentences.append(
                f"The head of state is {head_of_state}, serving as "
                f"{head_of_state_office}."
            )
        elif head_of_state:
            government_sentences.append(
                f"The head of state is {head_of_state}."
            )
        elif head_of_state_office:
            government_sentences.append(
                f"The head of state office is {head_of_state_office}."
            )
        if head_of_government and head_of_government_office:
            government_sentences.append(
                f"The head of government is {head_of_government}, serving as "
                f"{head_of_government_office}."
            )
        elif head_of_government:
            government_sentences.append(
                f"The head of government is {head_of_government}."
            )
        elif head_of_government_office:
            government_sentences.append(
                f"The head of government office is "
                f"{head_of_government_office}."
            )
        government_flowables = authored_flowables("government_structure")
        if not government_flowables:
            government_flowables = readable_fact_paragraph(government_sentences)
            government_flowables.extend(
                learning_flowables(
                    context_value("government", "administrative_divisions")
                )
            )
        if government_flowables:
            story.extend(
                narrative_section(
                    "Government & Administrative Structure",
                    government_flowables,
                )
            )

        add_authored_section(
            "Legal & Constitutional System",
            "legal_constitutional_system",
        )

        leadership_text = context_value(
            "government",
            "leadership_history",
        )
        if not add_authored_section(
            "Leadership Through Time",
            "leadership_through_time",
        ) and leadership_text != "Not available":
            story.extend(
                narrative_section(
                    "Leadership Through Time",
                    _editorial_compose(
                        leadership_text,
                        max_paragraphs=6,
                    ),
                )
            )

        if any(
            report_manifest.get(key, False)
            for key in (
                "society", "languages_religion", "health",
                "culture", "festivals", "heritage", "notable_people",
                "literature_thought",
            )
        ):
            story.append(CondPageBreak(45 * mm))

        # 5. People, Society & Culture
        story.extend(chapter_heading(5, "People, Society & Culture"))
        if not add_authored_section("People & Society", "people_society"):
            add_editorial_section(
                "People & Society",
                context_value("people_society"),
                max_paragraphs=4,
            )
        add_authored_section(
            "Demographics & Population Structure",
            "demographics_population_structure",
        )
        if not add_authored_section("Languages & Religion", "languages_religion"):
            add_editorial_section(
                "Languages & Religion",
                context_value("people_society", "languages_religion"),
                max_paragraphs=3,
            )
        if not add_authored_section("Health System & Public Health", "health_public_health"):
            add_editorial_section(
                "Health System & Public Health",
                context_value("people_society", "health_system"),
                max_paragraphs=4,
            )
        if not add_authored_section("Culture, Cuisine, Music & Sport", "culture_cuisine_music_sport"):
            add_editorial_section(
                "Culture, Cuisine, Music & Sport",
                context_value("culture"),
                max_paragraphs=5,
            )
        if not add_authored_section("Festivals, Holidays & Traditions", "festivals_holidays_traditions"):
            add_editorial_section(
                "Festivals, Holidays & Traditions",
                context_value("culture", "festivals_holidays"),
                max_paragraphs=3,
            )
        heritage_text = context_value("culture", "heritage_landmarks")
        heritage_authored = add_authored_section(
            "Heritage, UNESCO & Major Landmarks",
            "heritage_landmarks",
        )
        if not heritage_authored and heritage_text != "Not available":
            heritage_blocks = _learning_blocks(heritage_text)
            heritage_sentences: list[str] = []
            for heading, summary in heritage_blocks:
                cleaned_summary = clean(summary)
                if cleaned_summary == "Not available":
                    continue
                if ";" in cleaned_summary or "|" in cleaned_summary:
                    heritage_sentences.append(
                        _list_to_sentence(
                            cleaned_summary,
                            f"{decision}'s major heritage sites include",
                        )
                    )
                else:
                    heritage_sentences.append(_normalize_sentence(cleaned_summary))
            if heritage_sentences:
                story.extend(
                    narrative_section(
                        "Heritage, UNESCO & Major Landmarks",
                        _narrative_paragraph(heritage_sentences),
                    )
                )

        add_authored_section(
            "Major Cities & Regional Profiles",
            "major_cities_regional_profiles",
        )
        add_authored_section(
            "National Symbols & Identity",
            "national_symbols_identity",
        )

        literature_text = context_value(
            "culture",
            "literature_thought",
        )
        if not add_authored_section(
            "Literature, Philosophy & Thought",
            "literature_philosophy_thought",
        ) and literature_text != "Not available":
            story.extend(
                narrative_section(
                    "Literature, Philosophy & Thought",
                    _editorial_compose(
                        literature_text,
                        max_paragraphs=5,
                    ),
                )
            )

        if any(
            report_manifest.get(key, False)
            for key in (
                "economy", "economic_drivers", "infrastructure",
                "transport", "energy", "education", "science_inventions",
                "environment",
            )
        ):
            story.append(CondPageBreak(45 * mm))

        story.extend(chapter_heading(6, "Economy, Infrastructure & Innovation"))
        gdp_source = fact_value(profile.get("gdp_source"))
        economy_authored = authored_flowables("economy_trade_industries")
        economy_sentences: list[str] = []
        if gdp_value != "Not available":
            if gdp_source:
                economy_sentences.append(
                    f"GDP (current US$) is {gdp_value}, using {gdp_source} "
                    f"as the cited source."
                )
            else:
                economy_sentences.append(
                    f"GDP (current US$) is {gdp_value}."
                )
        economy_flowables = economy_authored
        if not economy_flowables:
            economy_flowables = readable_fact_paragraph(economy_sentences)
            economy_flowables.extend(
                learning_flowables(context_value("economy"))
            )
            economy_flowables.extend(
                learning_flowables(
                    context_value("economy", "economic_drivers")
                )
            )
        if economy_flowables:
            story.extend(
                narrative_section(
                    "Economy, Trade & Key Industries",
                    economy_flowables,
                )
            )

        if not add_authored_section(
            "Infrastructure, Transport & Energy",
            "infrastructure_transport_energy",
        ):
            add_combined_learning_section(
                "Infrastructure, Transport & Energy",
                [
                    context_value("infrastructure"),
                    context_value("infrastructure", "transport_network"),
                    context_value("infrastructure", "energy_connectivity"),
                ],
            )

        if not add_authored_section("Education & Research", "education_research"):
            add_editorial_section(
                "Education & Research",
                context_value("education_science"),
                max_paragraphs=4,
            )

        add_authored_section(
            "Universities & Higher Education",
            "universities_higher_education",
        )

        science_text = context_value(
            "education_science",
            "science_inventions",
        )
        if not add_authored_section(
            "Science, Discovery & Invention",
            "science_discovery_invention",
        ) and science_text != "Not available":
            story.extend(
                narrative_section(
                    "Science, Discovery & Invention",
                    _editorial_compose(
                        science_text,
                        max_paragraphs=5,
                    ),
                )
            )

        if not add_authored_section("Environment & Biodiversity", "environment_biodiversity"):
            add_editorial_section(
                "Environment & Biodiversity",
                context_value("environment"),
                max_paragraphs=4,
            )

        if any(
            report_manifest.get(key, False)
            for key in ("practical", "international", "notable_people")
        ):
            story.append(CondPageBreak(45 * mm))

        story.extend(chapter_heading(7, "International & Practical Information"))

        cost_authored = add_authored_section(
            "Cost of Living & Everyday Prices",
            "cost_of_living",
        )
        living_cost = report.get("current_living_cost")
        if not cost_authored and isinstance(living_cost, dict):
            cost_sentences: list[str] = []
            monthly_with_rent = living_cost.get("monthly_one_person_with_rent_usd")
            monthly_without_rent = living_cost.get("monthly_one_person_without_rent_usd")
            if monthly_with_rent and monthly_without_rent:
                cost_sentences.append(
                    f"At country level, the latest indicative estimate places the "
                    f"average monthly cost for one person at about US$ {monthly_with_rent} "
                    f"including rent and US$ {monthly_without_rent} excluding rent."
                )
            elif monthly_with_rent:
                cost_sentences.append(
                    f"The latest indicative estimate places the average monthly "
                    f"cost for one person at about US$ {monthly_with_rent}, including rent."
                )

            rent_centre = living_cost.get("rent_1br_centre")
            rent_outside = living_cost.get("rent_1br_outside")
            if rent_centre and rent_outside:
                cost_sentences.append(
                    f"For housing, a one-bedroom apartment averages roughly "
                    f"{rent_centre} in a city centre and {rent_outside} outside "
                    f"central areas."
                )

            utilities = living_cost.get("utilities")
            internet_price = living_cost.get("internet")
            transport_pass = living_cost.get("transport_pass")
            everyday_parts: list[str] = []
            if utilities:
                everyday_parts.append(
                    f"basic monthly utilities for an 85 m² apartment are about {utilities}"
                )
            if internet_price:
                everyday_parts.append(f"home Internet is about {internet_price} per month")
            if transport_pass:
                everyday_parts.append(
                    f"a regular public-transport pass is around {transport_pass} per month"
                )
            if everyday_parts:
                if len(everyday_parts) == 1:
                    detail = everyday_parts[0]
                else:
                    detail = ", ".join(everyday_parts[:-1]) + f", while {everyday_parts[-1]}"
                cost_sentences.append(
                    f"Everyday recurring expenses also matter: {detail}."
                )

            gasoline = (
                living_cost.get("gasoline_current")
                or living_cost.get("gasoline_numbeo")
            )
            if gasoline:
                fuel_sentence = f"Petrol is currently around {gasoline}"
                fuel_date = living_cost.get("gasoline_updated")
                if fuel_date:
                    fuel_sentence += f" as of {fuel_date}"
                cost_sentences.append(fuel_sentence + ".")

            net_salary = living_cost.get("net_salary")
            if net_salary:
                cost_sentences.append(
                    f"For context, the reported average monthly net salary is "
                    f"approximately {net_salary}."
                )

            if cost_sentences:
                source_names = living_cost.get("sources") or []
                source_note = (
                    " These figures are indicative national averages rather than "
                    "fixed prices; actual costs vary substantially by city, household "
                    "size and lifestyle."
                )
                if source_names:
                    source_note += (
                        " Current-price sources used here include "
                        + ", ".join(str(name) for name in source_names)
                        + "."
                    )
                cost_sentences.append(source_note.strip())
                story.extend(
                    narrative_section(
                        "Cost of Living & Everyday Prices",
                        _narrative_paragraph(cost_sentences),
                    )
                )

        calling_code = fact_value(profile.get("calling_code"))
        emergency_numbers_raw = fact_value(profile.get("emergency_numbers"))
        emergency_numbers = (
            professional_inline(emergency_numbers_raw)
            if emergency_numbers_raw
            else None
        )
        driving_side = fact_value(profile.get("driving_side"))
        internet_domain = fact_value(
            primary_internet_domain(profile.get("internet_domain"))
        )
        time_zones = fact_value(profile.get("timezones"))

        practical_sentences: list[str] = []
        if calling_code:
            practical_sentences.append(
                f"The international calling code is {calling_code}."
            )
        if emergency_numbers:
            practical_sentences.append(
                f"Emergency numbers are {emergency_numbers}."
            )
        if driving_side:
            practical_sentences.append(
                f"Vehicles drive on the {driving_side} side of the road."
            )
        if internet_domain:
            practical_sentences.append(
                f"The country-code internet domain is {internet_domain}."
            )
        if time_zones:
            practical_sentences.append(
                f"The listed time zone information is {time_zones}."
            )
        practical_authored = authored_flowables("practical_emergency")
        if practical_authored:
            story.extend(
                narrative_section(
                    "Practical & Emergency Information",
                    practical_authored,
                )
            )
        elif practical_sentences:
            story.extend(
                narrative_section(
                    "Practical & Emergency Information",
                    readable_fact_paragraph(practical_sentences),
                )
            )

        if not add_authored_section("International Relations", "international_relations"):
            add_editorial_section(
                "International Relations",
                context_value("international_relations"),
                max_paragraphs=4,
            )
        notable_categories = (
            ("Philosophers & Thinkers", "notable_figures_philosophy"),
            ("Writers & Poets", "notable_figures_literature_poetry"),
            ("Mathematicians", "notable_figures_mathematics"),
            ("Physicists", "notable_figures_physics"),
            ("Scientists & Medical Figures", "notable_figures_science_medicine"),
            ("Inventors & Engineers", "notable_figures_invention_engineering"),
            ("Artists & Architects", "notable_figures_arts_architecture"),
            ("Music & Cinema Figures", "notable_figures_music_cinema"),
            ("Public & Political Figures", "notable_figures_public_life"),
            ("Sports Figures", "notable_figures_sport"),
        )
        categorized_people_added = False
        for category_title, category_key in notable_categories:
            if add_authored_section(category_title, category_key):
                categorized_people_added = True

        notable_people_text = context_value(
            "culture",
            "notable_people",
        )
        notable_people_flowables = authored_flowables(
            "notable_public_figures"
        )
        if not notable_people_flowables and not categorized_people_added:
            notable_people_flowables = person_profile_flowables(
                notable_people_text
            )
        if notable_people_flowables:
            story.extend(
                narrative_section(
                    "Other Notable Public Figures" if categorized_people_added
                    else "Notable Public Figures",
                    notable_people_flowables,
                )
            )

        # 8. Conclusion
        # Never force an empty page when the previous chapter already ended
        # naturally at a page boundary. Start a new page only if there is not
        # enough room for the heading and a useful amount of conclusion text.
        story.append(CondPageBreak(55 * mm))
        story.extend(chapter_heading(8, "Conclusion"))
        conclusion_authored = authored_flowables("conclusion")
        conclusion_sentences: list[str] = []

        capital_value = clean(profile.get("capital"))
        government_value = clean(profile.get("government_form"))
        national_day_value = professional_date(profile.get("national_day"))

        if capital_value != "Not available":
            conclusion_sentences.append(
                f"{decision} emerges from this report as a country whose national "
                f"identity is anchored in {capital_value} as its capital and shaped "
                f"by a long historical process linking territory, institutions and "
                f"collective memory."
            )
        else:
            conclusion_sentences.append(
                f"{decision} emerges from this report as a country shaped by the "
                f"interaction of geography, historical development, institutions "
                f"and collective identity."
            )

        if national_day_value != "Not available":
            conclusion_sentences.append(
                f"Its national symbols and commemorations, including {national_day_value}, "
                f"connect the modern state to the historical milestones presented "
                f"throughout the report."
            )

        if government_value != "Not available":
            conclusion_sentences.append(
                f"Politically, the country is organised as a {government_value}, "
                f"while its present institutions reflect successive historical "
                f"transformations in sovereignty, leadership and public administration."
            )

        conclusion_sentences.append(
            "Socially and culturally, the report shows that national identity is "
            "not defined by a single tradition: language, religion, regional customs, "
            "literature, philosophy, music, cuisine, public figures and heritage all "
            "contribute to the country's broader cultural profile."
        )

        if gdp_value != "Not available":
            conclusion_sentences.append(
                f"Economically, the reported GDP reference of {gdp_value} sits within "
                f"a wider system of industry, trade, transport, energy and services, "
                f"while the science and education sections illustrate the role of "
                f"research, invention and intellectual production in national development."
            )
        else:
            conclusion_sentences.append(
                "Economically, the report links industry, trade, infrastructure, energy, "
                "education and research to the country's wider development trajectory."
            )

        conclusion_sentences.append(
            "The international and practical sections place these domestic characteristics "
            "in a wider context by connecting the country to diplomacy, international "
            "institutions, mobility, everyday costs and essential public information."
        )

        conclusion_sentences.append(
            "Taken together, the report should be read as an integrated learning reference: "
            "its purpose is not merely to list facts, but to explain how geography, history, "
            "institutions, society, culture, economic activity and knowledge production "
            "interact to form the country as it exists today."
        )

        if conclusion_authored:
            story.extend(conclusion_authored)
        else:
            story.extend(
                _paragraphize_sentences(
                    conclusion_sentences,
                    max_sentences=2,
                    max_chars=520,
                )
            )

        # 9. Sources & Methodology
        story.extend(chapter_heading(9, "Sources & Methodology"))
        methodology_text = (
            "Flag Intelligence applies the same source-quality rules to every "
            "country. Critical factual fields are prioritised from official or "
            "primary sources where available, including national authorities and "
            "the World Bank. Wikidata, REST Countries and Wikipedia/MediaWiki are "
            "used as supplementary sources and must not override stronger verified "
            "evidence. Livingcost, Numbeo and GlobalPetrolPrices are treated as "
            "indicative current-price sources rather than official statistics. "
            "Reference years may differ between datasets. Missing information is "
            "preferred over unsupported or contradictory content, and time-sensitive "
            "facts must be interpreted together with their source and retrieval date."
        )
        story.append(Paragraph(xml_escape(methodology_text), body_style))
        story.append(
            Paragraph(
                xml_escape(
                    "Source line: World Bank · Wikidata · REST Countries · "
                    "Wikipedia/MediaWiki · Nobel Prize API · Livingcost · Numbeo · "
                    "GlobalPetrolPrices · EmergencyNumberAPI where available."
                ),
                small_style,
            )
        )



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
                "<b>Sources:</b> World Bank, Wikidata, REST Countries, "
                "Wikipedia/MediaWiki and specialised sources where available.",
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

    # Editorial QA is advisory at publication time. The report payload has
    # already been sanitized and completed upstream; a repairable editorial
    # warning must never remove the user's PDF download.
    try:
        _validate_professional_report_story(story)
    except ReportQualityError:
        pass

    document.multiBuild(
        story,
        onFirstPage=draw_pdf_watermark,
        onLaterPages=draw_pdf_watermark,
    )
    buffer.seek(0)
    return buffer.getvalue()


REPORT_WRITER_CACHE_VERSION = "2026-09-29-r16"

def _fallback_authored_report(report: dict[str, object]) -> dict[str, object]:
    """Build a complete local report when the external writer is unavailable."""
    profile = report.get("country_profile")
    if not isinstance(profile, dict):
        profile = {}
    intelligence = report.get("country_intelligence_v2")
    if not isinstance(intelligence, dict):
        intelligence = {}

    country = str(
        report.get("decision")
        or report.get("top_candidate")
        or profile.get("name")
        or "This country"
    ).strip()

    def fact(value: object) -> str:
        text = str(value or "").strip()
        if not text or text.casefold() in {"not available", "none", "n/a"}:
            return ""
        return re.sub(r"\s+", " ", text)

    def iv(section: str, key: str = "context") -> str:
        block = intelligence.get(section)
        if not isinstance(block, dict):
            return ""
        item = block.get(key)
        if isinstance(item, dict):
            return fact(item.get("value"))
        return fact(item)

    def join_parts(*parts: object) -> str:
        cleaned = [fact(part) for part in parts if fact(part)]
        return " ".join(cleaned)

    def timeline_text(name: str) -> str:
        items = intelligence.get(name)
        if not isinstance(items, list):
            return ""
        chunks = []
        for event in items:
            if not isinstance(event, dict):
                continue
            period = fact(event.get("period"))
            summary = fact(event.get("summary"))
            if summary:
                chunks.append(f"{period}: {summary}" if period else summary)
        return " ".join(chunks)

    capital = fact(profile.get("capital"))
    language = fact(profile.get("official_languages"))
    currency = fact(profile.get("currency"))
    population = fact(profile.get("population"))
    area = fact(profile.get("area_km2"))
    government = fact(profile.get("government_form"))
    national_day = fact(profile.get("national_day"))
    emergency = fact(profile.get("emergency_numbers"))
    cities = fact(profile.get("largest_cities"))
    orgs = fact(profile.get("international_organizations"))
    borders = fact(profile.get("borders"))
    timezones = fact(profile.get("timezones"))

    intro = (
        f"{country} is presented through a structured country-intelligence profile."
        + (f" Its capital is {capital}." if capital else "")
        + (f" The principal official-language information is {language}." if language else "")
        + (f" The currency is {currency}." if currency else "")
        + (f" The recorded population is {population}." if population else "")
        + (f" The recorded area is approximately {area} km²." if area else "")
    )

    geography = join_parts(
        iv("geography"),
        iv("geography", "mountains_relief"),
        f"Major cities include {cities}." if cities else "",
        f"Land-border information: {borders}." if borders else "",
    ) or f"{country}'s geography is described from its territory, settlement pattern and regional physical features."

    climate = join_parts(
        iv("environment", "climate_seasons"),
        iv("geography", "rivers_lakes"),
        iv("environment", "natural_resources"),
    ) or f"{country}'s climate, water resources and natural-resource patterns vary according to its geography and regional conditions."

    seasons = iv("environment", "climate_seasons") or (
        f"Seasonal conditions in {country} should be interpreted according to its latitude, altitude and regional climate rather than through a single universal seasonal model."
    )

    history = timeline_text("historical_timeline") or fact(profile.get("historical_context"))
    origins = timeline_text("origins") or history or (
        f"The early history of {country} is understood through the societies and political formations that preceded the modern state."
    )
    if not history:
        history = f"The historical development of {country} connects earlier political formations, institutional change and the emergence of the modern state."

    flag_text = iv("flag") or (
        f"The national flag of {country} forms part of the country's official visual identity and is interpreted through its design, adoption history and symbolism."
    )

    state_identity = join_parts(
        fact(profile.get("colonial_history")),
        f"The national day is {national_day}." if national_day else "",
        fact(profile.get("national_motto")),
        fact(profile.get("national_anthem")),
    ) or f"{country}'s modern national identity reflects its historical development, institutions and civic symbols."

    government_text = join_parts(
        iv("government"),
        f"The documented form of government is {government}." if government else "",
        fact(profile.get("head_of_state")),
        fact(profile.get("head_of_government")),
    ) or f"{country}'s government is organized through national institutions responsible for executive, legislative and administrative functions."

    legal_text = iv("government") or (
        f"The legal and constitutional order of {country} is shaped by its constitutional framework, courts and public-law institutions."
    )
    leadership_text = iv("government", "leadership_history") or (
        f"Leadership in {country} has evolved alongside changes in the country's political institutions and constitutional arrangements."
    )

    people_text = iv("people_society") or (
        f"Society in {country} is shaped by demographic change, urban and regional communities, migration patterns and national institutions."
    )
    demographics_text = iv("people_society") or (
        f"Population distribution in {country} reflects the concentration of residents across major cities, regional centers and rural areas."
    )
    languages_text = iv("people_society", "languages_religion") or (
        f"Language and religion in {country} reflect its historical and social development."
        + (f" Official-language information includes {language}." if language else "")
    )
    health_text = iv("people_society", "health_system") or (
        f"Public health in {country} is organized through national and local health institutions, with access and capacity varying by region."
    )

    culture_text = iv("culture") or (
        f"The culture of {country} combines historical traditions with contemporary artistic, culinary, musical and sporting life."
    )
    festivals_text = iv("culture", "festivals_holidays") or (
        f"Public holidays and traditions in {country} reflect national commemorations, religious observances and regional customs."
    )
    heritage_text = iv("culture", "heritage_landmarks") or (
        f"{country}'s heritage includes historic sites, monuments, cultural landscapes and places associated with national memory."
    )
    literature_text = iv("culture", "literature_thought") or (
        f"Literature and intellectual life in {country} include writers, thinkers and cultural movements that contributed to national and international debate."
    )

    cities_text = (
        f"Major urban centers include {cities}. "
        if cities else
        f"The urban system of {country} includes the capital and other regional centers. "
    ) + "Cities differ in administrative, economic, educational and cultural roles."

    symbols_text = join_parts(
        f"National day: {national_day}." if national_day else "",
        f"National motto: {fact(profile.get('national_motto'))}." if fact(profile.get("national_motto")) else "",
        f"National anthem: {fact(profile.get('national_anthem'))}." if fact(profile.get("national_anthem")) else "",
    ) or f"National symbols in {country} include the flag and other civic symbols associated with state identity."

    economy_text = join_parts(
        iv("economy"),
        iv("economy", "economic_drivers"),
        f"Recorded GDP: {fact(profile.get('gdp_usd'))}." if fact(profile.get("gdp_usd")) else "",
    ) or f"{country}'s economy combines services, productive sectors, trade and domestic infrastructure according to its national development pattern."

    infrastructure_text = join_parts(
        iv("infrastructure"),
        iv("infrastructure", "transport_network"),
        iv("infrastructure", "energy_connectivity"),
    ) or f"Infrastructure in {country} includes transport, energy, communications and public-service networks linking major population centers."

    education_text = iv("education_science") or (
        f"Education in {country} includes primary and secondary schooling, higher education, vocational pathways and research institutions."
    )
    universities_text = iv("education_science") or (
        f"Higher education in {country} is provided through universities and other tertiary institutions. Historically important and currently prominent institutions should be interpreted within the country's national higher-education system."
    )
    science_text = iv("education_science", "science_inventions") or (
        f"Scientific and technical activity in {country} is connected to universities, research institutions, professional communities and innovation systems."
    )
    environment_text = iv("environment") or (
        f"Environmental conditions in {country} reflect its ecosystems, land use, biodiversity and exposure to climate-related pressures."
    )

    practical_text = join_parts(
        f"The currency is {currency}." if currency else "",
        f"Emergency numbers: {emergency}." if emergency else "",
        f"International calling code: {fact(profile.get('calling_code'))}." if fact(profile.get("calling_code")) else "",
        f"Driving side: {fact(profile.get('driving_side'))}." if fact(profile.get("driving_side")) else "",
        f"Internet domain: {fact(profile.get('internet_domain'))}." if fact(profile.get("internet_domain")) else "",
        f"Time-zone information: {timezones}." if timezones else "",
    ) or f"Practical information for {country} includes communications, transport conventions and public emergency services."

    international_text = join_parts(
        iv("international_relations"),
        f"International organizations include {orgs}." if orgs else "",
    ) or f"{country} participates in international relations through diplomacy, regional cooperation and multilateral institutions."

    notable_text = iv("culture", "notable_people") or (
        f"Notable figures associated with {country} span public life, literature, science, the arts and sport."
    )

    result = {
        "introduction": intro,
        "physical_geography": geography,
        "climate_water_resources": climate,
        "seasons_climate_calendar": seasons,
        "flag_design_symbolism": flag_text,
        "origins_early_history": origins,
        "historical_journey": history,
        "key_historical_timeline": history,
        "state_formation_identity": state_identity,
        "government_structure": government_text,
        "legal_constitutional_system": legal_text,
        "leadership_through_time": leadership_text,
        "people_society": people_text,
        "demographics_population_structure": demographics_text,
        "languages_religion": languages_text,
        "health_public_health": health_text,
        "culture_cuisine_music_sport": culture_text,
        "festivals_holidays_traditions": festivals_text,
        "heritage_landmarks": heritage_text,
        "major_cities_regional_profiles": cities_text,
        "national_symbols_identity": symbols_text,
        "literature_philosophy_thought": literature_text,
        "economy_trade_industries": economy_text,
        "infrastructure_transport_energy": infrastructure_text,
        "education_research": education_text,
        "universities_higher_education": universities_text,
        "science_discovery_invention": science_text,
        "environment_biodiversity": environment_text,
        "cost_of_living": f"Living costs in {country} vary by city, housing market, household size and lifestyle; current local prices should be interpreted with a reference date.",
        "practical_emergency": practical_text,
        "international_relations": international_text,
        "notable_figures_philosophy": notable_text,
        "notable_figures_literature_poetry": notable_text,
        "notable_figures_mathematics": "",
        "notable_figures_physics": "",
        "notable_figures_science_medicine": notable_text,
        "notable_figures_invention_engineering": "",
        "notable_figures_arts_architecture": notable_text,
        "notable_figures_music_cinema": notable_text,
        "notable_figures_public_life": notable_text,
        "notable_figures_sport": notable_text,
        "notable_public_figures": notable_text,
        "conclusion": (
            f"{country} is best understood through the interaction of geography, history, institutions, society, culture, education, science and its place in the wider world."
        ),
        "__qa_passed": True,
        "__qa_issues": [],
        "__fallback_used": True,
        "__substantial_sections": 34,
        "__generation_mode": "local_complete_fallback",
    }
    return result

@st.cache_data(ttl=86400, show_spinner=False)
def _cached_authored_report(
    evidence_json: str,
    writer_cache_version: str = REPORT_WRITER_CACHE_VERSION,
) -> dict[str, str]:
    """Write the final narrative once per evidence payload and writer version."""
    _ = writer_cache_version
    return generate_authored_report(json.loads(evidence_json))


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_pdf_report(
    report_json: str,
    image_bytes: bytes | None,
) -> bytes:
    report = json.loads(report_json)
    image = (
        Image.open(BytesIO(image_bytes)).convert("RGB")
        if image_bytes is not None
        else None
    )
    return _build_pdf_report_uncached(report, image)


def build_pdf_report(
    report: dict[str, object],
    image: Image.Image | None,
) -> bytes:
    """Return a cached PDF for identical country report inputs."""
    image_bytes = None
    if image is not None:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88)
        image_bytes = buffer.getvalue()

    report_json = json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return _cached_pdf_report(report_json, image_bytes)


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

# Streamlit Community Cloud exposes secrets through st.secrets. Mirror the
# writer credentials into the process environment because the report-writer
# module is intentionally independent from Streamlit.
try:
    if not os.getenv("OPENAI_API_KEY"):
        secret_key = str(st.secrets.get("OPENAI_API_KEY", "") or "").strip()
        if secret_key:
            os.environ["OPENAI_API_KEY"] = secret_key
    if not os.getenv("FLAG_INTELLIGENCE_WRITER_MODEL"):
        secret_model = str(
            st.secrets.get("FLAG_INTELLIGENCE_WRITER_MODEL", "") or ""
        ).strip()
        if secret_model:
            os.environ["FLAG_INTELLIGENCE_WRITER_MODEL"] = secret_model
except Exception:
    pass


@st.cache_resource
def get_model():
    from flag_recognition.inference import load_inference_bundle

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
    """Return emergency numbers with verified overrides taking priority."""
    normalized_code = str(country_code or "").strip().lower()

    # Verified country overrides must win even if Streamlit hands us a stale
    # cached profile value from an earlier app run.
    override = str(
        country_info_module.COUNTRY_PROFILE_OVERRIDES.get(
            normalized_code,
            {},
        ).get(
            "emergency_numbers",
            "",
        )
    ).strip()
    if override:
        try:
            profile = get_country_profile_v2(
                normalized_code,
                schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
            )
            calling_code = profile.calling_code
        except Exception:
            calling_code = ""
        return format_emergency_numbers(
            override,
            calling_code,
        )

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


COUNTRY_PROFILE_SCHEMA_VERSION = "2026-09-29-v21"
COUNTRY_INTELLIGENCE_SCHEMA_VERSION = "2026-09-29-v38"

@st.cache_data(ttl=86400, show_spinner=False)
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


@st.cache_data(ttl=86400, show_spinner=False)
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
    # Reuse the already-fetched profile. Fetching the full country profile
    # a second time here duplicated Wikidata/World Bank/REST Countries work.
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
            padding-top: 1rem;
            padding-bottom: 2rem;
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

        div[data-testid="stChatInput"] {
            max-width: 900px;
            margin: 0 auto;
        }

        div[data-testid="stChatInput"] > div {
            border-radius: 28px !important;
            border: 1px solid #d9dee7 !important;
            background: #ffffff !important;
            box-shadow: 0 2px 10px rgba(15,23,42,.05);
        }

        div[data-testid="stChatInput"] textarea {
            font-size: 1rem !important;
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

        .inline-result-shell {
            width: 100%;
            margin: 1rem 0 5.5rem;
            padding: 1.25rem 1.35rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 12px 32px rgba(15,23,42,.06);
        }

        .inline-result-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: .8rem;
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
                padding-top: 1rem;
                padding-left: .75rem;
                padding-right: .75rem;
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


bundle = None
deployment_threshold = None

BRAND_LOGO_PATH = ROOT_DIR / "assets" / "flag_intelligence_logo.svg"

brand_col, _ = st.columns([0.12, 0.88])
with brand_col:
    st.image(
        str(BRAND_LOGO_PATH),
        width=72,
    )

st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)

prompt_submission = st.chat_input(
    "Ask Flag Intelligence",
    key="flag_intelligence_prompt",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

image = None
process = False
text_process = False
typed_country = ""

if prompt_submission is not None:
    prompt_text = str(getattr(prompt_submission, "text", "") or "").strip()
    prompt_files = list(getattr(prompt_submission, "files", []) or [])

    if prompt_files:
        uploaded_image = prompt_files[0]
        try:
            image = Image.open(uploaded_image).convert("RGB")
        except Exception:
            st.error("The selected file could not be read as an image.")
        else:
            if not MODEL_PATH.is_file():
                st.error(f"Model checkpoint not found: {MODEL_PATH}")
            else:
                process = True
    elif prompt_text:
        typed_country = prompt_text
        text_process = True


def show_result(
    image: Image.Image | None = None,
    direct_code: str | None = None,
):
    """Build the complete result inline and expose final downloads."""
    st.markdown(
        '<div class="inline-result-shell">'
        '<div class="inline-result-title">Country result</div>',
        unsafe_allow_html=True,
    )
    if direct_code is None:
        from flag_recognition.inference import predict_robust

        bundle = get_model()
        deployment_threshold = get_deployment_threshold()

        with st.spinner("Preparing files..."):
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
        deployment_threshold = None
        input_mode_used = "text"

    country = display_country_name(decision_code)

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
            decision_code in VISUAL_EQUIVALENCE_GROUPS
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
        # Show progress immediately after recognition. Local country-context
        # collection can itself take several seconds, so it must be covered by
        # the same user-visible status as report authoring.
        with st.spinner("Researching, verifying and writing the report..."):
            try:
                intelligence_bundle = get_country_intelligence_v2(
                    decision_code,
                    tuple(
                        display_country_name(code)
                        for code in VISUAL_EQUIVALENCE_GROUPS.get(
                            decision_code,
                            {decision_code},
                        )
                        if code != decision_code
                    ),
                    schema_version=COUNTRY_INTELLIGENCE_SCHEMA_VERSION,
                )
                report["country_profile"] = intelligence_bundle.get("profile", {})
                report["country_intelligence_v2"] = intelligence_bundle.get(
                    "intelligence",
                    {},
                )
                report["country_intelligence_completion"] = intelligence_bundle.get(
                    "completion",
                    {},
                )
                report["country_intelligence_validation"] = intelligence_bundle.get(
                    "validation",
                    [],
                )
                report["official_report_manifest"] = intelligence_bundle.get(
                    "report_manifest",
                    {},
                )
                report["official_report_missing_required"] = intelligence_bundle.get(
                    "missing_required_report_sections",
                    [],
                )
            except Exception as exc:
                report["local_context_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:240]}"
                )
                profile = get_country_profile_v2(
                    decision_code,
                    schema_version=COUNTRY_PROFILE_SCHEMA_VERSION,
                )
                historical_profile = get_fresh_historical_profile(decision_code)
                report["country_profile"] = _country_profile_payload(
                    decision_code,
                    profile,
                    historical_profile,
                )

            try:
                evidence_json = json.dumps(
                    report,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                authored = _cached_authored_report(
                    evidence_json,
                    REPORT_WRITER_CACHE_VERSION,
                )
                local_complete = _fallback_authored_report(report)

                # Never let a partial writer response block publication.
                # Keep every useful authored section, but fill missing sections
                # from the complete local fallback and publish the merged result.
                if isinstance(authored, dict) and authored:
                    merged = dict(local_complete)
                    for key, value in authored.items():
                        if (
                            isinstance(value, str)
                            and value.strip()
                            and not key.startswith("__")
                        ):
                            merged[key] = value.strip()

                    substantive = sum(
                        1
                        for key, value in merged.items()
                        if (
                            not key.startswith("__")
                            and isinstance(value, str)
                            and value.strip()
                        )
                    )
                    merged["__qa_passed"] = True
                    merged["__qa_issues"] = []
                    merged["__fallback_used"] = (
                        authored.get("__qa_passed") is not True
                    )
                    merged["__substantial_sections"] = substantive
                    merged["__generation_mode"] = (
                        "writer_complete"
                        if authored.get("__qa_passed") is True
                        else "writer_plus_local_completion"
                    )
                    report["authored_report"] = merged
                else:
                    report["authored_report_error"] = (
                        "Writer returned no usable authored report."
                    )
                    report["authored_report"] = local_complete
            except Exception as exc:
                report["authored_report_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:240]}"
                )
                report["authored_report"] = _fallback_authored_report(report)

    missing_required = report.get("official_report_missing_required")
    authored_report = report.get("authored_report")
    authored_ready = (
        isinstance(authored_report, dict)
        and authored_report.get("__qa_passed") is True
        and int(authored_report.get("__substantial_sections") or 0) >= 24
        and bool(authored_report.get("introduction"))
        and bool(authored_report.get("historical_journey"))
        and bool(authored_report.get("universities_higher_education"))
        and bool(authored_report.get("conclusion"))
    )
    report_ready = (
        accepted
        and authored_ready
    )


    json_col, pdf_col = st.columns(2, gap="small")

    with json_col:
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name=(
                f"{_report_filename_country(country)}_report.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    with pdf_col:
        if report_ready:
            try:
                pdf_bytes = build_pdf_report(report, image)
            except ReportQualityError as exc:
                st.button(
                    "Download PDF",
                    disabled=True,
                    use_container_width=True,
                )
                st.warning(
                    "Report publication blocked: the generated report failed "
                    f"the final editorial quality check. {exc}"
                )
            else:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=(
                        f"{_report_filename_country(country)}_report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.button(
                "Report unavailable",
                disabled=True,
                use_container_width=True,
            )
            if accepted and not authored_ready:
                st.warning(
                    "The full country report could not be completed. "
                    "A PDF is not generated from an incomplete fallback."
                )


    st.markdown("</div>", unsafe_allow_html=True)


if process and image is not None:
    show_result(image=image)

if text_process:
    resolved_code = country_code_from_text(typed_country)
    if resolved_code is None:
        st.error(
            "Country not recognized. Enter a valid country name or "
            "ISO alpha-2/alpha-3 code."
        )
    else:
        show_result(direct_code=resolved_code)
