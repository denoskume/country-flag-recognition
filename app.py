# UI restoration redeploy marker
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
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
import pydeck as pdk
from PIL import Image, ImageDraw
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
    PageBreak,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PDF_LOGO_BASE64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCABgAGADASIAAhEBAxEB/8QAHAAAAgIDAQEAAAAAAAAAAAAAAAgGBwQFCQMC/8QAPhAAAQMDAgMFBQUHAgcAAAAAAQIDBAUGEQAHEiExCBMiQWEUUXGBkRUyUqKxIyRCcoKSoRZiMzZTdLPS4f/EABwBAAICAwEBAAAAAAAAAAAAAAUHBAYAAQIDCP/EADQRAAECBAMFBgUEAwAAAAAAAAECAwAEBRESITEGQVFxkQciYYGhsRMUJMHRMkJSYhXh8P/aAAwDAQACEQMRAD8AcvRo0ayMg0awLgrNLoFKeqtZnMwobIyt11WB8B7yfIDmdLRuX2jKrOddgWUx9nROafbX0BT6/VKTyQPjk/DUd+abYHfMHKLs7PVleGWTkNVHJI8/sLmGdqNQgU2OZFQmxobI6uPupQn6k6iU7drbeG4UPXfTSode6WXf8pB0j9Zq1UrMtUyrVGVPkK5lyQ6pw/56awtC11dV+6nrDHley9gJ+pfJP9QB739hD2Qt29t5jgQzd9NSo9O9UWx9VAal1OqECpRxIp06NMZPRxh1LifqDrnLrNo9WqdGmJmUmoyoEhJyHI7pbV/jWIqyv3JjJrswYKfpnyD/AGAPtb2MdFtGlZ207RlWgOtQL1Y+0YnJPtrCAl9v1Ukclj4YPx0y9ArNLr9KZqtHmszYbwyh1pWQfQ+4jzB5jRRiabfHdMLitbOz1GXhmU5HRQzSfP7GxjP0aNGpEA4Nau6q9TLYoEut1iQGIcVHEtXmo+SUjzUTyA1tNKD2pb+cuO7VW3AfJpVIcKF8J5PSOilH3hPNI/q9+o03MBhvFv3RYNmqEutTwY0SM1HgPydB13RDt2dxazuDXTKmrUxT2VEQ4SVeBpPvPvWfM/IctQvRo1V1rUtRUo3MfR0pKMybKWGE4Up0AifRNtJsvaZu+404POOzRFbp6GSVry53YwrPNRUemOnnqWW12cbyqMNEmqzqfRysZDLhLrg/mCeQ+upNZ1RZo/ZepdWkIcWzCrrUhaW/vKCJiVEDPny1unNw96a8PbLX25aiU9XiaVNB7xafI+JSP8DRNEvLjCVAm4BsLwuZyu1tSnkS60JSlxacSykWAtZIvr0Jitrq7PF8UmOqRTVwq22kZKI6yh3+1WM/Ik6qipU6oUyUqLUYMqG+k4U2+0pCh8iNM9H3suq2pKGNybDl09lRwJcRBCfoolKvkrU1h7vbX1OKl9dywUYGeCU0pC0/JSf01tUpLOHuLw+B/wB2jljaivySfqpYPJOikZg+acQ9BCdR7YuB+gy68ikyk0uIlJelLQUNjKgkAE/eOSOQzre7S7jVjb6uCTEUuRTXlD2yEVeF1P4h7ljyPyPLVs7/AO8lrV2z5lp2531RVKKA5KCC200ErSrw5GVE8OOgHrpcdQnQlhwfCVe2/wAYttMcercisVKXwJUbBJ1w2GZ33ve2Q3c46I2vXaZctBiVukSA/DlI40K8x70keRB5Ee8a2WlE7LF/OW9dYtioPn7LqzgS3xHkzI6JI9wV90+vDpu9WGUmA+3i374Rm0tCXRZ5TBzSc0niPyND1iNbpXD/AKV2/rNdSQHY0Y9xn/qq8KPzEaQNxa3HFOOKK1qJUpROSSep02vbFqCo22sKChWPbKkgK9UpSpX6hOlI0IqrmJ0J4CGj2aySWqauYtmtXonIet4NGjRoZDFhk7Hmxad2ZKPUJyuGLFr7Lz6uHiwhM1JUcefIdNSSTvwt9Rfoe3lyVOng8pXdlCVD3gBKv11G7FdhMdmWjP1ItJhN19hUgujKA2JqSriHmMZ1M3O0HtszL9lblVFbSTwh5uEe7x6DIVj5aPNrwpT3wnIQkp2U+PMvkSinyHXNCQBmn+Ivc8/KPe196LAuoLpVWUaRIX4HIlWbSG1enEcp+SsazJ20u1E9RqTlAp6Gz4ytiSptoj34SoJxr1kt7U7owe9dXRquQPvhfdSW/ieSx89LNv5btt2peLNItWUp2EYiXHke1d8EOlSspJ8uQTyOtvultGJwJWOP/Xjxo1NbnZsy8m47KualOZGXjdJ6jzjw36RazO4TsazxBFLjxm2v3Pm33gzxc/4j0ycnUB0aNAnFY1FVrXh0yMt8pLoYxFWEAXOp8TH004406h1pakOIUFJUDgpI5gjT/bY3CLqsGjV4kFyVGSXseTifCv8AMDrn/puex5UFStspUJZz7FUXEpHuSpKVfqVaI0pwh0p4iKH2lSSXaciYtmhXorX1AjW9tNCzaFBdAPCmoKSfiWzj9DpWNOh2pqKur7RTXmkcTlOebmADrwg8KvyqJ+Wkv1xU0kP34iJfZ0+lyjhA1SpQ65/eLgZ7Ou4DrKHUPUThWkKGZauhGfwawbj2Jveg0WRV5ztIMeOElYbkqKuagkYHAPMjTR3bRptctOFDg3PLtt1JaWZcYjiUAgjg5kcjnPy1Tu7Nt1+3LZYlvbn1eusP1CPHdhvKHAtKl5ycKPThGpL8k02kkJOmtxFepW2FRnXUIW8gEqthwKuRzGWfOIpuPAvqwNoItlV2NQ1UqXLJbfjOrW/xhfe88gDGeXTVL6aLtpf8sW//AN85/wCPSxwosidNYhRW1OyH3EtNIHVSlHAH1OoU8jA9gGgtaLhsbN/NUv5pwBKlqUpVshe+Z9Imu3W1F133S36pRUwm4rL3c8cl0o41YBPDhJzjIz8dRi66DUbXuGZQqq2hEyIvgc4DlJyAQQfMEEHThOPo2m2+tujQKZIqKzJZjyfZ2FLICjxPvHhHqcZ941H94rChVbeCyq7IZSuFMkiJOSR4VqbSpxvP8wSU/IDUlyQSGwE/qFr+cV6R24eXPLU+kfAUF4LanBnnzHqYpextjr4uqnN1JDMWlw3k8TTk5ZSpxPkQgAnHqcZ14X/sxetnU9ypSo8eoU9oZdkQllfdD3qSQFAeuMaubtA3RcbV+W1Y1HrDtAh1PgL85nwq8ThRjiyMBIGSARnOsuq7bXZWrKfokjdp6pU1KlLcIhpWtzAz3anA5kp88HWzJtnEhAJI33GseTW1lQQWZqadbQ25mEYVk4b2vcA55bz+IUfTU9i1tYsyuOkeBVRAHxDac/qNKtp0uy7RV0jaGA46jhdqDrkwgjnhRwn8qUn568aYkl+/AQW7RZhLdHwHVSgOmf2iyalDj1GnSYEtsOR5LSmnUH+JKgQR9DpAdwLZmWhd9Rt+YFcUV0htZGO8bPNCx8U4+eddBtVH2kdslXrQU1ekMg12nIPdpHWS11LfxHVPrkeeidRli83iTqIXuw20CaXOlp42bcsCeB3HluPXdFf783xT7w28hUKkUmviYxNjFff05bacltQSnP4lcQ4R555apaBb1wxXWKm9b9XEOO8FuvexucCQheFc8Y5EEfHVtQN6KJTqnMLtDmyGZMyCZDD7aQpKI8dLaiPFyWHEJUn4c8ah1U3GTKtur0pt6qp9suQ1JoFzCExTxEtEcXmVA8PTQuYLbisZVc/jSGFRGp6QY+ValrIJBuTf9VsWltIvy7NwrArDTLFz2XcMttvjeZRNoSyEhI8ahxeQHU+Q1CF3bs5Gue3bnplvTKbDguPkOMUngTIfwgIGQcK4cqVjyONaivbs2hVLvNXUzXUxH4MyI+0lhsKQH0JSFJ/aEKIweuPnqIS7hsNyxWrcbNx8VOnvzKe4plnDhWlAAd8XLmk/d8tSHZq5JBB8uFoC03Z0tNpSpt1IIsQFGwCgoKyty633RYt670bgyboqLdk0SWmlwkpDiZFKWp5s8OVKcH8HXkD5DOtu5ug5cO07C6vSK9Hr/geizY1JcVG9qQ5lhSVDlhRCQR/uI1GW95LSkVeVJnwK63F+0nZaGY5SkSkONoQQ7haSFJ4TggkYONaw7pWt9gW3Bbi1ZldHkR1ltLKOFaG5HeFIV3n4eWCnqBz1v5jMn4l73jn/AAd0NIEgUFBScQ10N7nfna+Vs/AxYCdwrDv6jwoW4Vn1JqooeUyhr2B5wKfTjjS0pscefeg8xyznrra0Pcm1KXTajQKLZNzU+DACmkpYpCyAopyorA5pPMHxcz11XEzeyhT34jr1Hmw3Vmc3MkwwhDiQ+lKEPt8+ToShOenng61tF3FtuNQK3b6XrqnqqElK4sqShD8hf7ANAHDgwc9McWBjrroTQBuFC/G2ekeCtm3FNlKpdaU3uEYyUpOKxIy4Z53JBiutu7YlXheNOt+IFfvLo75YH/DaHNa/kM/PGn8p8RiBBjwYrYbjx2ktNIHRKUjAH0Gqt7OO2Zsi31VSrMgV6ooHepPWO11DXx81euB5atnUmnyxZburUxX9ua+mqzoaZN228geJ3nluHK++DRo0aIRSIpPfbZJi63HbhtgNRa2RxPsE8LUv1z/Cv16Hz9+lTrFMqFHqLtOqsJ+FLZOHGXkFKk//AD110Y1H7zsy2bwhiLcNJYmBIw26Rwut/wAqxzH1xoZNU5LpxIyPpDB2b28fpqBLzYK2xof3D8jwOnHdHPzRplbp7MTS3FO2zcamkk5DE9rix/Wj/wBdQeZ2d9x2HCllmlyk+Sm5gGf7gNCVyL6NUwz5XbGizKbh8J8Fd33iotoreg9nbcZ9wJfapURJPNTkvix/aDqdWt2YozbiXbmuNb6QebEFvgB/rVk/lGsRIvr0T1jU1tlRZZNy+FeCe97ZdTC6USlVKt1Jqm0iC/NmOnCGWUcSj6+g9TyGmu2K2TjWitq4LkDUyu4yy0PE1D+H4l/7ug8vfqyrOs+27QgmJb1JjwkqH7RaRlxz+ZZ8R+Z1vtF5WnJaOJeZ9IWG0m3b9TQZeVBQ2df5K58B4Drug0aNGiUL+P/Z"

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
    # Page header contact - repeated on every page and right-aligned to
    # the same outer content margin as the rest of the report.
    # ------------------------------------------------------------------
    contact_y = page_height - 23 * mm

    canvas.setFillColor(colors.HexColor("#111111"))
    canvas.setFont("Helvetica", 9.3)
    canvas.drawRightString(right, contact_y, "Contact")

    canvas.setFont("Helvetica", 8.6)
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
    canvas.setFont("Helvetica", 6.8)
    canvas.drawCentredString(
        (left + right) / 2,
        18 * mm,
        "Sources: World Bank · Wikidata · REST Countries · Wikipedia · EmergencyNumberAPI",
    )
    canvas.setFont("Helvetica", 7.2)
    physical_page = int(getattr(document, "page", 1))
    if physical_page <= 2:
        footer_text = (
            "© 2026 Flag Intelligence · Version 0.1.0 · All rights reserved."
        )
    else:
        logical_page = physical_page - 2
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
                logical_page = max(int(self.page) - 2, 1)
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
            fontName="Helvetica-Bold",
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
    pdf_map.drawHeight = 82 * mm
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
        (r"&nbsp;|&amp;|&quot;", "HTML entity residue"),
        (r"\balt=", "image-alt extraction residue"),
        (r"\bthumb\|", "MediaWiki image residue"),
        (r"\bpx\s", "image-dimension residue"),
        (r"\|", "raw pipe separator"),
        (
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            "ambiguous numeric calendar date",
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
    buffer = BytesIO()
    document = FlagIntelligenceDocTemplate(
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
        fontSize=8.4,
        leading=12.0,
        textColor=colors.HexColor("#222222"),
        spaceAfter=2.2 * mm,
    )

    narrative_heading_style = ParagraphStyle(
        "NarrativeSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
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
        fontName="Helvetica-Bold",
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
        fontName="Helvetica-Bold",
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
        fontName="Helvetica",
        fontSize=9.2,
        leading=14,
        leftIndent=4 * mm,
        spaceAfter=1.2 * mm,
        textColor=colors.HexColor("#222222"),
    )

    narrative_subheading_style = ParagraphStyle(
        "NarrativeSubheading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
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
        fontName="Helvetica",
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
        """Render short factual sections as continuous report prose."""
        cleaned_sentences = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in sentences
            if sentence and sentence.strip()
        ]
        if not cleaned_sentences:
            return []
        return [
            Paragraph(
                xml_escape(" ".join(cleaned_sentences)),
                body_style,
            )
        ]

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

        def _normalize_sentence(text: str) -> str:
            """Normalize scraped prose into report-ready sentence text."""
            value = clean(text)
            if value == "Not available":
                return ""
            value = re.sub(r"^[*•\-]+\s*", "", value)
            value = re.sub(r"\s*\|\s*", ", ", value)
            value = re.sub(r"\s*;\s*", "; ", value)
            value = re.sub(r"\s+", " ", value).strip(" ;,")
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

        def _narrative_paragraph(sentences: list[str]) -> list[object]:
            """Build one coherent paragraph from connected factual sentences."""
            usable = [
                re.sub(r"\s+", " ", s).strip()
                for s in sentences
                if s and re.sub(r"\s+", " ", s).strip()
            ]
            if not usable:
                return []
            return [Paragraph(xml_escape(" ".join(usable)), body_style)]

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

                if len(chunk) >= 4:
                    paragraphs.append(
                        Paragraph(
                            xml_escape(" ".join(chunk)),
                            body_style,
                        )
                    )
                    chunk = []

            if chunk:
                paragraphs.append(
                    Paragraph(
                        xml_escape(" ".join(chunk)),
                        body_style,
                    )
                )
            return paragraphs

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
                paragraphs: list[object] = []
                chunk: list[str] = []
                char_count = 0
                for sentence in sentences:
                    chunk.append(sentence)
                    char_count += len(sentence)
                    if char_count >= 650:
                        paragraphs.append(
                            Paragraph(
                                xml_escape(" ".join(chunk)),
                                body_style,
                            )
                        )
                        chunk = []
                        char_count = 0
                if chunk:
                    paragraphs.append(
                        Paragraph(
                            xml_escape(" ".join(chunk)),
                            body_style,
                        )
                    )
                return paragraphs

            # Preserve only meaningful internal structure when the source
            # genuinely contains several distinct topics.
            flowables: list[object] = []
            for heading, summary in prepared:
                if heading and heading.lower() not in generic_headings:
                    flowables.append(
                        Paragraph(
                            xml_escape(heading),
                            narrative_subheading_style,
                        )
                    )
                flowables.append(
                    Paragraph(
                        xml_escape(summary),
                        body_style,
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
                    fontName="Helvetica-Bold",
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
                ("NATIONAL DAY", profile.get("national_day")),
            ],
            [
                ("CALLING CODE", profile.get("calling_code")),
                ("DRIVING SIDE", profile.get("driving_side")),
                ("INTERNET DOMAIN", profile.get("internet_domain")),
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
                                        fontName="Helvetica-Bold",
                                        fontSize=6.5,
                                        leading=7.5,
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
                                        fontName="Helvetica-Bold",
                                        fontSize=8.4,
                                        leading=10.2,
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
                                    fontName="Helvetica-Bold",
                                    fontSize=6.5,
                                    leading=7.5,
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
                                    fontName="Helvetica-Bold",
                                    fontSize=8.6,
                                    leading=10.5,
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
            location_map.drawHeight = 86 * mm
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
                fontName="Helvetica",
                fontSize=9.2,
                leading=14,
                leftIndent=0,
                firstLineIndent=0,
                spaceBefore=1.0 * mm,
                spaceAfter=1.0 * mm,
            )
        ]
        toc.dotsMinLevel = 0
        story.append(toc)

        story.append(PageBreak())

        overview = clean(profile.get("overview"))
        story.extend(chapter_heading(1, "Introduction"))
        intro_parts: list[object] = []
        if overview != "Not available":
            intro_parts.append(Paragraph(xml_escape(overview), body_style))
        intro_parts.append(
            Paragraph(
                xml_escape(
                    "This report provides a structured country profile based on "
                    "public, source-aware information. It brings together national "
                    "identity, physical geography, historical development, "
                    "institutions, society, culture, economy, infrastructure and "
                    "practical information. Reference years may differ between "
                    "datasets, and time-sensitive facts should be read together "
                    "with the source information provided in the report."
                ),
                body_style,
            )
        )
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
                f"France's urban geography is centred on {largest_cities}, "
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

        physical_flowables: list[object] = []
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

        add_combined_learning_section(
            "Climate, Water & Natural Resources",
            [climate_text, rivers_text, resources_text],
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

        if any(
            report_manifest.get(key, False)
            for key in ("flag", "origins", "history")
        ):
            story.append(PageBreak())

        # 3. Flag & Historical Journey
        story.extend(chapter_heading(3, "Flag & Historical Journey"))
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

                if flag_overview_flowables:
                    story.extend(
                        narrative_section(
                            "Flag Design, Adoption & Symbolism",
                            flag_overview_flowables,
                        )
                    )

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
                        flag_history_flowables: list[object] = []
                        for period, summary in rows:
                            flag_history_flowables.append(
                                Paragraph(
                                    f"<b>{xml_escape(period)}</b> - {xml_escape(summary)}",
                                    body_style,
                                )
                            )
                        story.extend(
                            narrative_section(
                                "Flag History",
                                flag_history_flowables,
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
                    timeline_flowables = _timeline_narrative(timeline_rows)
                    story.extend(
                        narrative_section(
                            "Historical Journey",
                            timeline_flowables,
                        )
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
                    "France does not follow the classical pattern of a former "
                    "colony becoming an independent state; its modern sovereignty "
                    "developed through the historical evolution of the French state."
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

        if any(
            report_manifest.get(key, False)
            for key in (
                "society", "languages_religion", "health",
                "culture", "festivals", "heritage",
            )
        ):
            story.append(PageBreak())

        # 5. People, Society & Culture
        story.extend(chapter_heading(5, "People, Society & Culture"))
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
        heritage_text = context_value("culture", "heritage_landmarks")
        if heritage_text != "Not available":
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
                            "France's major heritage sites include",
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

        if any(
            report_manifest.get(key, False)
            for key in (
                "economy", "economic_drivers", "infrastructure",
                "transport", "energy", "education", "environment",
            )
        ):
            story.append(PageBreak())

        story.extend(chapter_heading(6, "Economy, Infrastructure & Innovation"))
        gdp_source = fact_value(profile.get("gdp_source"))
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

        add_combined_learning_section(
            "Infrastructure, Transport & Energy",
            [
                context_value("infrastructure"),
                context_value("infrastructure", "transport_network"),
                context_value("infrastructure", "energy_connectivity"),
            ],
        )

        add_combined_learning_section(
            "Education, Innovation & Environment",
            [
                context_value("education_science"),
                context_value("environment"),
            ],
        )

        if any(
            report_manifest.get(key, False)
            for key in ("practical", "international", "notable_people")
        ):
            story.append(PageBreak())

        story.extend(chapter_heading(7, "International & Practical Information"))

        calling_code = fact_value(profile.get("calling_code"))
        emergency_numbers_raw = fact_value(profile.get("emergency_numbers"))
        emergency_numbers = (
            professional_inline(emergency_numbers_raw)
            if emergency_numbers_raw
            else None
        )
        driving_side = fact_value(profile.get("driving_side"))
        internet_domain = fact_value(profile.get("internet_domain"))
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
        if practical_sentences:
            story.extend(
                narrative_section(
                    "Practical & Emergency Information",
                    readable_fact_paragraph(practical_sentences),
                )
            )

        add_learning_section(
            "International Relations",
            context_value("international_relations"),
        )
        add_learning_section(
            "Notable Public Figures",
            context_value("culture", "notable_people"),
        )

        # 8. Conclusion
        story.append(PageBreak())
        story.extend(chapter_heading(8, "Conclusion"))
        conclusion_sentences = [
            (
                f"{decision} is presented in this report through a structured "
                f"profile covering geography, historical development, institutions, "
                f"society, culture, economy, infrastructure and practical information."
            )
        ]
        if clean(profile.get("capital")) != "Not available":
            conclusion_sentences.append(
                f"Its capital is {clean(profile.get('capital'))}."
            )
        if clean(profile.get("population")) != "Not available":
            conclusion_sentences.append(
                f"The population reference used in the report is {population_value}."
            )
        if clean(profile.get("government_form")) != "Not available":
            conclusion_sentences.append(
                f"The documented form of government is {clean(profile.get('government_form'))}."
            )
        conclusion_sentences.append(
            "Taken together, the preceding chapters provide a consolidated "
            "country-level reference rather than a substitute for specialised, "
            "real-time or legally authoritative information."
        )
        story.append(
            Paragraph(
                xml_escape(" ".join(conclusion_sentences)),
                body_style,
            )
        )

        # 9. Sources & Methodology
        story.extend(chapter_heading(9, "Sources & Methodology"))
        methodology_text = (
            "Flag Intelligence combines structured country facts with sourced "
            "educational context. Current source families include World Bank, "
            "Wikidata, REST Countries, Wikipedia/MediaWiki and specialised "
            "emergency-number data where available. Facts may use different "
            "reference years. Missing information is intentionally preferred "
            "over unsupported content, and time-sensitive information should be "
            "interpreted using the source and retrieval context available in the "
            "underlying record."
        )
        story.append(Paragraph(xml_escape(methodology_text), body_style))
        story.append(
            Paragraph(
                xml_escape(
                    "Source line: World Bank · Wikidata · REST Countries · "
                    "Wikipedia/MediaWiki · EmergencyNumberAPI where available."
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

    _validate_professional_report_story(story)

    document.multiBuild(
        story,
        onFirstPage=draw_pdf_watermark,
        onLaterPages=draw_pdf_watermark,
    )
    buffer.seek(0)
    return buffer.getvalue()


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

st.markdown(
    build_flag_banner_html(),
    unsafe_allow_html=True,
)

with st.container(border=True):
    input_mode = st.radio(
        "Choose input",
        ("Flag image", "Country name"),
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None
    process = False
    text_process = False
    typed_country = ""

    if input_mode == "Flag image":
        if not MODEL_PATH.is_file():
            st.error(f"Model checkpoint not found: {MODEL_PATH}")

        upload_col, preview_col = st.columns([1.35, 0.65], gap="medium")

        with upload_col:
            uploaded_file = st.file_uploader(
                "Select image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )

        with preview_col:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")

            render_fixed_upload_preview(image)

        process = st.button(
            "Process image",
            type="primary",
            use_container_width=True,
            disabled=(image is None or not MODEL_PATH.is_file()),
        )
    else:
        typed_country = st.text_input(
            "Country name",
            placeholder="Example: France, Côte d’Ivoire, Japan, BRA, XK",
        ).strip()
        st.caption(
            "Enter a country name or ISO alpha-2/alpha-3 code. "
            "The knowledge report opens directly without image recognition."
        )
        text_process = st.button(
            "Explore country",
            type="primary",
            use_container_width=True,
            disabled=not bool(typed_country),
        )


@st.dialog("Country result", width="small")
def show_result(
    image: Image.Image | None = None,
    direct_code: str | None = None,
):
    """Build the complete result privately and expose only final downloads."""
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
        try:
            with st.spinner("Preparing files..."):
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

    missing_required = report.get("official_report_missing_required")
    report_ready = (
        accepted
        and isinstance(missing_required, list)
        and not missing_required
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
                    "Official PDF withheld: the professional report quality "
                    f"gate detected an editorial issue. {exc}"
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
                "Download PDF",
                disabled=True,
                use_container_width=True,
            )


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
