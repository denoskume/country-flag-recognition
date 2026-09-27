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
    Image as PDFImage,
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


def _pdf_text(value: object) -> str:
    if value is None:
        return "Not available"

    text = str(value).strip()
    return text if text else "Not available"


def build_pdf_report(
    report: dict[str, object],
    image: Image.Image,
) -> bytes:
    """Build a printable A4 recognition report."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Flag Intelligence - Recognition Report",
        author="Flag Intelligence",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FlagTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=4 * mm,
    )
    subtitle_style = ParagraphStyle(
        "FlagSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.HexColor("#667085"),
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "FlagSection",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#101828"),
    )
    body_style = ParagraphStyle(
        "FlagBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    )
    small_style = ParagraphStyle(
        "FlagSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475467"),
    )

    story = [
        Paragraph("Flag Intelligence", title_style),
        Paragraph("Country Flag Recognition Report", subtitle_style),
    ]

    image_buffer = BytesIO()
    image.copy().convert("RGB").thumbnail((1200, 900))
    image.save(image_buffer, format="JPEG", quality=90)
    image_buffer.seek(0)

    preview = PDFImage(image_buffer)
    preview._restrictSize(160 * mm, 70 * mm)
    story.extend([preview, Spacer(1, 5 * mm)])

    confidence = float(report.get("confidence", 0.0))
    threshold = float(report.get("deployment_threshold", 0.0))
    status = "Accepted" if report.get("accepted") else "Rejected"

    recognition_rows = [
        ["Status", status],
        ["Decision", _pdf_text(report.get("decision"))],
        ["Top candidate", _pdf_text(report.get("top_candidate"))],
        ["Country code", _pdf_text(report.get("country_code")).upper()],
        ["Confidence", f"{confidence:.2%}"],
        ["Deployment threshold", f"{threshold:.2%}"],
        ["Generated at", _pdf_text(report.get("generated_at"))],
    ]

    recognition_table = Table(
        recognition_rows,
        colWidths=[52 * mm, 108 * mm],
        hAlign="LEFT",
    )
    recognition_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F4F7")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#101828")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.extend([
        Paragraph("Recognition", section_style),
        recognition_table,
        Paragraph("Top candidates", section_style),
    ])

    top_rows = [["Rank", "Country", "Code", "Confidence"]]
    for index, candidate in enumerate(report.get("top_candidates", []), start=1):
        top_rows.append([
            str(index),
            _pdf_text(candidate.get("country")),
            _pdf_text(candidate.get("code")).upper(),
            f"{float(candidate.get('confidence', 0.0)):.2%}",
        ])

    top_table = Table(
        top_rows,
        colWidths=[16 * mm, 90 * mm, 20 * mm, 34 * mm],
        repeatRows=1,
    )
    top_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#101828")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(top_table)

    profile = report.get("country_profile")
    if isinstance(profile, dict):
        story.append(Paragraph("Country profile", section_style))

        profile_fields = [
            ("Name", profile.get("name")),
            ("Capital", profile.get("capital")),
            ("Population", profile.get("population")),
            ("Currency", profile.get("currency")),
            ("Official language(s)", profile.get("official_languages")),
            ("Continent", profile.get("continent")),
            ("Area", (
                f"{float(profile['area_km2']):,.0f} km²"
                if profile.get("area_km2") is not None
                else "Not available"
            )),
            ("National Day", profile.get("national_day")),
            ("Independence", profile.get("independence_day")),
            ("National motto", profile.get("national_motto")),
            ("National anthem", profile.get("national_anthem")),
            ("Government form", profile.get("government_form")),
            ("Head of State", profile.get("head_of_state")),
            ("Head of State office", profile.get("head_of_state_office")),
            ("Head of Government", profile.get("head_of_government")),
            ("Head of Government office", profile.get("head_of_government_office")),
            ("Calling code", profile.get("calling_code")),
            ("Internet domain", profile.get("internet_domain")),
            ("Driving side", profile.get("driving_side")),
            ("Latitude", profile.get("latitude")),
            ("Longitude", profile.get("longitude")),
        ]

        profile_rows = [
            [
                Paragraph(f"<b>{label}</b>", body_style),
                Paragraph(_pdf_text(value), body_style),
            ]
            for label, value in profile_fields
        ]

        profile_table = Table(
            profile_rows,
            colWidths=[52 * mm, 108 * mm],
            hAlign="LEFT",
        )
        profile_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F9FAFB")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#EAECF0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(profile_table)

        overview = _pdf_text(profile.get("overview"))
        if overview != "Not available":
            story.extend([
                Paragraph("Overview", section_style),
                Paragraph(overview, small_style),
            ])

    story.extend([
        Spacer(1, 5 * mm),
        Paragraph(
            "Sources: Wikidata, World Bank and Wikipedia where available.",
            small_style,
        ),
    ])

    document.build(story)
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
            top_k=5,
        )

    accepted = prediction.top1_confidence >= deployment_threshold
    country = display_country_name(prediction.top1_country)

    preview_col, result_col = st.columns([.9, 1.1], gap="large")

    with preview_col:
        st.image(image, use_container_width=True)

    with result_col:
        st.markdown(
            f'<div class="result-name">{country if accepted else "Unknown"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-code">Top candidate · {country} ({prediction.top1_country.upper()})</div>',
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Confidence", f"{prediction.top1_confidence:.1%}")
        with m2:
            st.metric("Threshold", f"{deployment_threshold:.1%}")

        if accepted:
            st.markdown(
                '<div class="decision-ok">Accepted prediction</div>',
                unsafe_allow_html=True,
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
                prediction.top5,
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
            profile = get_country_profile(prediction.top1_country)

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
        "country_code": prediction.top1_country,
        "confidence": prediction.top1_confidence,
        "deployment_threshold": deployment_threshold,
        "top_candidates": [
            {
                "country": display_country_name(code),
                "code": code,
                "confidence": confidence,
            }
            for code, confidence in prediction.top5
        ],
    }

    if accepted:
        try:
            profile = get_country_profile(prediction.top1_country)
            report["country_profile"] = {
                "name": profile.name,
                "capital": profile.capital,
                "population": profile.population.value,
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
