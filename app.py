"""Focused Streamlit interface for worldwide flag recognition."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import strftime

import pandas as pd
from PIL import Image
import requests
import streamlit as st
import yaml


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.inference import load_inference_bundle, predict_image
from flag_recognition.taxonomy import country_name_from_code


MODEL_PATH = ROOT_DIR / "artifacts/models/worldwide_mobilenet_v3_small.pt"
CONFIG_PATH = ROOT_DIR / "configs/deployment.yaml"


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
            --bg: #f7f8fa;
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
            max-width: 760px;
            padding-top: 3.2rem;
            padding-bottom: 2rem;
        }

        .app-title {
            text-align: center;
            font-size: 1.7rem;
            font-weight: 850;
            letter-spacing: -.035em;
            color: var(--text);
            margin-bottom: .25rem;
        }

        .app-subtitle {
            text-align: center;
            color: var(--muted);
            font-size: .92rem;
            margin-bottom: 1.6rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            background: var(--surface);
            box-shadow: 0 10px 30px rgba(16,24,40,.055);
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 180px;
            border: 1.5px dashed #b9c4d5;
            border-radius: 14px;
            background: #fafbfc;
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
            font-weight: 800;
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
                padding-top: 2rem;
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

st.markdown('<div class="app-title">Flag Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Select an image and run recognition.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

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
    country = country_name_from_code(prediction.top1_country)

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
                "Country": country_name_from_code(code),
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
            with st.expander("Country information", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Capital:** {profile.capital}")
                    st.write(f"**Currency:** {profile.currency}")
                with c2:
                    st.write(f"**Continent:** {profile.continent}")
                    population = (
                        f"{profile.population.value:,}"
                        if profile.population.value is not None
                        else "Not available"
                    )
                    st.write(f"**Population:** {population}")
        except (requests.RequestException, LookupError, ValueError):
            pass

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
                "country": country_name_from_code(code),
                "code": code,
                "confidence": confidence,
            }
            for code, confidence in prediction.top5
        ],
    }

    st.download_button(
        "Download result",
        data=json.dumps(report, indent=2),
        file_name="flag_recognition_result.json",
        mime="application/json",
        use_container_width=True,
    )


if process and image is not None:
    show_result(image)
