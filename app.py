"""Product-grade Streamlit interface for worldwide flag recognition."""

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


@st.cache_data(show_spinner=False)
def get_deployment_threshold() -> float:
    if not CONFIG_PATH.is_file():
        return float(get_model().unknown_threshold)

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    return float(
        config.get("open_set", {}).get(
            "deployment_threshold",
            get_model().unknown_threshold,
        )
    )


def format_population(value: int | None) -> str:
    if value is None:
        return "Not available"
    return f"{value:,}".replace(",", " ")


def format_area(value: float | None) -> str:
    if value is None:
        return "Not available"
    return f"{value:,.0f} km²".replace(",", " ")


@st.cache_resource
def get_model():
    return load_inference_bundle(MODEL_PATH, device="cpu")


@st.cache_data(ttl=3600, show_spinner=False)
def get_country_profile(country_code: str):
    return fetch_country_profile(country_code)


st.set_page_config(
    page_title="Flag Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #eef3fb;
            --surface: rgba(255,255,255,.78);
            --surface-strong: rgba(255,255,255,.92);
            --text: #0b1220;
            --muted: #667085;
            --line: rgba(148,163,184,.22);
            --blue: #2563eb;
            --blue-2: #4f46e5;
            --cyan: #06b6d4;
            --success: #16a34a;
            --warning: #b45309;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(37,99,235,.14), transparent 28%),
                radial-gradient(circle at 88% 10%, rgba(79,70,229,.16), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(6,182,212,.08), transparent 30%),
                linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        [data-testid="stToolbar"] {
            top: .7rem;
            right: 1rem;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        .product-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: .78rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,.8);
            border-radius: 18px;
            background: rgba(255,255,255,.72);
            box-shadow:
                0 12px 40px rgba(15,23,42,.06),
                inset 0 1px 0 rgba(255,255,255,.75);
            backdrop-filter: blur(20px);
        }

        .product-left {
            display: flex;
            align-items: center;
            gap: .8rem;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background:
                linear-gradient(135deg, rgba(255,255,255,.18), rgba(255,255,255,0)),
                linear-gradient(135deg, #2563eb 0%, #4f46e5 55%, #06b6d4 100%);
            color: #fff;
            font-weight: 900;
            font-size: .83rem;
            box-shadow: 0 10px 24px rgba(37,99,235,.28);
        }

        .brand-name {
            font-size: .98rem;
            line-height: 1.1;
            font-weight: 850;
            color: var(--text);
            letter-spacing: -.015em;
        }

        .brand-meta {
            margin-top: .18rem;
            font-size: .73rem;
            color: var(--muted);
        }

        .top-status {
            display: flex;
            align-items: center;
            gap: .45rem;
            font-size: .75rem;
            font-weight: 800;
            color: #475467;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 0 5px rgba(34,197,94,.10);
        }

        .page-head {
            position: relative;
            overflow: hidden;
            margin: 0 0 1.15rem 0;
            padding: 2rem 2rem 1.85rem 2rem;
            min-height: 290px;
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,.78);
            background:
                linear-gradient(120deg, rgba(255,255,255,.96) 0%, rgba(248,250,252,.93) 50%, rgba(239,246,255,.88) 100%);
            box-shadow:
                0 22px 60px rgba(15,23,42,.08),
                inset 0 1px 0 rgba(255,255,255,.9);
        }

        .page-head::before {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            border-radius: 50%;
            right: -70px;
            top: -110px;
            background: radial-gradient(circle, rgba(59,130,246,.28) 0%, rgba(79,70,229,.14) 38%, transparent 70%);
            filter: blur(2px);
        }

        .page-head::after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            border-radius: 50%;
            right: 170px;
            bottom: -110px;
            background: radial-gradient(circle, rgba(6,182,212,.18) 0%, transparent 72%);
        }

        .hero-grid {
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: minmax(0,1.5fr) minmax(280px,.75fr);
            gap: 2rem;
            align-items: center;
            min-height: 245px;
        }

        .page-kicker {
            color: var(--blue);
            font-size: .69rem;
            font-weight: 900;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .7rem;
        }

        .page-title {
            margin: 0;
            color: #0a1020;
            font-size: clamp(2.5rem, 5vw, 4.65rem);
            line-height: .94;
            letter-spacing: -.055em;
            font-weight: 920;
            max-width: 780px;
        }

        .page-subtitle {
            margin-top: .95rem;
            color: #64748b;
            font-size: 1rem;
            line-height: 1.62;
            max-width: 700px;
        }

        .hero-badges {
            display: flex;
            gap: .55rem;
            flex-wrap: wrap;
            margin-top: 1.15rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            padding: .43rem .72rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,.25);
            background: rgba(255,255,255,.72);
            color: #334155;
            font-size: .72rem;
            font-weight: 800;
            box-shadow: 0 6px 18px rgba(15,23,42,.04);
        }

        .hero-orbit {
            position: relative;
            width: 250px;
            height: 250px;
            margin-left: auto;
        }

        .hero-core {
            position: absolute;
            inset: 55px;
            border-radius: 28px;
            display: grid;
            place-items: center;
            font-size: 2.2rem;
            background:
                linear-gradient(145deg, rgba(255,255,255,.98), rgba(239,246,255,.84));
            border: 1px solid rgba(147,197,253,.55);
            box-shadow:
                0 20px 50px rgba(37,99,235,.16),
                inset 0 1px 0 rgba(255,255,255,.95);
        }

        .hero-ring {
            position: absolute;
            inset: 18px;
            border-radius: 50%;
            border: 1px solid rgba(96,165,250,.30);
        }

        .hero-ring.r2 {
            inset: 0;
            border-style: dashed;
            opacity: .65;
        }

        .signal-dot {
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg,#2563eb,#06b6d4);
            box-shadow: 0 0 0 6px rgba(37,99,235,.08);
        }

        .signal-dot.a { top: 38px; left: 32px; }
        .signal-dot.b { right: 28px; top: 95px; }
        .signal-dot.c { bottom: 32px; left: 95px; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(255,255,255,.8) !important;
            border-radius: 22px !important;
            background: rgba(255,255,255,.74);
            box-shadow:
                0 14px 34px rgba(15,23,42,.05),
                inset 0 1px 0 rgba(255,255,255,.8);
            backdrop-filter: blur(16px);
        }

        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            margin-bottom: .9rem;
        }

        .panel-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: -.015em;
        }

        .panel-copy {
            color: var(--muted);
            font-size: .79rem;
            line-height: 1.45;
            margin-top: .22rem;
        }

        div[data-testid="stFileUploader"] {
            border: 0;
            background: transparent;
            padding: 0;
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 230px;
            border: 1.5px dashed rgba(37,99,235,.36);
            border-radius: 20px;
            background:
                radial-gradient(circle at 50% 0%, rgba(59,130,246,.11), transparent 44%),
                linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(248,250,252,.86) 100%);
            transition: all .22s ease;
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #2563eb;
            transform: translateY(-2px);
            box-shadow:
                0 16px 36px rgba(37,99,235,.10),
                inset 0 0 0 1px rgba(37,99,235,.08);
        }

        div[data-testid="stFileUploaderDropzone"] button {
            border-radius: 12px;
            border: 1px solid rgba(37,99,235,.16);
            background: rgba(255,255,255,.95);
            font-weight: 800;
            box-shadow: 0 8px 22px rgba(15,23,42,.08);
        }

        .config-stack {
            display: grid;
            gap: .62rem;
        }

        .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .8rem;
            padding: .82rem .9rem;
            border: 1px solid rgba(148,163,184,.20);
            border-radius: 14px;
            background:
                linear-gradient(180deg, rgba(255,255,255,.88) 0%, rgba(248,250,252,.76) 100%);
        }

        .config-label {
            color: #64748b;
            font-size: .75rem;
        }

        .config-value {
            color: #111827;
            font-size: .8rem;
            font-weight: 850;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.20);
            border-radius: 16px;
            background: rgba(255,255,255,.82);
            padding: .95rem 1rem;
            box-shadow: 0 8px 22px rgba(15,23,42,.035);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-size: .68rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 900;
            letter-spacing: -.025em;
        }

        .decision-card {
            border-radius: 14px;
            padding: .9rem 1rem;
            margin-top: .75rem;
            font-size: .86rem;
            font-weight: 800;
        }

        .decision-known {
            color: #166534;
            border: 1px solid rgba(34,197,94,.28);
            background: rgba(240,253,244,.86);
        }

        .decision-unknown {
            color: #92400e;
            border: 1px solid rgba(245,158,11,.28);
            background: rgba(255,251,235,.88);
        }

        .section-title {
            color: var(--text);
            font-size: 1.32rem;
            font-weight: 900;
            letter-spacing: -.025em;
            margin-bottom: .7rem;
        }

        .country-heading {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin-bottom: .75rem;
        }

        .country-name {
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 900;
            letter-spacing: -.03em;
        }

        .country-code {
            padding: .24rem .48rem;
            border-radius: 8px;
            background: rgba(37,99,235,.10);
            color: var(--blue);
            font-size: .69rem;
            font-weight: 900;
        }

        .technical {
            padding: .85rem .95rem;
            border-radius: 13px;
            background: rgba(248,250,252,.9);
            border: 1px solid rgba(148,163,184,.20);
            color: #64748b;
            font-size: .76rem;
            line-height: 1.55;
        }

        button[kind="secondary"] {
            border-radius: 12px !important;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .hero-orbit {
                display: none;
            }
            .page-head {
                min-height: auto;
                padding: 1.5rem;
            }
            .page-title {
                font-size: 2.65rem;
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

if "policy" not in st.session_state:
    st.session_state.policy = "Calibrated"
deployment_threshold = get_deployment_threshold()

if "custom_threshold" not in st.session_state:
    st.session_state.custom_threshold = deployment_threshold
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "live_enrichment" not in st.session_state:
    st.session_state.live_enrichment = True
if "show_candidates" not in st.session_state:
    st.session_state.show_candidates = True
if "show_technical" not in st.session_state:
    st.session_state.show_technical = False


def get_effective_threshold() -> float:
    if st.session_state.policy == "Strict":
        return min(0.99, deployment_threshold + 0.05)
    if st.session_state.policy == "Custom":
        return float(st.session_state.custom_threshold)
    return deployment_threshold


effective_threshold = get_effective_threshold()

top_left, top_right = st.columns([8.4, 1.6], gap="small")

with top_left:
    st.markdown(
        """
        <div class="product-bar">
            <div class="product-left">
                <div class="brand-icon">FI</div>
                <div>
                    <div class="brand-name">Flag Intelligence</div>
                    <div class="brand-meta">Worldwide recognition · decision-aware</div>
                </div>
            </div>
            <div class="top-status">
                <span class="status-dot"></span>
                Model online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    with st.popover("Settings", use_container_width=True):
        st.markdown("#### Decision settings")
        st.selectbox(
            "Policy",
            ["Calibrated", "Strict", "Custom"],
            key="policy",
            help="Calibrated uses the checkpoint threshold. Strict raises the gate.",
        )

        if st.session_state.policy == "Custom":
            st.slider(
                "Acceptance threshold",
                min_value=0.05,
                max_value=0.99,
                step=0.01,
                key="custom_threshold",
            )

        st.slider(
            "Candidate depth",
            min_value=3,
            max_value=10,
            step=1,
            key="top_k",
        )

        st.divider()
        st.markdown("#### Experience")
        st.toggle("Live country enrichment", key="live_enrichment")
        st.toggle("Show candidate ranking", key="show_candidates")
        st.toggle("Show technical details", key="show_technical")

effective_threshold = get_effective_threshold()

st.markdown(
    f"""
    <div class="page-head">
        <div class="hero-grid">
            <div>
                <div class="page-kicker">AI FLAG INTELLIGENCE</div>
                <h1 class="page-title">Recognize flags with confidence.</h1>
                <div class="page-subtitle">
                    A modern visual intelligence workspace for flag recognition,
                    uncertainty-aware decisions, ranked predictions, and live country context.
                </div>
                <div class="hero-badges">
                    <span class="hero-badge">250 classes</span>
                    <span class="hero-badge">{deployment_threshold:.1%} production threshold</span>
                    <span class="hero-badge">Top {st.session_state.top_k} candidates</span>
                </div>
            </div>
            <div class="hero-orbit">
                <div class="hero-ring r2"></div>
                <div class="hero-ring"></div>
                <div class="signal-dot a"></div>
                <div class="signal-dot b"></div>
                <div class="signal-dot c"></div>
                <div class="hero-core">🌐</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, config_col = st.columns([1.65, 0.75], gap="large")

with upload_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-head">
                <div>
                    <div class="panel-title">Input image</div>
                    <div class="panel-copy">JPG, PNG or WebP · one image per analysis</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

with config_col:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-head">
                <div>
                    <div class="panel-title">Active configuration</div>
                    <div class="panel-copy">Applied to this analysis</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="config-stack">
                <div class="config-row">
                    <div class="config-label">Decision policy</div>
                    <div class="config-value">{st.session_state.policy}</div>
                </div>
                <div class="config-row">
                    <div class="config-label">Threshold</div>
                    <div class="config-value">{effective_threshold:.1%}</div>
                </div>
                <div class="config-row">
                    <div class="config-label">Candidate depth</div>
                    <div class="config-value">Top {st.session_state.top_k}</div>
                </div>
                <div class="config-row">
                    <div class="config-label">Country enrichment</div>
                    <div class="config-value">{"On" if st.session_state.live_enrichment else "Off"}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if uploaded_file is None:
    st.markdown("")
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-head">
                <div>
                    <div class="panel-title">Ready for analysis</div>
                    <div class="panel-copy">
                        Upload a flag image to generate a decision, confidence score and ranked candidates.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.metric("Worldwide classes", "250")
        with c2:
            st.metric("Deployment threshold", f"{deployment_threshold:.1%}")
        with c3:
            st.metric("Runtime", "CPU")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Analyzing image..."):
    prediction = predict_image(
        image,
        bundle,
        top_k=st.session_state.top_k,
    )

decision_is_known = prediction.top1_confidence >= effective_threshold
decision_label = (
    country_name_from_code(prediction.top1_country)
    if decision_is_known
    else "Unknown"
)
top_candidate = country_name_from_code(prediction.top1_country)

st.markdown("")
result_head_col, result_action_col = st.columns([8.5, 1.5], gap="small")

with result_head_col:
    st.markdown(
        """
        <div class="page-kicker">Result</div>
        <div class="section-title">Recognition decision</div>
        """,
        unsafe_allow_html=True,
    )

with result_action_col:
    if st.button("Reset analysis", use_container_width=True):
        st.rerun()

preview_col, decision_col = st.columns([0.9, 1.55], gap="large")

with preview_col:
    with st.container(border=True):
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} × {image.height} px")

with decision_col:
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4, gap="small")

        with m1:
            st.metric("Decision", decision_label)
        with m2:
            st.metric("Top candidate", top_candidate)
        with m3:
            st.metric("Confidence", f"{prediction.top1_confidence:.1%}")
        with m4:
            st.metric("Latency", f"{prediction.inference_ms:.1f} ms")

        st.progress(int(round(prediction.top1_confidence * 100)))

        if decision_is_known:
            st.markdown(
                '<div class="decision-card decision-known">Accepted · confidence meets the active policy.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="decision-card decision-unknown">Rejected · candidate shown for inspection only.</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            f"Active threshold {effective_threshold:.1%} · deployment threshold {deployment_threshold:.1%}"
        )

        if st.session_state.show_candidates:
            ranking = pd.DataFrame(
                [
                    {
                        "Rank": rank,
                        "Code": country.upper(),
                        "Country": country_name_from_code(country),
                        "Confidence": confidence,
                    }
                    for rank, (country, confidence) in enumerate(
                        prediction.top5,
                        start=1,
                    )
                ]
            )

            with st.expander("Candidate ranking", expanded=True):
                st.dataframe(
                    ranking.style.format({"Confidence": "{:.1%}"}),
                    hide_index=True,
                    use_container_width=True,
                )

        report = {
            "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
            "decision": decision_label,
            "top_candidate": top_candidate,
            "confidence": prediction.top1_confidence,
            "latency_ms": prediction.inference_ms,
            "active_threshold": effective_threshold,
            "checkpoint_threshold": bundle.unknown_threshold,
            "deployment_threshold": deployment_threshold,
            "policy": st.session_state.policy,
            "accepted": decision_is_known,
            "top_candidates": [
                {
                    "country": country_name_from_code(country),
                    "code": country,
                    "confidence": confidence,
                }
                for country, confidence in prediction.top5
            ],
        }

        st.download_button(
            "Download analysis report",
            data=json.dumps(report, indent=2),
            file_name="flag_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.session_state.show_technical:
            st.markdown(
                f"""
                <div class="technical">
                    Checkpoint: {MODEL_PATH}<br>
                    Policy: {st.session_state.policy}<br>
                    Active threshold: {effective_threshold:.4f}<br>
                    Deployment threshold: {deployment_threshold:.4f}<br>
                    Device: {bundle.device}
                </div>
                """,
                unsafe_allow_html=True,
            )

if not decision_is_known:
    st.stop()

if not st.session_state.live_enrichment:
    st.info("Country enrichment is disabled in Settings.")
    st.stop()

try:
    with st.spinner("Loading country profile..."):
        profile = get_country_profile(prediction.top1_country)

    st.markdown("")
    st.markdown('<div class="page-kicker">Country profile</div>', unsafe_allow_html=True)
    st.markdown(
        (
            '<div class="country-heading">'
            f'<div class="country-name">{profile.name}</div>'
            f'<div class="country-code">{prediction.top1_country.upper()}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "Government", "Geography"])

    with tabs[0]:
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            st.metric("Capital", profile.capital)
        with k2:
            st.metric("Population", format_population(profile.population.value))
        with k3:
            st.metric("Currency", profile.currency)
        with k4:
            st.metric("Area", format_area(profile.area_km2))

        col_a, col_b = st.columns([1.0, 1.35], gap="medium")
        with col_a:
            with st.container(border=True):
                st.markdown("##### Official language(s)")
                st.write(profile.official_languages)
        with col_b:
            with st.container(border=True):
                st.markdown("##### Country profile")
                st.write(profile.overview)

    with tabs[1]:
        gov_a, gov_b = st.columns(2, gap="medium")
        with gov_a:
            with st.container(border=True):
                st.markdown("##### Government form")
                st.write(profile.government_form)
                st.divider()
                st.markdown("##### Head of State")
                st.write(profile.head_of_state)
                if profile.head_of_state_office != "Not available":
                    st.caption(profile.head_of_state_office)

        with gov_b:
            with st.container(border=True):
                st.markdown("##### Head of Government")
                st.write(profile.head_of_government)
                if profile.head_of_government_office != "Not available":
                    st.caption(profile.head_of_government_office)
                st.divider()
                st.markdown("##### Practical")
                st.write(f"Calling code: {profile.calling_code}")
                st.write(f"Internet domain: {profile.internet_domain}")
                st.write(f"Driving side: {profile.driving_side}")

    with tabs[2]:
        map_col, info_col = st.columns([1.45, 1.0], gap="large")
        with map_col:
            with st.container(border=True):
                if profile.latitude is not None and profile.longitude is not None:
                    st.map(
                        pd.DataFrame(
                            [{"lat": profile.latitude, "lon": profile.longitude}]
                        ),
                        latitude="lat",
                        longitude="lon",
                        zoom=3,
                        use_container_width=True,
                    )
                else:
                    st.info("Geographic coordinates are not available.")

        with info_col:
            g1, g2 = st.columns(2)
            with g1:
                st.metric("Continent", profile.continent)
            with g2:
                st.metric("Area", format_area(profile.area_km2))

            if profile.latitude is not None and profile.longitude is not None:
                st.caption(
                    f"Reference coordinates · {profile.latitude:.3f}, {profile.longitude:.3f}"
                )

    st.caption(
        "Live metadata: Wikidata · Population: World Bank · Overview: Wikipedia · cached for one hour."
    )

except (requests.RequestException, LookupError, ValueError) as error:
    st.warning("Country data could not be loaded right now.")
    st.caption(str(error))
