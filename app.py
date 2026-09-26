"""Product-grade Streamlit interface for worldwide flag recognition."""

from __future__ import annotations

import json
from pathlib import Path
from time import strftime

import pandas as pd
from PIL import Image
import requests
import streamlit as st

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.inference import load_inference_bundle, predict_image
from flag_recognition.taxonomy import country_name_from_code


MODEL_PATH = Path("artifacts/models/worldwide_mobilenet_v3_small.pt")


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
            --bg: #f6f8fb;
            --surface: #ffffff;
            --surface-2: #f9fafb;
            --text: #101828;
            --muted: #667085;
            --line: #e4e7ec;
            --line-strong: #d0d5dd;
            --brand: #2f5bea;
            --brand-soft: #eef3ff;
            --success: #067647;
            --success-soft: #ecfdf3;
            --warning: #b54708;
            --warning-soft: #fffaeb;
            --danger: #b42318;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: rgba(246,248,251,.94);
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(12px);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1440px;
            padding-top: .8rem;
            padding-bottom: 2.5rem;
        }

        .product-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: .15rem 0 .95rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }

        .product-left {
            display: flex;
            align-items: center;
            gap: .75rem;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, #2f5bea, #3da0ff);
            color: #fff;
            font-weight: 900;
            font-size: .85rem;
            box-shadow: 0 8px 18px rgba(47,91,234,.18);
        }

        .brand-name {
            font-size: 1rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--text);
        }

        .brand-meta {
            margin-top: .2rem;
            font-size: .76rem;
            color: var(--muted);
        }

        .top-status {
            display: flex;
            align-items: center;
            gap: .5rem;
            font-size: .78rem;
            font-weight: 700;
            color: #475467;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #12b76a;
            box-shadow: 0 0 0 4px rgba(18,183,106,.10);
        }

        .page-head {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: .55rem 0 1rem 0;
        }

        .page-kicker {
            color: var(--brand);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .28rem;
        }

        .page-title {
            margin: 0;
            color: var(--text);
            font-size: clamp(1.9rem, 3vw, 2.75rem);
            line-height: 1.02;
            letter-spacing: -.035em;
            font-weight: 850;
        }

        .page-subtitle {
            margin-top: .45rem;
            color: var(--muted);
            font-size: .93rem;
            line-height: 1.5;
            max-width: 760px;
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: 0 8px 24px rgba(16,24,40,.035);
        }

        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
            margin-bottom: .85rem;
        }

        .panel-title {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 800;
        }

        .section-title {
            color: var(--text);
            font-size: 1.28rem;
            font-weight: 850;
            letter-spacing: -.02em;
            margin-bottom: .7rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.45;
            margin-top: .2rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            background: var(--surface);
            box-shadow: 0 8px 24px rgba(16,24,40,.035);
        }

        div[data-testid="stFileUploader"] {
            border: 0;
            background: transparent;
            padding: 0;
        }

        div[data-testid="stFileUploaderDropzone"] {
            min-height: 220px;
            border: 1.5px dashed #b8c0cc;
            border-radius: 16px;
            background:
                linear-gradient(180deg, #fbfcfe 0%, #f8fafc 100%);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--surface);
            padding: .9rem .95rem;
            min-height: 100px;
        }

        div[data-testid="stMetricLabel"] {
            color: #667085;
            font-size: .69rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 850;
        }

        .config-stack {
            display: grid;
            gap: .65rem;
        }

        .config-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .8rem;
            padding: .65rem .72rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface-2);
        }

        .config-label {
            color: var(--muted);
            font-size: .78rem;
        }

        .config-value {
            color: var(--text);
            font-size: .82rem;
            font-weight: 800;
        }

        .decision-card {
            border-radius: 14px;
            padding: .9rem 1rem;
            margin-top: .75rem;
            font-size: .88rem;
            font-weight: 750;
        }

        .decision-known {
            color: var(--success);
            border: 1px solid #abefc6;
            background: var(--success-soft);
        }

        .decision-unknown {
            color: var(--warning);
            border: 1px solid #fedf89;
            background: var(--warning-soft);
        }

        .subtle {
            color: var(--muted);
            font-size: .8rem;
        }

        .country-heading {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin-bottom: .75rem;
        }

        .country-name {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 850;
            letter-spacing: -.025em;
        }

        .country-code {
            padding: .2rem .45rem;
            border-radius: 7px;
            background: var(--brand-soft);
            color: var(--brand);
            font-size: .71rem;
            font-weight: 800;
        }

        .technical {
            padding: .8rem .9rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px solid var(--line);
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.55;
        }

        div[data-baseweb="tab-list"] {
            gap: 1.25rem;
        }

        button[data-baseweb="tab"] {
            padding-left: 0;
            padding-right: 0;
        }

        @media (max-width: 900px) {
            .page-head {
                align-items: flex-start;
                flex-direction: column;
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
if "custom_threshold" not in st.session_state:
    st.session_state.custom_threshold = float(bundle.unknown_threshold)
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
        return max(float(bundle.unknown_threshold), 0.75)
    if st.session_state.policy == "Custom":
        return float(st.session_state.custom_threshold)
    return float(bundle.unknown_threshold)


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
    """
    <div class="page-head">
        <div>
            <div class="page-kicker">Analyze</div>
            <h1 class="page-title">Flag recognition workspace</h1>
            <div class="page-subtitle">
                Upload an image, review the model decision, and inspect uncertainty before
                accepting a country result.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, config_col = st.columns([1.75, 0.85], gap="large")

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
            st.metric("Checkpoint threshold", f"{bundle.unknown_threshold:.1%}")
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
            f"Active threshold {effective_threshold:.1%} · checkpoint threshold {bundle.unknown_threshold:.1%}"
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
