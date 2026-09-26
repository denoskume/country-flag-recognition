"""Interactive Streamlit product UI for worldwide country-flag recognition."""

from __future__ import annotations

from pathlib import Path

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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-soft: #f8fafc;
            --text: #0f172a;
            --muted: #667085;
            --line: #e6eaf0;
            --blue: #2457e6;
            --blue-soft: #eef3ff;
            --green: #137a4b;
            --green-soft: #ecfdf3;
            --amber: #9a6700;
            --amber-soft: #fff8e5;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% -10%, rgba(36,87,230,.08), transparent 30rem),
                radial-gradient(circle at -5% 18%, rgba(2,132,199,.05), transparent 26rem),
                var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: rgba(245,247,251,.88);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(230,234,240,.9);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 0;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] .stCaption p {
            color: #cbd5e1 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] div[data-testid="stToggle"] {
            color: #f8fafc;
        }

        .brandbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .brand-left {
            display: flex;
            align-items: center;
            gap: .72rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 13px;
            display: grid;
            place-items: center;
            font-weight: 900;
            color: white;
            background: linear-gradient(135deg, #2457e6, #10a6c7);
            box-shadow: 0 10px 30px rgba(36,87,230,.22);
        }

        .brand-name {
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1.1;
        }

        .brand-sub {
            font-size: .77rem;
            color: var(--muted);
            margin-top: .16rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .42rem .72rem;
            border: 1px solid #dfe5ee;
            border-radius: 999px;
            background: rgba(255,255,255,.82);
            color: #475467;
            font-size: .77rem;
            font-weight: 700;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34,197,94,.10);
        }

        .hero-shell {
            border: 1px solid var(--line);
            border-radius: 24px;
            background:
                linear-gradient(120deg, rgba(36,87,230,.06), rgba(16,166,199,.03)),
                #fff;
            padding: 1.55rem 1.7rem;
            box-shadow: 0 18px 50px rgba(15,23,42,.05);
            margin-bottom: 1rem;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            color: var(--blue);
            font-size: .74rem;
            font-weight: 800;
            letter-spacing: .09em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }

        .hero-title {
            color: var(--text);
            font-weight: 850;
            letter-spacing: -.035em;
            font-size: clamp(2.3rem, 4vw, 4.5rem);
            line-height: .98;
            margin: 0;
        }

        .hero-copy {
            max-width: 830px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin-top: .75rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .65rem;
            margin-top: 1.1rem;
        }

        .mini-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(255,255,255,.82);
            padding: .8rem .9rem;
        }

        .mini-label {
            color: #98a2b3;
            text-transform: uppercase;
            letter-spacing: .07em;
            font-size: .68rem;
            font-weight: 800;
        }

        .mini-value {
            color: var(--text);
            font-weight: 800;
            margin-top: .2rem;
            font-size: .94rem;
        }

        .section-label {
            color: var(--blue);
            font-size: .73rem;
            font-weight: 800;
            letter-spacing: .09em;
            text-transform: uppercase;
            margin-bottom: .18rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -.02em;
            margin-bottom: .75rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
            background: rgba(255,255,255,.95);
            box-shadow: 0 10px 32px rgba(15,23,42,.035);
        }

        div[data-testid="stFileUploader"] {
            border: 0;
            padding: 0;
            background: transparent;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed #b9c4d4;
            border-radius: 16px;
            background: #f9fbfd;
            min-height: 120px;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 16px;
            padding: .95rem 1rem;
            min-height: 106px;
        }

        div[data-testid="stMetricLabel"] {
            color: #667085;
            font-size: .72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 850;
        }

        .decision {
            border-radius: 14px;
            padding: .9rem 1rem;
            font-weight: 750;
            margin-top: .7rem;
        }

        .decision-known {
            color: var(--green);
            background: var(--green-soft);
            border: 1px solid #cdebd9;
        }

        .decision-unknown {
            color: var(--amber);
            background: var(--amber-soft);
            border: 1px solid #f0dfa5;
        }

        .muted {
            color: var(--muted);
            font-size: .84rem;
        }

        .country-head {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin-bottom: .5rem;
        }

        .country-name {
            color: var(--text);
            font-size: 1.8rem;
            font-weight: 850;
            letter-spacing: -.025em;
        }

        .country-code {
            padding: .2rem .48rem;
            border-radius: 8px;
            background: var(--blue-soft);
            color: var(--blue);
            font-size: .73rem;
            font-weight: 800;
        }

        .tech-note {
            padding: .8rem .9rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px solid var(--line);
            color: #667085;
            font-size: .82rem;
            line-height: 1.5;
        }

        @media (max-width: 900px) {
            .mini-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2.35rem;
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

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Tune the interface and decision policy without retraining the model.")

    policy = st.selectbox(
        "Decision policy",
        ["Calibrated", "Strict", "Custom"],
        index=0,
        help=(
            "Calibrated uses the checkpoint threshold. Strict raises the gate. "
            "Custom lets you inspect another operating point."
        ),
    )

    if policy == "Calibrated":
        effective_threshold = float(bundle.unknown_threshold)
        st.caption(f"Threshold: {effective_threshold:.1%}")
    elif policy == "Strict":
        effective_threshold = max(float(bundle.unknown_threshold), 0.75)
        st.caption(f"Threshold: {effective_threshold:.1%}")
    else:
        effective_threshold = st.slider(
            "Acceptance threshold",
            min_value=0.05,
            max_value=0.99,
            value=float(bundle.unknown_threshold),
            step=0.01,
            format="%.2f",
        )

    top_k = st.slider(
        "Candidates shown",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    live_enrichment = st.toggle(
        "Live country enrichment",
        value=True,
        help="Fetch geographic, demographic and government information for accepted predictions.",
    )

    show_candidates = st.toggle(
        "Show candidate ranking",
        value=True,
    )

    show_technical = st.toggle(
        "Show technical details",
        value=False,
    )

    st.divider()
    st.caption("Model")
    st.write("MobileNetV3-Small")
    st.caption("Scope")
    st.write("250 worldwide classes")
    st.caption("Runtime")
    st.write("CPU inference")


st.markdown(
    """
    <div class="brandbar">
        <div class="brand-left">
            <div class="brand-mark">FI</div>
            <div>
                <div class="brand-name">Flag Intelligence</div>
                <div class="brand-sub">Worldwide visual recognition</div>
            </div>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            Model ready
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="eyebrow">Vision system · confidence-aware recognition</div>
        <h1 class="hero-title">Recognize a flag.<br>Inspect the decision.</h1>
        <div class="hero-copy">
            Upload a flag image and inspect the model decision, candidate ranking,
            confidence and latency. Low-confidence results are rejected instead of
            being presented as confirmed countries.
        </div>
        <div class="mini-grid">
            <div class="mini-card">
                <div class="mini-label">Decision policy</div>
                <div class="mini-value">{policy}</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Acceptance threshold</div>
                <div class="mini-value">{effective_threshold:.1%}</div>
            </div>
            <div class="mini-card">
                <div class="mini-label">Candidate depth</div>
                <div class="mini-value">Top {top_k}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, info_col = st.columns([1.65, 1.0], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyze an image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload flag image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

with info_col:
    with st.container(border=True):
        st.markdown('<div class="section-label">Current configuration</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recognition settings</div>', unsafe_allow_html=True)
        st.write(f"**Policy:** {policy}")
        st.write(f"**Threshold:** {effective_threshold:.1%}")
        st.write(f"**Candidates:** Top {top_k}")
        st.write(f"**Live enrichment:** {'On' if live_enrichment else 'Off'}")
        st.caption("All controls are reversible and affect presentation or decision gating only.")

if uploaded_file is None:
    st.markdown("")
    with st.container(border=True):
        st.markdown('<div class="section-label">Workflow</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What happens after upload</div>', unsafe_allow_html=True)
        a, b, c, d = st.columns(4, gap="medium")
        with a:
            st.markdown("**01 · Prepare**")
            st.caption("The image is normalized to the deployment input format.")
        with b:
            st.markdown("**02 · Rank**")
            st.caption("The classifier scores all worldwide flag classes.")
        with c:
            st.markdown("**03 · Gate**")
            st.caption("The selected confidence policy accepts or rejects the result.")
        with d:
            st.markdown("**04 · Enrich**")
            st.caption("Accepted results can load live country intelligence.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
prediction = predict_image(image, bundle, top_k=top_k)

decision_is_known = prediction.top1_confidence >= effective_threshold
decision_label = (
    country_name_from_code(prediction.top1_country)
    if decision_is_known
    else "Unknown"
)
top_candidate = country_name_from_code(prediction.top1_country)

st.markdown("")
st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Recognition result</div>', unsafe_allow_html=True)

preview_col, result_col = st.columns([0.9, 1.65], gap="large")

with preview_col:
    with st.container(border=True):
        st.image(image, use_container_width=True)
        st.caption(f"{image.width} × {image.height} px")

with result_col:
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
                '<div class="decision decision-known">Accepted · confidence meets the active decision policy.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="decision decision-unknown">Rejected · the candidate is shown for inspection only.</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            f"Active threshold {effective_threshold:.1%} · "
            f"checkpoint threshold {bundle.unknown_threshold:.1%}"
        )

        if show_candidates:
            ranking = pd.DataFrame(
                [
                    {
                        "Rank": rank,
                        "Code": country.upper(),
                        "Candidate": country_name_from_code(country),
                        "Confidence": confidence,
                    }
                    for rank, (country, confidence) in enumerate(prediction.top5, start=1)
                ]
            )

            with st.expander("Candidate ranking", expanded=True):
                st.dataframe(
                    ranking.style.format({"Confidence": "{:.1%}"}),
                    hide_index=True,
                    use_container_width=True,
                )

        if show_technical:
            st.markdown(
                f"""
                <div class="tech-note">
                    Checkpoint: {MODEL_PATH}<br>
                    Model threshold: {bundle.unknown_threshold:.4f}<br>
                    Active threshold: {effective_threshold:.4f}<br>
                    Policy: {policy}<br>
                    Device: {bundle.device}
                </div>
                """,
                unsafe_allow_html=True,
            )

if not decision_is_known:
    st.stop()

if not live_enrichment:
    st.info("Live country enrichment is disabled in Control Center.")
    st.stop()

try:
    with st.spinner("Loading country intelligence..."):
        profile = get_country_profile(prediction.top1_country)

    st.markdown("")
    st.markdown('<div class="section-label">Country intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        (
            '<div class="country-head">'
            f'<div class="country-name">{profile.name}</div>'
            f'<div class="country-code">{prediction.top1_country.upper()}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if profile.population.year:
        st.markdown(
            f'<div class="muted">Latest available population observation · {profile.population.year}</div>',
            unsafe_allow_html=True,
        )

    overview_tab, government_tab, geography_tab = st.tabs(
        ["Overview", "Government", "Geography"]
    )

    with overview_tab:
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            st.metric("Capital", profile.capital)
        with k2:
            st.metric("Population", format_population(profile.population.value))
        with k3:
            st.metric("Currency", profile.currency)
        with k4:
            st.metric("Area", format_area(profile.area_km2))

        left, right = st.columns([1.0, 1.3], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("##### Official language(s)")
                st.write(profile.official_languages)
        with right:
            with st.container(border=True):
                st.markdown("##### Country profile")
                st.write(profile.overview)

    with government_tab:
        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown("##### Government form")
                st.write(profile.government_form)
                st.divider()
                st.markdown("##### Head of State")
                st.write(profile.head_of_state)
                if profile.head_of_state_office != "Not available":
                    st.caption(profile.head_of_state_office)

        with right:
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

    with geography_tab:
        map_col, detail_col = st.columns([1.45, 1.0], gap="large")

        with map_col:
            with st.container(border=True):
                if profile.latitude is not None and profile.longitude is not None:
                    map_data = pd.DataFrame(
                        [{"lat": profile.latitude, "lon": profile.longitude}]
                    )
                    st.map(
                        map_data,
                        latitude="lat",
                        longitude="lon",
                        zoom=3,
                        use_container_width=True,
                    )
                else:
                    st.info("Geographic coordinates are not available.")

        with detail_col:
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
    st.warning("The flag was recognized, but live country metadata could not be retrieved.")
    st.caption(str(error))
