"""Professional Streamlit dashboard for Country Flag Recognition."""

from pathlib import Path

import pandas as pd
from PIL import Image
import requests
import streamlit as st

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.inference import (
    load_inference_bundle,
    predict_image,
)
from flag_recognition.taxonomy import country_name_from_code


MODEL_PATH = Path(
    "artifacts/models/worldwide_mobilenet_v3_small.pt"
)


def display_country(label: str) -> str:
    """Convert a canonical class slug into readable fallback text."""
    return (
        label.replace("_", " ")
        .replace("-", " ")
        .title()
    )


def format_population(
    value: int | None,
) -> str:
    if value is None:
        return "Not available"

    return f"{value:,}".replace(",", " ")


def format_area(
    value: float | None,
) -> str:
    if value is None:
        return "Not available"

    return (
        f"{value:,.0f} km²"
        .replace(",", " ")
    )


@st.cache_resource
def get_model():
    return load_inference_bundle(
        MODEL_PATH,
        device="cpu",
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def get_country_profile(
    country_code: str,
):
    """Cache live metadata for one hour."""
    return fetch_country_profile(
        country_code
    )


st.set_page_config(
    page_title="Country Flag Recognition",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(47, 109, 246, 0.08), transparent 28rem),
                radial-gradient(circle at 92% 8%, rgba(0, 180, 160, 0.06), transparent 24rem),
                #f7f9fc;
        }

        .block-container {
            max-width: 1360px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero {
            padding: 0.15rem 0 0.85rem 0;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.34rem 0.62rem;
            border: 1px solid rgba(47, 109, 246, 0.16);
            border-radius: 999px;
            background: rgba(47, 109, 246, 0.07);
            color: #315fbe;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: clamp(2.15rem, 3.4vw, 3.55rem);
            font-weight: 800;
            line-height: 1.02;
            margin: 0;
            color: #111827;
        }

        .hero-subtitle {
            max-width: 900px;
            color: #5f6b7a;
            font-size: 1rem;
            line-height: 1.55;
            margin-top: 0.55rem;
        }

        div[data-testid="stFileUploader"] {
            background: transparent;
            border: 0;
            padding: 0;
            box-shadow: none;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #cbd5e1;
            border-radius: 14px;
            background: #fbfcfe;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #e5e9f0;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
            min-height: 104px;
        }

        div[data-testid="stMetricLabel"] {
            color: #697586;
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.045em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: #101828;
            font-weight: 750;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.94);
            border-radius: 18px;
            border-color: #e5e9f0 !important;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        .section-kicker {
            color: #315fbe;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }

        .section-title {
            color: #111827;
            font-size: 1.55rem;
            font-weight: 750;
            margin-bottom: 0.85rem;
        }

        .country-title {
            font-size: 2rem;
            font-weight: 800;
            color: #111827;
            margin: 0;
        }

        .country-code {
            display: inline-block;
            margin-left: 0.5rem;
            padding: 0.18rem 0.48rem;
            border-radius: 8px;
            background: #eef2f7;
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            vertical-align: middle;
        }

        .muted {
            color: #7a8697;
            font-size: 0.88rem;
        }

        .confidence-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: #e9edf3;
            overflow: hidden;
            margin: 0.4rem 0 0.8rem 0;
        }

        .decision-banner {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.8rem 0 0.7rem 0;
            font-weight: 650;
        }

        .decision-known {
            background: #ecfdf3;
            border: 1px solid #ccebd8;
            color: #17663a;
        }

        .decision-unknown {
            background: #fff8e6;
            border: 1px solid #f2dfaa;
            color: #8a5a00;
        }

        .scope-card {
            padding: 0.95rem 1rem;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            background: #fbfcfe;
        }

        .scope-title {
            font-weight: 750;
            color: #111827;
            margin-bottom: 0.45rem;
        }

        .scope-copy {
            color: #697586;
            line-height: 1.5;
            font-size: 0.9rem;
        }

        .scope-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .scope-badge {
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: #eef4ff;
            border: 1px solid #d9e6ff;
            color: #315fbe;
            font-size: 0.74rem;
            font-weight: 700;
        }

        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.55rem;
        }

        .workflow-card {
            min-height: 118px;
            padding: 0.95rem 1rem;
            border: 1px solid #e5e9f0;
            border-radius: 14px;
            background: #fbfcfe;
        }

        .workflow-number {
            color: #315fbe;
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .workflow-title {
            color: #111827;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .workflow-copy {
            color: #697586;
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .source-note {
            color: #7a8697;
            font-size: 0.78rem;
            line-height: 1.5;
        }

        hr {
            border-color: #e7ebf0 !important;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1.2rem;
            }

            .hero-title {
                font-size: 2.25rem;
            }

            .workflow-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Computer Vision · Live Country Intelligence</div>
        <div class="hero-title">Country Flag Recognition</div>
        <div class="hero-subtitle">
            Upload a flag image. The vision model identifies the flag,
            measures confidence and, for an accepted prediction, turns the
            result into a live country dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    upload_left, upload_right = st.columns(
        [1.6, 1.0],
        gap="large",
    )

    with upload_left:
        st.markdown(
            '<div class="section-kicker">Input</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Analyze a flag</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            label_visibility="collapsed",
        )

    with upload_right:
        st.markdown(
            """
            <div class="scope-card">
                <div class="scope-title">Recognition scope</div>
                <div class="scope-copy">
                    Worldwide flag recognition with confidence-aware rejection
                    for uncertain inputs and varied real-world presentations.
                </div>
                <div class="scope-badges">
                    <span class="scope-badge">250 classes</span>
                    <span class="scope-badge">Open-set gate</span>
                    <span class="scope-badge">CPU ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if not MODEL_PATH.is_file():
    st.info(
        "The dashboard UI is ready. The trained checkpoint is still required at "
        + str(MODEL_PATH)
        + " before live recognition can run."
    )
    st.stop()


bundle = get_model()


if uploaded_file is None:
    st.markdown("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-kicker">Workflow</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">From image to country intelligence</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="workflow-grid">
                <div class="workflow-card">
                    <div class="workflow-number">01</div>
                    <div class="workflow-title">Upload</div>
                    <div class="workflow-copy">Provide a JPG, PNG or WebP flag image.</div>
                </div>
                <div class="workflow-card">
                    <div class="workflow-number">02</div>
                    <div class="workflow-title">Recognize</div>
                    <div class="workflow-copy">MobileNetV3 ranks worldwide candidates.</div>
                </div>
                <div class="workflow-card">
                    <div class="workflow-number">03</div>
                    <div class="workflow-title">Validate</div>
                    <div class="workflow-copy">Confidence gating accepts or rejects the result.</div>
                </div>
                <div class="workflow-card">
                    <div class="workflow-number">04</div>
                    <div class="workflow-title">Enrich</div>
                    <div class="workflow-copy">Accepted results unlock live country intelligence.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


image = Image.open(
    uploaded_file
).convert("RGB")

prediction = predict_image(
    image,
    bundle,
    top_k=5,
)

st.markdown("")
st.markdown(
    '<div class="section-kicker">Model output</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-title">Recognition result</div>',
    unsafe_allow_html=True,
)

flag_column, result_column = st.columns(
    [1.0, 1.65],
    gap="large",
)

with flag_column:
    with st.container(border=True):
        st.markdown("##### Uploaded flag")
        st.image(
            image,
            use_container_width=True,
        )
        st.caption(
            f"{image.width} × {image.height} px"
        )

with result_column:
    with st.container(border=True):
        decision_label = (
            country_name_from_code(prediction.top1_country)
            if prediction.is_known
            else "Unknown"
        )
        top_candidate = country_name_from_code(
            prediction.top1_country
        )

        result_1, result_2, result_3, result_4 = st.columns(
            4,
            gap="medium",
        )

        with result_1:
            st.metric(
                "Decision",
                decision_label,
            )

        with result_2:
            st.metric(
                "Top candidate",
                top_candidate,
            )

        with result_3:
            st.metric(
                "Confidence",
                f"{prediction.top1_confidence:.1%}",
            )

        with result_4:
            st.metric(
                "Latency",
                f"{prediction.inference_ms:.1f} ms",
            )

        confidence_percent = max(
            0.0,
            min(
                100.0,
                prediction.top1_confidence * 100.0,
            ),
        )

        st.progress(
            int(round(confidence_percent))
        )

        if prediction.is_known:
            st.markdown(
                (
                    '<div class="decision-banner decision-known">'
                    'Accepted · confidence is above the model threshold.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                (
                    '<div class="decision-banner decision-unknown">'
                    'Not accepted · top candidate shown for inspection only.'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        st.caption(
            "Acceptance threshold "
            f"{prediction.unknown_threshold:.1%}"
        )

        top5_table = pd.DataFrame(
            [
                {
                    "Rank": rank,
                    "Code": country.upper(),
                    "Candidate": country_name_from_code(
                        country
                    ),
                    "Confidence": confidence,
                }
                for rank, (
                    country,
                    confidence,
                ) in enumerate(
                    prediction.top5,
                    start=1,
                )
            ]
        )

        with st.expander(
            "View Top-5 candidates"
        ):
            st.dataframe(
                top5_table.style.format(
                    {
                        "Confidence": "{:.1%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


if not prediction.is_known:
    st.info(
        "No country profile is shown because this prediction was not accepted."
    )
    st.stop()


try:
    with st.spinner(
        "Loading current country information..."
    ):
        profile = get_country_profile(
            prediction.top1_country
        )

    st.markdown("")
    st.markdown(
        '<div class="section-kicker">Country intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="country-title">'
            + profile.name
            + '<span class="country-code">'
            + prediction.top1_country.upper()
            + "</span></div>"
        ),
        unsafe_allow_html=True,
    )

    if profile.population.year:
        st.markdown(
            (
                '<div class="muted">'
                "Population · latest available World Bank observation · "
                + profile.population.year
                + "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    overview_tab, government_tab, geography_tab = st.tabs(
        [
            "Overview",
            "Government",
            "Geography",
        ]
    )

    with overview_tab:
        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(
            4,
            gap="medium",
        )

        with kpi_1:
            st.metric("Capital", profile.capital)

        with kpi_2:
            st.metric(
                "Population",
                format_population(
                    profile.population.value
                ),
            )

        with kpi_3:
            st.metric("Currency", profile.currency)

        with kpi_4:
            st.metric(
                "Area",
                format_area(
                    profile.area_km2
                ),
            )

        detail_left, detail_right = st.columns(
            2,
            gap="medium",
        )

        with detail_left:
            with st.container(border=True):
                st.markdown("##### Official language(s)")
                st.write(profile.official_languages)

        with detail_right:
            with st.container(border=True):
                st.markdown("##### Country profile")
                st.write(profile.overview)

    with government_tab:
        government_left, government_right = st.columns(
            2,
            gap="medium",
        )

        with government_left:
            with st.container(border=True):
                st.markdown("##### Government form")
                st.write(profile.government_form)

                st.divider()

                st.markdown("##### Head of State")
                st.write(profile.head_of_state)

                if (
                    profile.head_of_state_office
                    != "Not available"
                ):
                    st.caption(
                        profile.head_of_state_office
                    )

        with government_right:
            with st.container(border=True):
                st.markdown("##### Head of Government")
                st.write(profile.head_of_government)

                if (
                    profile.head_of_government_office
                    != "Not available"
                ):
                    st.caption(
                        profile.head_of_government_office
                    )

                st.divider()

                st.markdown("##### Practical")
                st.write(
                    f"Calling code: {profile.calling_code}"
                )
                st.write(
                    f"Internet domain: {profile.internet_domain}"
                )
                st.write(
                    f"Driving side: {profile.driving_side}"
                )

    with geography_tab:
        geo_left, geo_right = st.columns(
            [1.4, 1.0],
            gap="large",
        )

        with geo_left:
            with st.container(border=True):
                if (
                    profile.latitude is not None
                    and profile.longitude is not None
                ):
                    map_data = pd.DataFrame(
                        [
                            {
                                "lat": profile.latitude,
                                "lon": profile.longitude,
                            }
                        ]
                    )

                    st.map(
                        map_data,
                        latitude="lat",
                        longitude="lon",
                        zoom=3,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Geographic coordinates are not available."
                    )

        with geo_right:
            metric_a, metric_b = st.columns(2)

            with metric_a:
                st.metric(
                    "Continent",
                    profile.continent,
                )

            with metric_b:
                st.metric(
                    "Area",
                    format_area(
                        profile.area_km2
                    ),
                )

            if (
                profile.latitude is not None
                and profile.longitude is not None
            ):
                st.caption(
                    "Reference coordinates · "
                    f"{profile.latitude:.3f}, "
                    f"{profile.longitude:.3f}"
                )

    st.markdown("")
    st.markdown(
        (
            '<div class="source-note">'
            "Live metadata: Wikidata · Population: World Bank · "
            "Overview: Wikipedia · cached for one hour."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

except (
    requests.RequestException,
    LookupError,
    ValueError,
) as error:
    st.warning(
        "The flag was recognized, but live country metadata could not be "
        "retrieved right now."
    )
    st.caption(
        str(error)
    )
