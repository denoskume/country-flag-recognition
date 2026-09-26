"""Streamlit dashboard for Country Flag Recognition."""

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


MODEL_PATH = Path(
    "artifacts/models/baseline_mobilenet_v3_small.pt"
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

    return f"{value:,}".replace(
        ",",
        " ",
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
    """Cache live country metadata briefly while keeping it reasonably fresh."""
    return fetch_country_profile(
        country_code
    )


st.set_page_config(
    page_title="Country Flag Recognition",
    page_icon="🌍",
    layout="wide",
)

st.title("Country Flag Recognition")
st.caption(
    "Computer vision recognition + live country intelligence"
)

st.write(
    "Upload a flag image. The model identifies the flag, estimates confidence, "
    "and—when the prediction is accepted as a known class—loads current country "
    "metadata from external reference sources."
)


if not MODEL_PATH.is_file():
    st.info(
        "The dashboard is implemented, but the trained baseline checkpoint is "
        "not present yet. Train the model and place the checkpoint at "
        + str(MODEL_PATH)
        + "."
    )
    st.stop()


bundle = get_model()

uploaded_file = st.file_uploader(
    "Upload a flag image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)

if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    prediction = predict_image(
        image,
        bundle,
        top_k=5,
    )

    st.divider()

    image_column, prediction_column = st.columns(
        [1.1, 1.4]
    )

    with image_column:
        st.subheader("Detected flag")
        st.image(
            image,
            caption="Uploaded flag",
            use_container_width=True,
        )

    with prediction_column:
        st.subheader("Recognition result")

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.metric(
                "Top prediction",
                display_country(
                    prediction.top1_country
                ),
            )

        with metric_2:
            st.metric(
                "Confidence",
                f"{prediction.top1_confidence:.1%}",
            )

        with metric_3:
            st.metric(
                "Inference",
                f"{prediction.inference_ms:.1f} ms",
            )

        if prediction.is_known:
            st.success(
                "Known flag — confidence is above the validation-derived "
                "acceptance threshold."
            )
        else:
            st.warning(
                "Possible unknown flag — the supervised prediction is below "
                "the acceptance threshold. Country metadata is not asserted "
                "from this low-confidence result."
            )

        st.caption(
            "Known / unknown threshold: "
            f"{prediction.unknown_threshold:.1%}"
        )

        top5_table = pd.DataFrame(
            [
                {
                    "Country code": country.upper(),
                    "Candidate": display_country(
                        country
                    ),
                    "Confidence": confidence,
                }
                for country, confidence
                in prediction.top5
            ]
        )

        with st.expander(
            "Top-5 model candidates"
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

    if prediction.is_known:
        st.divider()
        st.subheader("Country dashboard")

        try:
            with st.spinner(
                "Loading current country information..."
            ):
                profile = get_country_profile(
                    prediction.top1_country
                )

            country_name = profile.name

            st.markdown(
                f"## {country_name}"
            )

            row_1 = st.columns(4)

            with row_1[0]:
                st.metric(
                    "Continent",
                    profile.continent,
                )

            with row_1[1]:
                st.metric(
                    "Capital",
                    profile.capital,
                )

            with row_1[2]:
                st.metric(
                    "Currency",
                    profile.currency,
                )

            with row_1[3]:
                population_label = (
                    format_population(
                        profile.population.value
                    )
                )
                st.metric(
                    "Population",
                    population_label,
                )
                if profile.population.year:
                    st.caption(
                        "Latest available observation: "
                        + profile.population.year
                    )

            st.divider()

            political_column, geography_column = st.columns(
                [1.15, 1.0]
            )

            with political_column:
                st.subheader(
                    "Government & leadership"
                )

                st.markdown(
                    "**Government form**  \n"
                    + profile.government_form
                )

                st.markdown(
                    "**Head of State**  \n"
                    + profile.head_of_state
                )

                if (
                    profile.head_of_state_office
                    != "Not available"
                ):
                    st.caption(
                        profile.head_of_state_office
                    )

                st.markdown(
                    "**Head of Government**  \n"
                    + profile.head_of_government
                )

                if (
                    profile.head_of_government_office
                    != "Not available"
                ):
                    st.caption(
                        profile.head_of_government_office
                    )

            with geography_column:
                st.subheader(
                    "Geographic location"
                )

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

                    st.caption(
                        f"Approximate country coordinates: "
                        f"{profile.latitude:.3f}, "
                        f"{profile.longitude:.3f}"
                    )
                else:
                    st.info(
                        "Geographic coordinates are not available "
                        "for this entity."
                    )

            st.divider()

            st.caption(
                "Political/geographic metadata: Wikidata · "
                "Population: World Bank latest non-empty annual observation. "
                "Country information is fetched at runtime and cached for one hour."
            )

        except (
            requests.RequestException,
            LookupError,
            ValueError,
        ) as error:
            st.warning(
                "The flag was recognized, but live country metadata "
                "could not be retrieved right now."
            )
            st.caption(
                str(error)
            )

    else:
        st.info(
            "Zero-shot country identification will be used here in the next "
            "project stage so an unseen flag can be named before country "
            "metadata is loaded."
        )
