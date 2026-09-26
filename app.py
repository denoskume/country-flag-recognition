"""Streamlit demo for Country Flag Recognition."""

from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from flag_recognition.inference import (
    load_inference_bundle,
    predict_image,
)


MODEL_PATH = Path(
    "artifacts/models/baseline_mobilenet_v3_small.pt"
)


def display_country(label: str) -> str:
    """Convert a canonical class slug into readable display text."""
    return label.replace("_", " ").replace("-", " ").title()


st.set_page_config(
    page_title="Country Flag Recognition",
    page_icon="🌍",
    layout="centered",
)

st.title("Country Flag Recognition")
st.caption("Seen, Unseen and Open-Set Evaluation")

st.write(
    "Upload a country-flag image. The supervised model returns its "
    "Top-5 candidates and reports whether the confidence is above the "
    "validation-derived known/unknown threshold."
)


@st.cache_resource
def get_model():
    return load_inference_bundle(
        MODEL_PATH,
        device="cpu",
    )


if not MODEL_PATH.is_file():
    st.info(
        "The demo interface is ready, but no trained baseline artifact is "
        "present yet. Train the model and place the checkpoint at "
        + str(MODEL_PATH)
        + "."
    )
    st.stop()


bundle = get_model()

uploaded_file = st.file_uploader(
    "Upload a flag image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Input image",
        use_container_width=True,
    )

    prediction = predict_image(
        image,
        bundle,
        top_k=5,
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.metric(
            "Top prediction",
            display_country(
                prediction.top1_country
            ),
        )

    with right:
        st.metric(
            "Confidence",
            f"{prediction.top1_confidence:.1%}",
        )

    if prediction.is_known:
        st.success(
            "Known-class decision: confidence is above the "
            "validation-derived acceptance threshold."
        )
    else:
        st.warning(
            "Possible unseen flag: confidence is below the "
            "validation-derived acceptance threshold."
        )

    st.caption(
        "Known / unknown threshold: "
        f"{prediction.unknown_threshold:.1%} · "
        f"Inference: {prediction.inference_ms:.1f} ms"
    )

    top5_table = pd.DataFrame(
        [
            {
                "Country": display_country(
                    country
                ),
                "Confidence": confidence,
            }
            for country, confidence
            in prediction.top5
        ]
    )

    st.subheader("Top-5 candidates")
    st.dataframe(
        top5_table.style.format(
            {"Confidence": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "A low supervised confidence does not identify the unseen country "
        "by itself. Zero-shot recognition is evaluated separately."
    )
