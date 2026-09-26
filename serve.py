"""FastAPI product server for Flag Intelligence."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import yaml

from flag_recognition.country_info import fetch_country_profile
from flag_recognition.inference import load_inference_bundle, predict_image
from flag_recognition.taxonomy import country_name_from_code


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
MODEL_PATH = BASE_DIR / "artifacts/models/worldwide_mobilenet_v3_small.pt"
DEPLOYMENT_CONFIG_PATH = BASE_DIR / "configs/deployment.yaml"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Flag Intelligence",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

bundle = load_inference_bundle(MODEL_PATH, device="cpu")
deployment_config = yaml.safe_load(
    DEPLOYMENT_CONFIG_PATH.read_text(encoding="utf-8")
)
DEPLOYMENT_THRESHOLD = float(
    deployment_config["open_set"]["deployment_threshold"]
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "mobilenet_v3_small",
        "classes": len(bundle.index_to_class),
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
        "checkpoint_threshold": bundle.unknown_threshold,
        "device": str(bundle.device),
    }


@app.post("/api/analyze")
async def analyze(
    file: Annotated[UploadFile, File(...)],
    threshold: float | None = Query(default=None, ge=0.05, le=0.99),
    top_k: int = Query(default=5, ge=3, le=10),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Use JPG, PNG or WebP.",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    prediction = predict_image(image, bundle, top_k=top_k)
    active_threshold = (
        float(threshold)
        if threshold is not None
        else DEPLOYMENT_THRESHOLD
    )
    accepted = prediction.top1_confidence >= active_threshold

    return {
        "decision": (
            country_name_from_code(prediction.top1_country)
            if accepted
            else "Unknown"
        ),
        "accepted": accepted,
        "top_candidate": {
            "code": prediction.top1_country,
            "country": country_name_from_code(prediction.top1_country),
            "confidence": prediction.top1_confidence,
        },
        "confidence": prediction.top1_confidence,
        "latency_ms": prediction.inference_ms,
        "active_threshold": active_threshold,
        "deployment_threshold": DEPLOYMENT_THRESHOLD,
        "checkpoint_threshold": bundle.unknown_threshold,
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "candidates": [
            {
                "rank": rank,
                "code": code,
                "country": country_name_from_code(code),
                "confidence": confidence,
            }
            for rank, (code, confidence)
            in enumerate(prediction.top5, start=1)
        ],
    }


@app.get("/api/country/{code}")
def country(code: str):
    try:
        profile = fetch_country_profile(code)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Country data is temporarily unavailable.",
        ) from exc

    data = asdict(profile)
    return data
