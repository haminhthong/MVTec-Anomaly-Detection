"""HTTP REST API Server for Industrial Visual Anomaly Detection (FastAPI Enterprise).

Adheres to strict architectural separation:
- API layer handles HTTP transport, validation, error mapping, and serialization ONLY.
- ALL ML logic, feature extraction, nearest neighbors, and thresholding are delegated to AnomalyDetector.
- Strict category lookup with ModelNotFoundError mapped directly to HTTP 404.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
import torch

from ..model.registry import ModelNotFoundError, ModelRegistry
from .schemas import (
    BatchInspectionResponse,
    HealthResponse,
    InspectionResponse,
    ReadinessResponse,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_DIR: Path = PROJECT_ROOT / "models"

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB per file
MAX_BATCH_FILES: int = 16
MAX_IMAGE_PIXELS: int = 25_000_000

app = FastAPI(
    title="Industrial Visual Anomaly Detection API",
    description="PatchCore-style visual anomaly detection service for manufacturing QC",
    version="2.0.0",
)

registry = ModelRegistry(base_dir=MODEL_DIR)


def _validate_and_load_image(raw_bytes: bytes) -> Image.Image:
    """Validate image bytes against decompression bombs and format errors."""
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image upload exceeds limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415,
            detail=f"Uploaded file is not a valid image format: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Basic health check endpoint."""
    categories = registry.list_categories()
    ready = len(categories) > 0
    version = registry.version(categories[0]) if categories else "not_trained"
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        model_version=version,
        categories=categories,
    )


@app.get("/health/live", tags=["Monitoring"])
def health_live() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessResponse, tags=["Monitoring"])
def health_ready() -> ReadinessResponse:
    """Readiness probe checking model availability and runtime index."""
    categories = registry.list_categories()
    if not categories:
        raise HTTPException(
            status_code=503,
            detail="No trained model artifacts available in system.",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        registry.get_detector(categories[0])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize model: {exc}",
        ) from exc

    return ReadinessResponse(
        ready=True,
        categories=categories,
        active_device=device,
    )


@app.get("/models", tags=["Model Registry"])
def list_models() -> dict[str, Any]:
    """List all categories with available trained models."""
    categories = registry.list_categories()
    return {
        "total_categories": len(categories),
        "categories": categories,
    }


@app.get("/models/{category}", tags=["Model Registry"])
def get_model_details(category: str) -> dict[str, Any]:
    """Get metadata for a specific category model."""
    try:
        return registry.get_metadata(category)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/inspect", response_model=InspectionResponse, tags=["Inspection"])
async def inspect(
    file: Annotated[UploadFile, File(..., description="Product image file (PNG/JPG)")],
    category: Annotated[
        str | None, Query(description="Product category (e.g. 'bottle'). If omitted, first available model is used")
    ] = None,
    include_overlay: Annotated[
        bool, Form(description="Whether to include Base64 heatmap overlay string")
    ] = True,
) -> InspectionResponse:
    """Inspect single product image and return operational QC decision."""
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _validate_and_load_image(content)

    target_category = category
    if not target_category:
        available = registry.list_categories()
        if not available:
            raise HTTPException(status_code=503, detail="No models loaded.")
        target_category = available[0]

    try:
        detector = registry.get_detector(target_category)
        result = detector.inspect(image, include_overlay=include_overlay)
        return InspectionResponse(**result)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/inspect/batch", response_model=BatchInspectionResponse, tags=["Inspection"])
async def inspect_batch(
    files: Annotated[list[UploadFile], File(..., description="Multiple product image files")],
    category: Annotated[
        str | None, Query(description="Product category. If omitted, first available is used")
    ] = None,
    include_overlay: Annotated[
        bool, Form(description="Whether to include Base64 heatmap overlay string")
    ] = False,
) -> BatchInspectionResponse:
    """High-throughput batch inspection endpoint for factory production lines."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch inspection.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum limit of {MAX_BATCH_FILES} files.",
        )

    images: list[Image.Image] = []
    for f in files:
        raw_bytes = await f.read(MAX_UPLOAD_BYTES + 1)
        images.append(_validate_and_load_image(raw_bytes))

    target_category = category
    if not target_category:
        available = registry.list_categories()
        if not available:
            raise HTTPException(status_code=503, detail="No models loaded.")
        target_category = available[0]

    try:
        detector = registry.get_detector(target_category)
        results = detector.inspect_batch(images, include_overlay=include_overlay)
        items = [InspectionResponse(**r) for r in results]
        return BatchInspectionResponse(
            batch_size=len(items),
            category=target_category,
            items=items,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
