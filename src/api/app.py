"""FastAPI transport cho PatchCore-style anomaly detection."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from ..inference.detector import AnomalyDetector
from ..model.artifacts import ModelArtifact
from ..path_safety import ensure_safe_segment
from .schemas import BatchInspectionResponse, HealthResponse, InspectionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 16
MAX_IMAGE_PIXELS = 25_000_000

app = FastAPI(
    title="MVTec PatchCore-style Anomaly Detection",
    description="Phát hiện bất thường từ ảnh normal, định vị vùng nghi vấn và chuyển kiểm tra thủ công.",
    version="1.0.0",
)


def _category_dir(category: str) -> Path:
    """Resolve category an toàn dưới thư mục models."""
    try:
        safe_category = ensure_safe_segment(category.strip(), "category")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MODEL_DIR / safe_category


def _available_categories() -> list[str]:
    """Liệt kê category có đủ metadata và memory bank."""
    if not MODEL_DIR.exists():
        return []
    return sorted(
        path.name
        for path in MODEL_DIR.iterdir()
        if path.is_dir()
        and not path.name.startswith((".", "_"))
        and (path / "metadata.json").exists()
        and (path / "memory_bank.npy").exists()
    )


def _validate_and_load_image(raw_bytes: bytes) -> Image.Image:
    """Kiểm tra file tải lên, chống ảnh giải nén quá lớn và chuyển sang RGB."""
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh tải lên vượt quá giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415,
            detail=f"File tải lên không phải định dạng ảnh hợp lệ: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Kiểm tra process và sự có mặt của model artifact."""
    categories = _available_categories()
    version = "not_trained"
    if categories:
        try:
            version = ModelArtifact.load(MODEL_DIR / categories[0]).metadata.model_version
        except (FileNotFoundError, ValueError):
            version = "invalid_artifact"
    return HealthResponse(
        status="ok" if categories else "degraded",
        model_ready=bool(categories),
        model_version=version,
        categories=categories,
    )


@app.post("/inspect", response_model=InspectionResponse, tags=["Inspection"])
async def inspect(
    file: Annotated[UploadFile, File(..., description="Ảnh sản phẩm (PNG/JPG)")],
    category: Annotated[str, Query(description="MVTec category, ví dụ bottle")],
    camera_id: Annotated[str | None, Query(description="Mã camera tùy chọn")] = None,
    include_overlay: Annotated[
        bool, Form(description="Trả heatmap overlay Base64 nếu true")
    ] = True,
) -> InspectionResponse:
    """Inspect một ảnh; category được truyền trực tiếp vào detector."""
    image = _validate_and_load_image(await file.read(MAX_UPLOAD_BYTES + 1))
    try:
        detector = AnomalyDetector(model_dir=_category_dir(category))
        result = detector.inspect(
            image,
            include_overlay=include_overlay,
            camera_id=camera_id,
        )
        return InspectionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Model không sẵn sàng: {exc}") from exc


@app.post("/inspect/batch", response_model=BatchInspectionResponse, tags=["Inspection"])
async def inspect_batch(
    files: Annotated[list[UploadFile], File(..., description="Nhiều ảnh sản phẩm")],
    category: Annotated[str, Query(description="MVTec category, ví dụ bottle")],
    camera_id: Annotated[str | None, Query(description="Mã camera tùy chọn")] = None,
    include_overlay: Annotated[
        bool, Form(description="Trả heatmap overlay Base64 nếu true")
    ] = False,
) -> BatchInspectionResponse:
    """Tính điểm theo batch và giữ nguyên thứ tự file đầu vào."""
    if not files:
        raise HTTPException(status_code=400, detail="Chưa cung cấp file cho batch inspection.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Số file trong batch vượt quá giới hạn {MAX_BATCH_FILES} file.",
        )
    images = [
        _validate_and_load_image(await file.read(MAX_UPLOAD_BYTES + 1))
        for file in files
    ]
    try:
        detector = AnomalyDetector(model_dir=_category_dir(category))
        results = detector.inspect_batch(
            images,
            include_overlay=include_overlay,
            camera_id=camera_id,
        )
        return BatchInspectionResponse(
            batch_size=len(results),
            category=detector.category,
            items=[InspectionResponse(**result) for result in results],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Model không sẵn sàng: {exc}") from exc
