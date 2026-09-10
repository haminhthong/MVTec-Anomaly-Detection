"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import src.api.app as app_module
from src.config import TrainConfig
from src.training.trainer import train_patchcore


@pytest.fixture(scope="module")
def setup_api_model(tmp_path_factory: pytest.TempPathFactory):
    """Set up a test model for the API test suite."""
    tmp_path = tmp_path_factory.mktemp("api_test_env")
    raw_dir = tmp_path / "data" / "raw"
    category = "api_item"
    train_good = raw_dir / category / "train" / "good"
    test_good = raw_dir / category / "test" / "good"
    test_defect = raw_dir / category / "test" / "hole"
    gt_dir = raw_dir / category / "ground_truth" / "hole"

    for d in (train_good, test_good, test_defect, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(25):
        Image.new("RGB", (32, 32), color=(i * 10, i * 5, 200)).save(train_good / f"{i:03d}.png")
    for i in range(5):
        Image.new("RGB", (32, 32), color=(255, i, 0)).save(test_good / f"{i:03d}.png")
        Image.new("RGB", (32, 32), color=(0, 255, i)).save(test_defect / f"{i:03d}.png")
        Image.new("L", (32, 32), color=255).save(gt_dir / f"{i:03d}_mask.png")

    models_dir = tmp_path / "models"
    cfg = TrainConfig(
        category=category,
        batch_size=4,
        min_calibration_samples=5,
        coreset_size=20,
    )
    _ = train_patchcore(config=cfg, models_dir=models_dir, data_dir=raw_dir)

    original_model_dir = app_module.MODEL_DIR
    app_module.MODEL_DIR = models_dir

    yield category

    app_module.MODEL_DIR = original_model_dir


def test_health_endpoint(setup_api_model: str) -> None:
    """Health chỉ kiểm tra model category có sẵn."""
    client = TestClient(app_module.app)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is True
    assert setup_api_model in data["categories"]


def test_inspect_single_image(setup_api_model: str) -> None:
    """Test POST /inspect with single image upload."""
    client = TestClient(app_module.app)

    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        f"/inspect?category={setup_api_model}",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["category"] == setup_api_model
    assert data["decision"] in {"PASS_CANDIDATE", "REVIEW_REQUIRED", "RECAPTURE_REQUIRED"}
    assert "image_threshold" in data
    assert "anomalous_area_ratio" in data


def test_inspect_batch_images(setup_api_model: str) -> None:
    """Test POST /inspect/batch with multiple image files."""
    client = TestClient(app_module.app)

    files_payload = []
    for i in range(3):
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files_payload.append(("files", (f"img_{i}.png", buf.getvalue(), "image/png")))

    res = client.post(
        f"/inspect/batch?category={setup_api_model}",
        files=files_payload,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["batch_size"] == 3
    assert len(data["items"]) == 3
    assert data["category"] == setup_api_model


def test_inspect_missing_category_404() -> None:
    """Test POST /inspect with non-existent category returns 404."""
    client = TestClient(app_module.app)
    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    res = client.post(
        "/inspect?category=unknown_item_404",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 404
