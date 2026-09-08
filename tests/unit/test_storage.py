"""Kiểm thử lifecycle inspection và human QC trong SQLite."""

from __future__ import annotations

import sqlite3

import pytest

from src.storage.inspections import InspectionStore


def _inspection_result() -> dict[str, object]:
    return {
        "inspection_id": "insp_test_001",
        "decision": "HUMAN_REVIEW",
        "scores": {"anomaly_score": 2.5},
        "localization": {"anomalous_area_ratio": 0.1},
        "model": {"version": "1.0.0"},
    }


def test_inspection_review_lifecycle_enforces_contract(tmp_path) -> None:
    """Inspection phải tồn tại trước khi ghi QC và outcome phải thuộc V1."""
    store = InspectionStore(tmp_path / "inspections.sqlite3")
    store.record_inspection(_inspection_result())
    store.record_review("insp_test_001", "qa-01", "QC_REJECT", defect_type="scratch")

    result = store.get_inspection("insp_test_001")
    assert result is not None
    assert result["final_outcome"] == "QC_REJECT"
    assert result["payload"]["decision"] == "HUMAN_REVIEW"

    # Ghi lại cùng inspection_id không được làm mất review đã có.
    updated = _inspection_result()
    updated["scores"] = {"anomaly_score": 3.0}
    store.record_inspection(updated)
    result_after_upsert = store.get_inspection("insp_test_001")
    assert result_after_upsert is not None
    assert result_after_upsert["final_outcome"] == "QC_REJECT"
    assert result_after_upsert["payload"]["scores"]["anomaly_score"] == 3.0

    with pytest.raises(ValueError, match="final_outcome"):
        store.record_review("insp_test_001", "qa-01", "REJECT")

    with pytest.raises(sqlite3.IntegrityError):
        store.record_review("insp_missing", "qa-01", "QC_PASS")


def test_inspection_store_rejects_unknown_decision(tmp_path) -> None:
    """Không lưu decision ngoài contract V1."""
    store = InspectionStore(tmp_path / "inspections.sqlite3")
    result = _inspection_result()
    result["decision"] = "FAIL"
    with pytest.raises(ValueError, match="decision"):
        store.record_inspection(result)
