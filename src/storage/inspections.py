"""SQLite repository tối giản cho inspection/review lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any


ALLOWED_DECISIONS = {"RECAPTURE_REQUIRED", "PASS_CANDIDATE", "REVIEW_REQUIRED"}
ALLOWED_QC_OUTCOMES = {"QC_PASS", "QC_REJECT"}


class InspectionStore:
    """Lưu evidence model và outcome QC trong một database local."""

    def __init__(self, database_path: str | Path = "data/inspections.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        """Mở và đóng connection rõ ràng để Windows không khóa file database."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Tạo schema một lần; không có trigger update memory bank."""
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    camera_id TEXT,
                    inspected_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    anomaly_score REAL,
                    anomaly_extent REAL,
                    decision TEXT NOT NULL,
                    image_ref TEXT,
                    overlay_ref TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    inspection_id TEXT PRIMARY KEY REFERENCES inspections(inspection_id),
                    reviewer_id TEXT NOT NULL,
                    final_outcome TEXT NOT NULL,
                    defect_type TEXT,
                    notes TEXT,
                    reviewed_at TEXT NOT NULL
                );
                """
            )

    def record_inspection(self, result: dict[str, Any], policy_version: str = "v1") -> None:
        """Ghi prediction; không bao giờ append prediction vào memory bank."""
        if result.get("decision") not in ALLOWED_DECISIONS:
            raise ValueError(f"decision không hợp lệ: {result.get('decision')!r}.")
        anomaly_score = result.get("anomaly_score")
        anomaly_extent = result.get("anomalous_area_ratio", 0.0)
        inspected_at = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO inspections(
                    inspection_id,camera_id,inspected_at,model_version,policy_version,
                    anomaly_score,anomaly_extent,decision,image_ref,overlay_ref,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(inspection_id) DO UPDATE SET
                    camera_id=excluded.camera_id,
                    inspected_at=excluded.inspected_at,
                    model_version=excluded.model_version,
                    policy_version=excluded.policy_version,
                    anomaly_score=excluded.anomaly_score,
                    anomaly_extent=excluded.anomaly_extent,
                    decision=excluded.decision,
                    image_ref=excluded.image_ref,
                    overlay_ref=excluded.overlay_ref,
                    payload_json=excluded.payload_json
                """,
                (
                    result["inspection_id"],
                    result.get("camera_id"),
                    inspected_at,
                    result.get("model_version", "unknown"),
                    policy_version,
                    anomaly_score,
                    anomaly_extent,
                    result["decision"],
                    result.get("image_ref"),
                    result.get("overlay_ref"),
                    json.dumps(result, ensure_ascii=False),
                ),
            )

    def record_review(
        self,
        inspection_id: str,
        reviewer_id: str,
        final_outcome: str,
        defect_type: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Ghi QC_PASS/QC_REJECT; human review là nguồn label độc lập."""
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id không được để trống.")
        if final_outcome not in ALLOWED_QC_OUTCOMES:
            raise ValueError(f"final_outcome không hợp lệ: {final_outcome!r}.")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO reviews(
                    inspection_id,reviewer_id,final_outcome,defect_type,notes,reviewed_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    inspection_id,
                    reviewer_id,
                    final_outcome,
                    defect_type,
                    notes,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_inspection(self, inspection_id: str) -> dict[str, Any] | None:
        """Đọc evidence inspection kèm kết quả QC nếu đã review."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT i.*, r.reviewer_id, r.final_outcome, r.defect_type, r.notes, r.reviewed_at
                FROM inspections AS i
                LEFT JOIN reviews AS r ON r.inspection_id = i.inspection_id
                WHERE i.inspection_id = ?
                """,
                (inspection_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result
