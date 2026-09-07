"""SQLite repository tối giản cho inspection/review lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any


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
                    line_id TEXT,
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
        scores = result.get("scores", {})
        localization = result.get("localization", {})
        model = result.get("model", {})
        inspected_at = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO inspections(
                    inspection_id,line_id,camera_id,inspected_at,model_version,policy_version,
                    anomaly_score,anomaly_extent,decision,image_ref,overlay_ref,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result["inspection_id"],
                    result.get("line_id"),
                    result.get("camera_id"),
                    inspected_at,
                    model.get("version", result.get("model_version", "unknown")),
                    policy_version,
                    scores.get("anomaly_score"),
                    localization.get("anomalous_area_ratio"),
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
