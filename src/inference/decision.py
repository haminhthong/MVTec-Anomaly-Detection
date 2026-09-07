"""Decision semantics an toàn cho inspection V1.

Model chỉ được phép AUTO_PASS ảnh có rủi ro normal thấp. Mọi ảnh còn lại đi
HUMAN_REVIEW; model không tự suy ra PASS/REJECT cuối cùng hay mức độ major/minor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.artifacts import ThresholdPolicy


@dataclass(frozen=True)
class OperationalPolicy:
    """Policy frozen của detector: một ngưỡng auto-pass và một pixel threshold."""

    auto_pass_threshold: float
    pixel_threshold: float

    @classmethod
    def from_threshold_policy(cls, policy: ThresholdPolicy) -> "OperationalPolicy":
        """Tạo policy runtime từ artifact đã calibration."""
        return cls(
            auto_pass_threshold=policy.auto_pass_threshold,
            pixel_threshold=policy.pixel_threshold,
        )


def classify_decision(
    anomaly_score: float,
    threshold_policy: ThresholdPolicy | None = None,
    operational_policy: OperationalPolicy | None = None,
    auto_pass_threshold: float | None = None,
) -> str:
    """Trả về đúng một trong hai trạng thái AUTO_PASS hoặc HUMAN_REVIEW."""
    if operational_policy is not None:
        threshold = operational_policy.auto_pass_threshold
    elif threshold_policy is not None:
        threshold = threshold_policy.auto_pass_threshold
    elif auto_pass_threshold is not None:
        threshold = float(auto_pass_threshold)
    else:
        raise ValueError("Cần cung cấp threshold policy hoặc auto_pass_threshold.")
    return "AUTO_PASS" if float(anomaly_score) < threshold else "HUMAN_REVIEW"


def classify_decision_and_severity(
    anomaly_score: float,
    threshold_policy: ThresholdPolicy | None = None,
    operational_policy: OperationalPolicy | None = None,
    review_threshold: float | None = None,
    fail_threshold: float | None = None,
    anomalous_area_ratio: float = 0.0,
    peak_score: float = 0.0,
) -> tuple[str, str | None]:
    """API tương thích: không còn trả về business severity.

    Giá trị thứ hai chỉ là nhãn mô tả extent để client cũ không vỡ schema;
    ``ANOMALY_LOCALIZED``/``ANOMALY_EXTENSIVE`` không phải quyết định reject.
    """
    if operational_policy is None and threshold_policy is None:
        selected = fail_threshold if fail_threshold is not None else review_threshold
        if selected is None:
            raise ValueError("Thiếu threshold cho decision.")
        operational_policy = OperationalPolicy(float(selected), float(selected))
    decision = classify_decision(
        anomaly_score,
        threshold_policy=threshold_policy,
        operational_policy=operational_policy,
    )
    if decision == "AUTO_PASS":
        return decision, "NORMAL"
    extent = "ANOMALY_EXTENSIVE" if anomalous_area_ratio >= 0.05 else "ANOMALY_LOCALIZED"
    return decision, extent
