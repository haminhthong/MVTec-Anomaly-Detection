"""Operational Decision Engine for visual quality control.

Decision categories:
- PASS: anomaly_score < review_threshold (Product meets quality standards)
- REVIEW: review_threshold <= anomaly_score < fail_threshold (Requires secondary QC inspection)
- FAIL: anomaly_score >= fail_threshold (Defect exceeds tolerance)

Severity classification:
- PASS: Normal
- REVIEW: Warning
- FAIL_MINOR: Small defect area and moderate peak distance
- FAIL_MAJOR: Extensive defect area (>= 5% of surface) or high peak distance (>= 1.5 * fail_threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from ..model.artifacts import ThresholdPolicy


@dataclass(frozen=True)
class OperationalPolicy:
    """Operational inspection policy separating statistical thresholds from business QC rules."""

    review_threshold: float
    fail_threshold: float
    pixel_threshold: float
    minor_area_ratio: float = 0.01
    major_area_ratio: float = 0.05
    peak_score_multiplier: float = 1.5

    @classmethod
    def from_threshold_policy(
        cls,
        policy: ThresholdPolicy,
        minor_area_ratio: float = 0.01,
        major_area_ratio: float = 0.05,
        peak_score_multiplier: float = 1.5,
    ) -> OperationalPolicy:
        """Construct OperationalPolicy from a calibrated ThresholdPolicy."""
        return cls(
            review_threshold=policy.review_threshold,
            fail_threshold=policy.fail_threshold,
            pixel_threshold=policy.pixel_threshold,
            minor_area_ratio=minor_area_ratio,
            major_area_ratio=major_area_ratio,
            peak_score_multiplier=peak_score_multiplier,
        )


def classify_decision_and_severity(
    anomaly_score: float,
    threshold_policy: ThresholdPolicy | None = None,
    operational_policy: OperationalPolicy | None = None,
    review_threshold: float | None = None,
    fail_threshold: float | None = None,
    anomalous_area_ratio: float = 0.0,
    peak_score: float = 0.0,
) -> tuple[str, str]:
    """Classify inspection sample into operational decision and severity tier.

    Args:
        anomaly_score: Global anomaly score of sample.
        threshold_policy: Optional ThresholdPolicy object.
        operational_policy: Optional OperationalPolicy object.
        review_threshold: Optional review threshold (used if policy is None).
        fail_threshold: Optional fail threshold (used if policy is None).
        anomalous_area_ratio: Fraction of surface exceeding pixel threshold.
        peak_score: Highest anomaly score across heatmap pixels.

    Returns:
        tuple[str, str]: (decision, severity)
    """
    if operational_policy is not None:
        r_th = operational_policy.review_threshold
        f_th = operational_policy.fail_threshold
        major_ratio = operational_policy.major_area_ratio
        peak_mult = operational_policy.peak_score_multiplier
    elif threshold_policy is not None:
        r_th = threshold_policy.review_threshold
        f_th = threshold_policy.fail_threshold
        major_ratio = 0.05
        peak_mult = 1.5
    else:
        if review_threshold is None or fail_threshold is None:
            raise ValueError("Must provide either operational_policy, threshold_policy, or review_threshold and fail_threshold.")
        r_th = review_threshold
        f_th = fail_threshold
        major_ratio = 0.05
        peak_mult = 1.5

    if anomaly_score < r_th:
        return "PASS", "PASS"
    elif anomaly_score < f_th:
        return "REVIEW", "REVIEW"
    else:
        is_major = (anomalous_area_ratio >= major_ratio) or (peak_score >= (peak_mult * f_th))
        severity = "FAIL_MAJOR" if is_major else "FAIL_MINOR"
        return "FAIL", severity
