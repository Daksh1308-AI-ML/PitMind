"""Mistake detection: threshold rules → mistake classes + confidence.

Consumes the feature table with reference deltas (from features.py) and applies
configurable thresholds to classify each corner/lap into mistake types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

from pitmind.config import Config


class MistakeType(str, Enum):
    """Classified driver mistake categories."""
    EARLY_BRAKE = "early_brake"
    LATE_BRAKE = "late_brake"
    LOW_APEX_SPEED = "low_apex_speed"
    LATE_THROTTLE = "late_throttle"
    SLOW_EXIT = "slow_exit"
    EXCESS_STEERING = "excess_steering"


class ConfidenceLevel(str, Enum):
    """Confidence levels based on threshold bands."""
    WEAK = "weak"       # barely crosses threshold
    SIGNIFICANT = "significant"  # clear mistake
    STRONG = "strong"   # large deviation


@dataclass(frozen=True)
class Mistake:
    """A detected mistake instance."""
    lap: int
    corner: int
    corner_name: str
    mistake_type: MistakeType
    confidence: ConfidenceLevel
    delta_value: float      # raw delta (e.g., brake_point_m delta)
    threshold_value: float  # threshold that was crossed
    message: str            # human-readable description


# Threshold bands from config
def _get_brake_thresholds(cfg: Config) -> dict[str, float]:
    d = cfg.ranges.brake_point_delta_m
    return {
        "significant": d.get("significant", 3.0),
        "potential": d.get("potential", 10.0),
        "strong": d.get("strong", 15.0),
    }


def _classify_brake(delta_m: float, cfg: Config) -> tuple[MistakeType | None, ConfidenceLevel | None, float | None]:
    """Classify brake point delta. Positive = later than ref, negative = earlier."""
    thresh = _get_brake_thresholds(cfg)
    if delta_m <= -thresh["strong"]:
        return MistakeType.EARLY_BRAKE, ConfidenceLevel.STRONG, thresh["strong"]
    if delta_m <= -thresh["potential"]:
        return MistakeType.EARLY_BRAKE, ConfidenceLevel.SIGNIFICANT, thresh["potential"]
    if delta_m <= -thresh["significant"]:
        return MistakeType.EARLY_BRAKE, ConfidenceLevel.WEAK, thresh["significant"]
    if delta_m >= thresh["strong"]:
        return MistakeType.LATE_BRAKE, ConfidenceLevel.STRONG, thresh["strong"]
    if delta_m >= thresh["potential"]:
        return MistakeType.LATE_BRAKE, ConfidenceLevel.SIGNIFICANT, thresh["potential"]
    if delta_m >= thresh["significant"]:
        return MistakeType.LATE_BRAKE, ConfidenceLevel.WEAK, thresh["significant"]
    return None, None, None


def _classify_apex_speed(delta_kmh: float, cfg: Config) -> tuple[MistakeType | None, ConfidenceLevel | None, float | None]:
    """Negative delta = slower apex than reference."""
    thresh = cfg.ranges.apex_speed_delta_kmh
    if delta_kmh <= -thresh:
        return MistakeType.LOW_APEX_SPEED, ConfidenceLevel.SIGNIFICANT, thresh
    return None, None, None


def _classify_throttle(delta_s: float, cfg: Config) -> tuple[MistakeType | None, ConfidenceLevel | None, float | None]:
    """Positive delta = later throttle application."""
    thresh = cfg.ranges.throttle_delay_s
    if delta_s >= thresh:
        return MistakeType.LATE_THROTTLE, ConfidenceLevel.SIGNIFICANT, thresh
    return None, None, None


def _classify_exit_speed(delta_kmh: float, cfg: Config) -> tuple[MistakeType | None, ConfidenceLevel | None, float | None]:
    """Negative delta = slower exit."""
    thresh = cfg.ranges.exit_speed_delta_kmh
    if delta_kmh <= -thresh:
        return MistakeType.SLOW_EXIT, ConfidenceLevel.SIGNIFICANT, thresh
    return None, None, None


def _classify_steering(steering_max: float, cfg: Config) -> tuple[MistakeType | None, ConfidenceLevel | None, float | None]:
    """Absolute steering > threshold indicates overdriving."""
    thresh = cfg.ranges.steering_excess
    if abs(steering_max) >= thresh:
        return MistakeType.EXCESS_STEERING, ConfidenceLevel.SIGNIFICANT, thresh
    return None, None, None


def detect_mistakes(
    feature_table: pd.DataFrame,
    cfg: Config,
) -> list[Mistake]:
    """Run all mistake classifiers on the feature table.

    Args:
        feature_table: Output from features.build_feature_table (with delta columns)
        cfg: Configuration with thresholds

    Returns:
        List of Mistake objects (one per detected issue)
    """
    mistakes: list[Mistake] = []

    for _, row in feature_table.iterrows():
        lap = int(row["lap"])
        corner = int(row["corner"])
        corner_name = str(row["name"])

        # 1. Brake point
        bp_delta = row.get("delta_brake_point_m")
        if pd.notna(bp_delta):
            mtype, conf, thresh = _classify_brake(bp_delta, cfg)
            if mtype:
                direction = "early" if mtype == MistakeType.EARLY_BRAKE else "late"
                mistakes.append(Mistake(
                    lap=lap, corner=corner, corner_name=corner_name,
                    mistake_type=mtype, confidence=conf,
                    delta_value=bp_delta, threshold_value=thresh,
                    message=f"Braking {direction} by {abs(bp_delta):.1f}m at {corner_name} (lap {lap})",
                ))

        # 2. Apex speed
        apex_delta = row.get("delta_apex_speed_kmh")
        if pd.notna(apex_delta):
            mtype, conf, thresh = _classify_apex_speed(apex_delta, cfg)
            if mtype:
                mistakes.append(Mistake(
                    lap=lap, corner=corner, corner_name=corner_name,
                    mistake_type=mtype, confidence=conf,
                    delta_value=apex_delta, threshold_value=thresh,
                    message=f"Apex speed {abs(apex_delta):.1f}km/h below reference at {corner_name} (lap {lap})",
                ))

        # 3. Throttle application
        thr_delta = row.get("delta_throttle_on_s")
        if pd.notna(thr_delta):
            mtype, conf, thresh = _classify_throttle(thr_delta, cfg)
            if mtype:
                mistakes.append(Mistake(
                    lap=lap, corner=corner, corner_name=corner_name,
                    mistake_type=mtype, confidence=conf,
                    delta_value=thr_delta, threshold_value=thresh,
                    message=f"Throttle delayed by {thr_delta:.2f}s at {corner_name} (lap {lap})",
                ))

        # 4. Exit speed
        exit_delta = row.get("delta_exit_speed_kmh")
        if pd.notna(exit_delta):
            mtype, conf, thresh = _classify_exit_speed(exit_delta, cfg)
            if mtype:
                mistakes.append(Mistake(
                    lap=lap, corner=corner, corner_name=corner_name,
                    mistake_type=mtype, confidence=conf,
                    delta_value=exit_delta, threshold_value=thresh,
                    message=f"Exit speed {abs(exit_delta):.1f}km/h below reference at {corner_name} (lap {lap})",
                ))

        # 5. Excess steering
        steer = row.get("steering_max")
        if pd.notna(steer):
            mtype, conf, thresh = _classify_steering(steer, cfg)
            if mtype:
                mistakes.append(Mistake(
                    lap=lap, corner=corner, corner_name=corner_name,
                    mistake_type=mtype, confidence=conf,
                    delta_value=steer, threshold_value=thresh,
                    message=f"Excessive steering {steer:.2f} at {corner_name} (lap {lap})",
                ))

    return mistakes


def mistakes_to_dataframe(mistakes: list[Mistake]) -> pd.DataFrame:
    """Convert list of Mistake to DataFrame for analysis/plotting."""
    if not mistakes:
        return pd.DataFrame(columns=[
            "lap", "corner", "corner_name", "mistake_type", "confidence",
            "delta_value", "threshold_value", "message",
        ])
    return pd.DataFrame([
        {
            "lap": m.lap,
            "corner": m.corner,
            "corner_name": m.corner_name,
            "mistake_type": m.mistake_type.value,
            "confidence": m.confidence.value,
            "delta_value": m.delta_value,
            "threshold_value": m.threshold_value,
            "message": m.message,
        }
        for m in mistakes
    ])


def summarize_mistakes(mistakes: list[Mistake]) -> dict:
    """Aggregate statistics across all mistakes."""
    if not mistakes:
        return {"total": 0, "by_type": {}, "by_confidence": {}, "by_lap": {}, "by_corner": {}}

    df = mistakes_to_dataframe(mistakes)
    return {
        "total": len(df),
        "by_type": df["mistake_type"].value_counts().to_dict(),
        "by_confidence": df["confidence"].value_counts().to_dict(),
        "by_lap": df["lap"].value_counts().sort_index().to_dict(),
        "by_corner": df["corner_name"].value_counts().to_dict(),
    }