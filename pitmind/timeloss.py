"""Kinematic time-loss estimation per mistake.

Uses the reference corner time + kinematic model to estimate how much time
each mistake costs. Based on doc v0.3-0.4 time-loss methodology.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pitmind import mistakes
from pitmind.config import Config


@dataclass(frozen=True)
class TimeLoss:
    """Time loss estimate for a single mistake."""
    lap: int
    corner: int
    corner_name: str
    mistake_type: str
    confidence: str
    time_loss_s: float          # estimated seconds lost
    delta_value: float          # raw delta that caused the loss
    method: str                 # "kinematic" | "empirical" | "corner_time_delta"


def _kinematic_brake_loss(delta_m: float, entry_speed_ms: float, cfg: Config) -> float:
    """Time loss from braking early/late.

    Early braking: lost time = distance / avg_speed during over-braking
    Late braking: lost time from overshoot + correction (more complex)

    Simplified: time = delta_distance / (entry_speed * 0.5) for early brake
    """
    if delta_m < 0:  # early brake
        # Time to cover extra distance at reduced speed
        avg_speed = entry_speed_ms * 0.6  # rough avg during brake zone
        return abs(delta_m) / max(avg_speed, 1.0)
    else:  # late brake
        # Overshoot + recovery: roughly 2x the distance at corner speed
        corner_speed = entry_speed_ms * 0.5
        return 2.0 * delta_m / max(corner_speed, 1.0)


def _kinematic_apex_loss(delta_kmh: float, ref_speed_kmh: float, radius_m: float, cfg: Config) -> float:
    """Time loss from low apex speed.

    delta_kmh = actual - reference (negative = slower)
    Time loss ≈ corner_arc_length * (1/v_actual - 1/v_ref)
    """
    v_ref_kmh = ref_speed_kmh
    v_act_kmh = v_ref_kmh + delta_kmh  # actual = ref + delta (delta is negative)

    # Approximate corner arc length from radius and typical angle
    # Using 90 degrees = pi/2 rad as typical
    arc_len = radius_m * 1.57

    v_ref = v_ref_kmh / 3.6
    v_act = v_act_kmh / 3.6

    if v_act <= 0:
        return 1.0  # fallback
    return arc_len * (1.0 / v_act - 1.0 / v_ref)


def _kinematic_throttle_loss(delta_s: float, exit_speed_ms: float, cfg: Config) -> float:
    """Time loss from late throttle application.

    Simple: delta_s seconds of delayed acceleration
    Distance lost ≈ 0.5 * a * t^2, but we just use time directly
    """
    return delta_s * 0.8  # 80% of delay translates to lap time


def _kinematic_exit_loss(delta_kmh: float, ref_speed_kmh: float, straight_len_m: float, cfg: Config) -> float:
    """Time loss from slow exit speed on following straight.

    delta_kmh = actual - reference (negative = slower)
    Time lost on straight = straight_len * (1/v_act - 1/v_ref)
    """
    v_ref_kmh = ref_speed_kmh
    v_act_kmh = v_ref_kmh + delta_kmh

    v_ref = v_ref_kmh / 3.6
    v_act = v_act_kmh / 3.6

    if v_act <= 0 or straight_len_m <= 0:
        return 0.5  # fallback
    return straight_len_m * (1.0 / v_act - 1.0 / v_ref)


def _kinematic_steering_loss(steering_max: float, corner_time_s: float, cfg: Config) -> float:
    """Time loss from excessive steering (overdriving, scrubbing speed)."""
    thresh = cfg.ranges.steering_excess
    excess = abs(steering_max) - thresh
    if excess <= 0:
        return 0.0
    # Roughly proportional to corner time and excess steering
    return corner_time_s * (excess / thresh) * 0.1


def estimate_time_loss(
    mistake_list: list[mistakes.Mistake],
    feature_table: pd.DataFrame,
    cfg: Config,
    straight_lengths: dict[int, float] | None = None,
) -> list[TimeLoss]:
    """Estimate time loss for each detected mistake using kinematic model.

    Args:
        mistake_list: Output from mistakes.detect_mistakes
        feature_table: Feature table with reference deltas and corner data
        cfg: Configuration
        straight_lengths: Optional dict mapping corner_index -> straight length after corner
                         If None, uses default 200m

    Returns:
        List of TimeLoss objects
    """
    if straight_lengths is None:
        straight_lengths = {}

    # Build lookup for corner data from feature table (reference lap)
    ref_data = feature_table[feature_table["lap"] == 1].set_index("corner")

    time_losses: list[TimeLoss] = []

    for m in mistake_list:
        corner_idx = m.corner

        # Get reference corner data
        ref = ref_data.loc[corner_idx] if corner_idx in ref_data.index else None

        if m.mistake_type == mistakes.MistakeType.EARLY_BRAKE or m.mistake_type == mistakes.MistakeType.LATE_BRAKE:
            entry_speed = ref["entry_speed_kmh"] / 3.6 if ref is not None and pd.notna(ref["entry_speed_kmh"]) else 80/3.6
            loss = _kinematic_brake_loss(m.delta_value, entry_speed, cfg)
            method = "kinematic_brake"

        elif m.mistake_type == mistakes.MistakeType.LOW_APEX_SPEED:
            ref_apex = ref["apex_speed_kmh"] if ref is not None and pd.notna(ref["apex_speed_kmh"]) else 100.0
            radius = ref["apex_speed_kmh"] if ref is not None else 100  # fallback
            # Estimate radius from apex speed: v = sqrt(a_lat * r) -> r = v^2 / a_lat
            v_ref = ref_apex / 3.6
            radius = (v_ref ** 2) / 14.0  # A_LAT = 14
            loss = _kinematic_apex_loss(m.delta_value, ref_apex, radius, cfg)
            method = "kinematic_apex"

        elif m.mistake_type == mistakes.MistakeType.LATE_THROTTLE:
            exit_speed = ref["exit_speed_kmh"] / 3.6 if ref is not None and pd.notna(ref["exit_speed_kmh"]) else 100/3.6
            loss = _kinematic_throttle_loss(m.delta_value, exit_speed, cfg)
            method = "kinematic_throttle"

        elif m.mistake_type == mistakes.MistakeType.SLOW_EXIT:
            ref_exit = ref["exit_speed_kmh"] if ref is not None and pd.notna(ref["exit_speed_kmh"]) else 100.0
            straight_len = straight_lengths.get(corner_idx, 200.0)
            loss = _kinematic_exit_loss(m.delta_value, ref_exit, straight_len, cfg)
            method = "kinematic_exit"

        elif m.mistake_type == mistakes.MistakeType.EXCESS_STEERING:
            corner_time = ref["corner_time_s"] if ref is not None and pd.notna(ref["corner_time_s"]) else 3.0
            loss = _kinematic_steering_loss(m.delta_value, corner_time, cfg)
            method = "kinematic_steering"

        else:
            loss = 0.0
            method = "unknown"

        # Clamp to reasonable bounds
        loss = max(0.0, min(loss, 5.0))

        time_losses.append(TimeLoss(
            lap=m.lap,
            corner=m.corner,
            corner_name=m.corner_name,
            mistake_type=m.mistake_type.value,
            confidence=m.confidence.value,
            time_loss_s=loss,
            delta_value=m.delta_value,
            method=method,
        ))

    return time_losses


def time_loss_to_dataframe(time_losses: list[TimeLoss]) -> pd.DataFrame:
    """Convert to DataFrame."""
    if not time_losses:
        return pd.DataFrame(columns=[
            "lap", "corner", "corner_name", "mistake_type", "confidence",
            "time_loss_s", "delta_value", "method",
        ])
    return pd.DataFrame([
        {
            "lap": t.lap,
            "corner": t.corner,
            "corner_name": t.corner_name,
            "mistake_type": t.mistake_type,
            "confidence": t.confidence,
            "time_loss_s": t.time_loss_s,
            "delta_value": t.delta_value,
            "method": t.method,
        }
        for t in time_losses
    ])


def total_time_loss_per_lap(time_losses: list[TimeLoss]) -> dict[int, float]:
    """Aggregate time loss by lap."""
    totals: dict[int, float] = {}
    for t in time_losses:
        totals[t.lap] = totals.get(t.lap, 0.0) + t.time_loss_s
    return totals


def total_time_loss_per_corner(time_losses: list[TimeLoss]) -> dict[str, float]:
    """Aggregate time loss by corner."""
    totals: dict[str, float] = {}
    for t in time_losses:
        totals[t.corner_name] = totals.get(t.corner_name, 0.0) + t.time_loss_s
    return totals
