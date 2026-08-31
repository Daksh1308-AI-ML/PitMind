"""Reference lap selection + per-corner reference feature values.

Picks the best (fastest) clean lap and extracts per-corner reference metrics
for mistake detection and time-loss estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pitmind.config import Config
from pitmind import corners, events, segmentation


@dataclass(frozen=True)
class CornerReference:
    """Per-corner reference values from the best lap."""
    corner_index: int
    corner_name: str
    brake_point_m: float
    entry_speed_kmh: float
    apex_speed_kmh: float
    exit_speed_kmh: float
    throttle_on_s: float
    steering_max: float
    corner_time_s: float


@dataclass(frozen=True)
class ReferenceLap:
    """Reference lap data + per-corner references."""
    lap_number: int
    lap_time_s: float
    corners: tuple[CornerReference, ...]

    def get_corner(self, corner_index: int) -> CornerReference | None:
        for c in self.corners:
            if c.corner_index == corner_index:
                return c
        return None


def pick_reference_lap(laps: list[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Select the fastest valid lap as reference.

    Args:
        laps: List of valid lap DataFrames
        cfg: Configuration (unused currently, kept for future ranking logic)

    Returns:
        The reference lap DataFrame
    """
    if not laps:
        raise ValueError("No valid laps provided")
    # Find lap with minimum duration
    best_lap = min(laps, key=lambda df: df["timestamp"].max() - df["timestamp"].min())
    return best_lap


def build_references(
    laps: list[pd.DataFrame],
    detected_corners: list,
    L: float,
    cfg: Config,
    reference_lap: pd.DataFrame | None = None,
) -> ReferenceLap:
    """Build per-corner reference features from the reference lap.

    Args:
        laps: All valid laps (used if reference_lap not provided)
        detected_corners: List of detected CornerRegion objects
        L: Lap length in meters
        cfg: Configuration
        reference_lap: Optional pre-selected reference lap. If None, picks fastest.

    Returns:
        ReferenceLap with per-corner reference values
    """
    if reference_lap is None:
        reference_lap = pick_reference_lap(laps, cfg)

    lap_time = reference_lap["timestamp"].max() - reference_lap["timestamp"].min()
    lap_number = int(reference_lap["lap_number"].iloc[0])

    ref_corners: list[CornerReference] = []
    for corner in detected_corners:
        ev = events.extract_corner(reference_lap, corner, L, cfg)
        ref_corners.append(CornerReference(
            corner_index=corner.index,
            corner_name=corner.name,
            brake_point_m=ev.brake_point_m,
            entry_speed_kmh=ev.entry_speed_kmh,
            apex_speed_kmh=ev.apex_speed_kmh,
            exit_speed_kmh=ev.exit_speed_kmh,
            throttle_on_s=ev.throttle_on_s,
            steering_max=ev.steering_max,
            corner_time_s=ev.corner_time_s,
        ))

    return ReferenceLap(
        lap_number=lap_number,
        lap_time_s=lap_time,
        corners=tuple(ref_corners),
    )


def compare_to_reference(
    table: pd.DataFrame,
    reference: ReferenceLap,
) -> pd.DataFrame:
    """Add delta columns comparing each lap's corners to reference.

    Args:
        table: Feature table from events.corner_features_table
        reference: ReferenceLap with reference values

    Returns:
        Table with additional delta columns
    """
    out = table.copy()
    for ref in reference.corners:
        mask = out["corner"] == ref.corner_index
        if mask.any():
            out.loc[mask, "delta_brake_point_m"] = out.loc[mask, "brake_point_m"] - ref.brake_point_m
            out.loc[mask, "delta_entry_speed_kmh"] = out.loc[mask, "entry_speed_kmh"] - ref.entry_speed_kmh
            out.loc[mask, "delta_apex_speed_kmh"] = out.loc[mask, "apex_speed_kmh"] - ref.apex_speed_kmh
            out.loc[mask, "delta_exit_speed_kmh"] = out.loc[mask, "exit_speed_kmh"] - ref.exit_speed_kmh
            out.loc[mask, "delta_throttle_on_s"] = out.loc[mask, "throttle_on_s"] - ref.throttle_on_s
            out.loc[mask, "delta_corner_time_s"] = out.loc[mask, "corner_time_s"] - ref.corner_time_s
    return out