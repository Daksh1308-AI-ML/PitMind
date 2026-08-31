"""Feature engineering: thin wrapper around corner_features_table + reference deltas.

This module provides a single entry point to get the full feature table
with reference comparisons included. It does not duplicate the extraction
logic from events.corner_features_table.
"""

from __future__ import annotations

import pandas as pd

from pitmind.config import Config
from pitmind import corners, events, reference, segmentation


def build_feature_table(
    session: pd.DataFrame,
    cfg: Config,
    detected_corners: list | None = None,
    L: float | None = None,
    reference_lap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete corner feature table with reference deltas.

    This is the main entry point for downstream consumers (mistakes, timeloss, coaching).

    Args:
        session: Full session DataFrame (multi-lap)
        cfg: Configuration
        detected_corners: Optional pre-detected corners. If None, runs detection on lap 1.
        L: Optional pre-computed lap length. If None, computed from lap 1.
        reference_lap: Optional pre-selected reference lap.

    Returns:
        DataFrame with one row per (lap, corner) and columns:
        - Basic corner metrics (from events.corner_features_table)
        - Delta columns vs reference lap (delta_brake_point_m, delta_apex_speed_kmh, etc.)
        - lap_number, corner index, name
    """
    # Preprocess if needed (ensure resampled/smoothed)
    from pitmind import preprocess
    session = preprocess.preprocess(session, cfg)

    # Get valid laps
    laps = segmentation.valid_laps(session)
    if not laps:
        raise ValueError("No valid laps in session")

    # Detect corners if not provided (use first clean lap)
    if detected_corners is None:
        lap1 = session[session.lap_number == 1].reset_index(drop=True)
        if len(lap1) == 0:
            lap1 = laps[0]
        detected_corners = corners.detect_corners(lap1, cfg)

    # Compute lap length if not provided
    if L is None:
        L = corners.track_length_m(laps[0])

    # Build base feature table
    table = events.corner_features_table(laps, detected_corners, L, cfg)

    # Build reference and add deltas
    ref = reference.build_references(laps, detected_corners, L, cfg, reference_lap)
    table = reference.compare_to_reference(table, ref)

    return table


def get_corner_features_for_lap(
    table: pd.DataFrame,
    lap_number: int,
) -> pd.DataFrame:
    """Filter feature table to a specific lap."""
    return table[table["lap"] == lap_number].copy()


def get_corner_features_for_corner(
    table: pd.DataFrame,
    corner_index: int,
) -> pd.DataFrame:
    """Filter feature table to a specific corner."""
    return table[table["corner"] == corner_index].copy()


def get_worst_corners(
    table: pd.DataFrame,
    metric: str = "delta_corner_time_s",
    top_k: int = 3,
) -> pd.DataFrame:
    """Get the worst corners for a given lap by a metric.

    Args:
        table: Feature table (typically for a single lap)
        metric: Column to rank by (higher = worse)
        top_k: Number of corners to return

    Returns:
        Top-k worst corners sorted by metric descending
    """
    if metric not in table.columns:
        raise ValueError(f"Metric '{metric}' not in table columns")
    return table.nlargest(top_k, metric)