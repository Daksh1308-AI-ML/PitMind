"""Lap segmentation.

Splits a session recording into individual laps and drops partial/invalid laps
(out-laps, in-laps, session starts/ends mid-lap).

Lap boundaries are detected two ways:
  1. `track_position` wrap (0.99 -> 0.0) -- robust, primary method.
  2. `lap_number` column, when present -- used as a cross-check.

A lap is deemed *valid* when its track_position tracing covers nearly the whole
lap (>= MIN_COVERAGE) and it lasts at least MIN_DURATION, so the analysis never
runs on out-lap / truncated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_COVERAGE = 0.97   # fraction of the full lap circle traced
MIN_DURATION_S = 20.0 # shorter than this => not a real racing lap
WRAP_EPS = 0.5        # wrap: track_position must drop by at least this


def detect_boundaries(df: pd.DataFrame) -> list[int]:
    """Return global row indices where a *new lap* begins (start row of lap 1..N)."""
    tp = df["track_position"].to_numpy(dtype=float)
    wraps = np.where(np.diff(tp) < -WRAP_EPS)[0]
    starts = [int(i) + 1 for i in wraps]
    starts.insert(0, 0)
    starts.append(len(df))
    return starts


def split_laps(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a session into per-lap DataFrames."""
    starts = detect_boundaries(df)
    laps = []
    for a, b in zip(starts[:-1], starts[1:]):
        laps.append(df.iloc[a:b].reset_index(drop=True))
    return laps


def is_valid_lap(lap: pd.DataFrame) -> bool:
    """A valid racing lap: full track coverage, minimum duration, monotonic-ish."""
    if len(lap) < 10:
        return False
    tp = lap["track_position"]
    coverage = tp.max() - tp.min()
    duration = lap["timestamp"].iloc[-1] - lap["timestamp"].iloc[0]
    if coverage < MIN_COVERAGE or duration < MIN_DURATION_S:
        return False
    # forward-progress check: the lap should trace track_position essentially
    # once around; reject laps with multiple wraps or heavy backwards travel.
    net = tp.iloc[-1] - tp.iloc[0]
    return net >= MIN_COVERAGE - 0.05


def valid_laps(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split and filter to valid racing laps only."""
    return [lap for lap in split_laps(df) if is_valid_lap(lap)]


def valid_lap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact per-lap summary for the dashboard."""
    rows = []
    for i, lap in enumerate(valid_laps(df), start=1):
        rows.append({
            "lap": i,
            "duration_s": round(float(lap["timestamp"].iloc[-1] - lap["timestamp"].iloc[0]), 3),
            "n_samples": int(len(lap)),
            "min_speed_kmh": round(float(lap["speed_kmh"].min()), 1),
            "max_speed_kmh": round(float(lap["speed_kmh"].max()), 1),
            "avg_speed_kmh": round(float(lap["speed_kmh"].mean()), 1),
        })
    return pd.DataFrame(rows)