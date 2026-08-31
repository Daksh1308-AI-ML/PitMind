"""Telemetry preprocessing: resample to a fixed rate, smooth, fill gaps.

Design rule: thresholds/rates come from config, never magic numbers (design.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitmind.config import Config

_NUMERIC_COLS = ["speed_kmh", "throttle", "brake", "steering", "rpm", "x", "y", "z"]


def _resample_lap(lap: pd.DataFrame, sample_rate_hz: float) -> pd.DataFrame:
    """Resample one lap's numeric channels onto a uniform time grid."""
    if len(lap) < 2:
        return lap
    t = lap["timestamp"].to_numpy(dtype=float)
    dt = 1.0 / sample_rate_hz
    t_new = np.arange(t[0], t[-1] + dt, dt)
    out = {"timestamp": t_new}
    out["track_position"] = np.interp(t_new, t, lap["track_position"].to_numpy(dtype=float))
    for col in _NUMERIC_COLS:
        if col in lap.columns:
            out[col] = np.interp(t_new, t, lap[col].to_numpy(dtype=float))
    for col in ["lap_number", "sector", "gear"]:
        if col in lap.columns:
            vals = lap[col].to_numpy(dtype=float)
            filled = np.interp(t_new, t, vals)
            out[col] = np.rint(filled).astype(int)
    return pd.DataFrame(out)


def resample(df: pd.DataFrame, sample_rate_hz: float) -> pd.DataFrame:
    """Resample a whole session (per lap) to a uniform sampling rate."""
    frames = [_resample_lap(lap, sample_rate_hz) for _, lap in df.groupby("lap_number")]
    return pd.concat(frames, ignore_index=True)


def smooth(df: pd.DataFrame, window: int = 5, cols: tuple = _NUMERIC_COLS) -> pd.DataFrame:
    """Rolling-mean smoothing (per lap boundaries, to avoid bleeding across laps)."""
    out = df.copy()
    for _, lap_idx in out.groupby("lap_number").groups.items():
        rows = out.loc[lap_idx]
        for col in cols:
            if col in out.columns:
                out.loc[rows.index, col] = rows[col].rolling(window, center=True, min_periods=1).mean()
    return out


def preprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full preprocessing pipeline: validate -> resample -> smooth."""
    required = ["timestamp", "track_position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"telemetry missing required columns: {missing}")

    clean = df.copy().sort_values("timestamp")
    clean = clean.dropna(subset=["timestamp", "track_position"])
    clean = clean.reset_index(drop=True)
    clean = resample(clean, float(cfg.detection.sample_rate_hz))
    clean = smooth(clean, window=cfg.synthetic.get("smooth_window", 5))
    return clean