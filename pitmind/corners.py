"""Automatic corner detection from telemetry (track-agnostic).

Corners are found purely from the car's path: position (x, y) is resampled onto a
uniform arc-length grid, heading/curvature are derived, and contiguous high-
curvature regions become corners. No per-track geometry or known corner list is
required (architect.md, rule 1).

Corner *distances* are expressed two ways:
  - track_position `tp` in [0, 1) (matches the CSV contract)
  - meters `s` = tp * L, where L is the lap length estimated from the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_N = 4000   # resample points around the lap for curvature analysis
CHORD_W = 24    # half-window for chord-angle curvature (~w * ds meters each side)


@dataclass
class CornerRegion:
    index: int
    start_tp: float
    end_tp: float
    apex_tp: float
    angle_deg: float          # signed; magnitude = total turning
    radius_m: float           # 1 / peak curvature
    min_speed_kmh: float

    @property
    def name(self) -> str:
        return f"T{self.index + 1}"

    @property
    def center_tp(self) -> float:
        # unwrapped center across the lap
        c = (self.start_tp + self.end_tp) / 2.0
        return c


def resample_track_lap(lap: pd.DataFrame, n: int = GRID_N) -> pd.DataFrame:
    """Resample a lap onto a uniform track_position grid [0, 1)."""
    tp = lap["track_position"].to_numpy(dtype=float)
    order = np.argsort(tp)
    tp_sorted = tp[order]
    grid = np.linspace(0.0, 1.0, n, endpoint=False)
    data: dict[str, np.ndarray] = {"track_position": grid}
    for col in ["x", "y", "speed_kmh", "throttle", "brake", "steering"]:
        if col in lap.columns:
            data[col] = np.interp(grid, tp_sorted, lap[col].to_numpy(dtype=float)[order])
    return pd.DataFrame(data)


def track_length_m(lap: pd.DataFrame) -> float:
    """Estimate lap length from the resampled path polyline."""
    grid = resample_track_lap(lap)
    dx = np.diff(grid["x"].to_numpy())
    dy = np.diff(grid["y"].to_numpy())
    return float(np.sum(np.hypot(dx, dy)))


def curvature_profile(lap: pd.DataFrame, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tp, curvature, turn_angle) arrays.

    Curvature is signed (rad/m); negative for right-handers. Instead of raw
    per-sample heading differences (which are a random walk when the path has
    position jitter), the turning angle at point i is measured between two long
    chords: (i-w -> i) and (i -> i+w). Chord directions average out jitter, so
    straights collapse to ~0 and only sustained turning registers.
    """
    grid = resample_track_lap(lap)
    x = grid["x"].to_numpy()
    y = grid["y"].to_numpy()
    tp0 = grid["track_position"].to_numpy()
    ds = L / GRID_N
    w = CHORD_W
    i = np.arange(w, GRID_N - w)
    d1 = np.column_stack((x[i] - x[i - w], y[i] - y[i - w]))
    d2 = np.column_stack((x[i + w] - x[i], y[i + w] - y[i]))
    ang1 = np.arctan2(d1[:, 1], d1[:, 0])
    ang2 = np.arctan2(d2[:, 1], d2[:, 0])
    turn = ang2 - ang1
    turn = (turn + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    curvature = turn / (2.0 * w * ds)
    return tp0[i], curvature, turn


def detect_corners(lap: pd.DataFrame, cfg) -> list[CornerRegion]:
    """Detect corners in a lap (run on a clean/reference lap)."""
    L = track_length_m(lap)
    tp, curv, turn = curvature_profile(lap, L)

    thresh = cfg.detection.corner_curv_threshold
    engaged = np.abs(curv) > thresh

    # contiguous runs -> candidate regions
    bounds = []
    in_run = False
    for i in range(len(engaged)):
        if engaged[i] and not in_run:
            start = i
            in_run = True
        elif not engaged[i] and in_run:
            bounds.append((start, i))
            in_run = False
    if in_run:
        bounds.append((start, len(engaged)))

    # merge runs separated by a small gap (track-agnostic join, e.g. chicane teeth)
    merged = []
    for b in bounds:
        if merged and tp[b[0]] - tp[merged[-1][1]] < cfg.detection.corner_merge_m / L:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)

    regions: list[CornerRegion] = []
    for si, ei in merged:
        total_turn = float(np.sum(turn[si:ei]))  # telescopes to net heading change
        if abs(total_turn) < np.deg2rad(cfg.ranges.corner_angle_deg):
            continue
        pk = int(np.argmax(np.abs(seg)))
        apex_tp = float(tp[si + pk])
        start_tp = float(tp[si])
        end_tp = float(tp[ei - 1] if ei - 1 > si else tp[si])
        # min speed inside the region (skip blend edges)
        speed_seg = lap_speed(lap, start_tp, end_tp)
        min_sp = float(speed_seg.min()) if len(speed_seg) else np.nan
        regions.append(
            CornerRegion(
                index=len(regions),
                start_tp=start_tp,
                end_tp=end_tp,
                apex_tp=apex_tp,
                angle_deg=np.rad2deg(total_turn),
                radius_m=1.0 / max(np.abs(seg).max(), 1e-9),
                min_speed_kmh=min_sp,
            )
        )
    return regions


def lap_speed(lap: pd.DataFrame, start_tp: float, end_tp: float) -> np.ndarray:
    """Speed samples within a track_position window (handles wrap)."""
    tp = lap["track_position"].to_numpy(dtype=float)
    speed = lap["speed_kmh"].to_numpy(dtype=float)
    if start_tp <= end_tp:
        mask = (tp >= start_tp) & (tp <= end_tp)
    else:  # corner crosses the lap boundary
        mask = (tp >= start_tp) | (tp <= end_tp)
    return speed[mask]