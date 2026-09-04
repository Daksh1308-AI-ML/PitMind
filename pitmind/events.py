"""Corner event extraction: brake point, entry/apex/exit speeds, throttle-on.

For each detected corner and each lap, produce the metrics that feed mistake
detection and time-loss estimation (doc SS11, SS14, design.md corner schema).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitmind.config import Config


@dataclass
class CornerEvent:
    lap: int
    corner: int
    name: str
    start_tp: float
    end_tp: float
    apex_tp: float
    brake_start_tp: float
    brake_point_m: float
    entry_speed_kmh: float
    apex_speed_kmh: float
    exit_speed_kmh: float
    throttle_on_s: float
    steering_max: float
    corner_time_s: float
    corner_speed_mins: float  # min speed inside corner (m/s-normalised? kmh)


def _tp_wrap_delta(a: float, b: float) -> float:
    """Signed shortest distance along the lap from a (start) to b (end)."""
    d = b - a
    return d if abs(d) <= 0.5 else (d - 1.0 if d > 0 else d + 1.0)


def _mask_window(tp: np.ndarray, start_tp: float, end_tp: float) -> np.ndarray:
    if start_tp <= end_tp:
        return (tp >= start_tp) & (tp <= end_tp)
    return (tp >= start_tp) | (tp <= end_tp)


def extract_corner(lap: pd.DataFrame, corner, L: float, cfg: Config) -> CornerEvent:
    tp = lap["track_position"].to_numpy(dtype=float)
    t = lap["timestamp"].to_numpy(dtype=float)
    speed = lap["speed_kmh"].to_numpy(dtype=float)
    brk = lap["brake"].to_numpy(dtype=float)
    thr = lap["throttle"].to_numpy(dtype=float)
    steer = lap["steering"].to_numpy(dtype=float)
    lap_no = int(lap["lap_number"].iloc[0])

    st, en = corner.start_tp, corner.end_tp
    in_corner = _mask_window(tp, st, en)
    idx = np.where(in_corner)[0]
    if len(idx) < 3:
        return CornerEvent(lap=lap_no, corner=corner.index, name=corner.name, start_tp=st, end_tp=en,
                           apex_tp=corner.apex_tp, brake_start_tp=np.nan, brake_point_m=np.nan,
                           entry_speed_kmh=np.nan, apex_speed_kmh=np.nan, exit_speed_kmh=np.nan,
                           throttle_on_s=np.nan, steering_max=np.nan, corner_time_s=np.nan,
                           corner_speed_mins=np.nan)
    first, last = idx[0], idx[-1]

    # ---- braking onset before the corner ----
    lookback_tp = (cfg.detection.max_brake_lookback_m / L)
    win_start_tp = (st - lookback_tp) % 1.0
    in_win = _mask_window(tp, win_start_tp, st)
    win_idx = np.where(in_win)[0]
    brake_point_m = np.nan
    brake_start_tp = np.nan
    if len(win_idx) >= 2:
        brk_win = brk[win_idx]
        on = win_idx[brk_win >= cfg.detection.min_brake_pressure]
        if len(on):
            # find the trailing contiguous brake-on run ending at the last on-index
            k = len(on) - 1
            while k > 0 and on[k] - on[k - 1] == 1:
                k -= 1
            onset = on[k]
            brake_start_tp = float(tp[onset])
            brake_point_m = _tp_wrap_delta(brake_start_tp, st) * L  # meters before corner start

    # ---- speeds ----
    entry_speed_kmh = float(speed[first])
    apex_i = int(idx[np.argmin(speed[idx])])
    apex_speed_kmh = float(speed[apex_i])
    exit_speed_kmh = float(speed[last])

    # ---- throttle-on after apex ----
    after_apex = idx[idx >= apex_i]
    throttle_on_s = np.nan
    if len(after_apex):
        for j in after_apex:
            if thr[j] >= cfg.detection.throttle_resume:
                throttle_on_s = float(t[j] - t[apex_i])
                break

    corner_time_s = float(t[last] - t[first])
    return CornerEvent(
        lap=lap_no, corner=corner.index, name=corner.name, start_tp=st, end_tp=en, apex_tp=corner.apex_tp,
        brake_start_tp=brake_start_tp, brake_point_m=brake_point_m,
        entry_speed_kmh=entry_speed_kmh, apex_speed_kmh=apex_speed_kmh, exit_speed_kmh=exit_speed_kmh,
        throttle_on_s=throttle_on_s, steering_max=float(np.abs(steer[idx]).max()),
        corner_time_s=corner_time_s, corner_speed_mins=apex_speed_kmh,
    )


def corner_features_table(laps: list[pd.DataFrame], corners: list, L: float, cfg: Config) -> pd.DataFrame:
    """Per-lap, per-corner feature table (design.md "corner dataset")."""
    rows = []
    for lap in laps:
        for corner in corners:
            ev = extract_corner(lap, corner, L, cfg)
            rows.append(vars(ev))
    return pd.DataFrame(rows)
