"""OpenF1 -> PitMind CSV-contract bridge + client (todo.md M5 / plan C1).

OpenF1 (api.openf1.org) exposes F1 car telemetry as flat JSON rows. This module
converts a session's `car_data` into the same 13-column contract that
`pitmind/` consumes and that `f1/fastf1_bridge.py` produces — so a user can
point the CLI at either FastF1 or OpenF1 and get an identical analysis.

OpenF1 units vs contract:
  - speed  (km/h, unchanged)                      -> speed_kmh
  - throttle (0-100 %, /100)                      -> throttle (0-1)
  - brake  (0/1, unchanged)                       -> brake (0/1 bool->0/1)
  - n_gear (int, -1 reverse .. 8)                 -> gear
  - rpm    (unchanged)                            -> rpm
  - x / y / z  (already meters)                   -> x / y / z

Synthesized like the FastF1 bridge:
  - `track_position` = normalized cumulative arc length of x/y path in [0,1)
  - `sector` from the `[sector]` segment when provided (OpenF1 car_data doesn't
    broadcast it per-row; we derive it by splitting the lap into thirds of arc)
  - `timestamp` from OpenF1 `date` (epoch) when present, else a 0..n-1 ramp

NOTE (non-commercial / live tier):
  * Historical REST (`car_data`) is free for personal use on sessions 2023+.
  * Live during-session streaming (MQTT/WebSocket) is the OpenF1 sponsor tier.
  Network access is required; tests mock HTTP (see tests/fixtures/openf1_*.json).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1.fastf1_bridge import (
    BRIDGE_COLUMNS,
    STEERING_SENTINEL,
    _synthesize_track_position,
)

OPENF1_BASE = "https://api.openf1.org/v1"


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
def fetch_car_data(
    session_key: int,
    driver_number: int,
    *,
    base_url: str = OPENF1_BASE,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """GET OpenF1 car_data for one driver+session and return the raw rows."""
    qs = urllib.parse.urlencode({
        "session_key": session_key,
        "driver_number": driver_number,
    })
    url = f"{base_url}/car_data?{qs}"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bridge
# --------------------------------------------------------------------------- #
@dataclass
class OpenF1Bridge:
    """Convert an OpenF1 car_data DataFrame to the PitMind contract."""

    def convert(self, car_data: pd.DataFrame) -> pd.DataFrame:
        """One OpenF1 telemetry frame (rows sorted by time) -> one session frame."""
        if car_data is None or len(car_data) < 2:
            return pd.DataFrame(columns=BRIDGE_COLUMNS)
        df = car_data.sort_values("date").reset_index(drop=True)

        n = len(df)
        speed = _num(df, "speed").to_numpy(dtype=float)
        throttle = _num(df, "throttle").to_numpy(dtype=float) / 100.0
        brake = _num(df, "brake").fillna(0).to_numpy(dtype=float)
        brake = (brake > 0.0).astype(float)
        gear = _num(df, "n_gear").fillna(0).to_numpy(dtype=int)
        rpm = _num(df, "rpm").to_numpy(dtype=float)
        x = _num(df, "x").to_numpy(dtype=float)
        y = _num(df, "y").to_numpy(dtype=float)
        z = _num(df, "z").to_numpy(dtype=float)

        track_position = _synthesize_track_position(x, y, n)
        sector = _arc_sectors(track_position, n)

        return pd.DataFrame(
            {
                "timestamp": _timestamps(df, n),
                "lap_number": np.ones(n, dtype=int),
                "sector": sector,
                "track_position": track_position,
                "speed_kmh": speed,
                "throttle": throttle,
                "brake": brake,
                "steering": np.full(n, STEERING_SENTINEL, dtype=float),
                "gear": gear,
                "rpm": rpm,
                "x": x,
                "y": y,
                "z": z,
            },
            columns=BRIDGE_COLUMNS,
        )


def to_contract(car_data: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: OpenF1 rows -> contract DataFrame."""
    return OpenF1Bridge().convert(car_data)


# ------------------------------------------------------------------ helpers
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(0.0, index=df.index)


def _timestamps(df: pd.DataFrame, n: int) -> np.ndarray:
    if "date" in df.columns:
        t = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if t.notna().any():
            return np.array(t.astype("int64") / 1e9, dtype=float)  # epoch seconds
    return np.arange(n, dtype=float)


def _arc_sectors(track_position: np.ndarray, n: int) -> np.ndarray:
    """Split a lap into 3 equal-arc sectors (OpenF1 car_data has no sector col)."""
    return (np.clip(np.ceil(track_position * 3), 1, 3)).astype(int)


def to_csv(session: pd.DataFrame, path: str) -> str:
    session.to_csv(path, index=False)
    return path
