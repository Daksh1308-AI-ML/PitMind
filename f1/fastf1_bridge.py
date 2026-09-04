"""FastF1 -> PitMind CSV-contract bridge.

This module is the single conversion point from real Formula 1 telemetry
(FastF1) to the 13-column session DataFrame that `pitmind/` consumes
(design.md "CSV Telemetry Schema (the contract)").

Unit conversions are the bridge's job:
  - Speed      -> speed_kmh     (km/h, unchanged)
  - Throttle   -> throttle      (0-100 % -> 0-1, /100)
  - Brake      -> brake         (bool -> 0.0 / 1.0)
  - nGear      -> gear          (int, -1 reverse ... 8)
  - RPM        -> rpm
  - X/Y/Z      -> x/y/z         (1/10 m -> meters, /10)

Synthesized fields:
  - `track_position` is NOT broadcast by F1. It is synthesized as the
    normalized cumulative arc length of the x/y path (0 at lap start, 1 at
    lap end). This is the one non-trivial derivation (architect.md): F1
    samples at ~4 Hz, but our chord-based corner detection resamples onto
    its own 4000-point arc grid, so coarseness is fine.

The bridge produces a full multi-lap session DataFrame (concatenated laps).
Requires the optional "f1" extra:  pip install -e ".[f1]"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BRIDGE_COLUMNS = [
    "timestamp",
    "lap_number",
    "sector",
    "track_position",
    "speed_kmh",
    "throttle",
    "brake",
    "steering",
    "gear",
    "rpm",
    "x",
    "y",
    "z",
]

# F1 broadcasts no steering channel -> column is filled with NaN.
STEERING_SENTINEL = float("nan")


@dataclass
class FastF1Bridge:
    """Convert a FastF1 per-lap telemetry DataFrame to the PitMind contract.

    FastF1 users normally obtain telemetry via:
        lap = session.laps.pick_driver("VER").pick_fastest()
        telemetry = lap.get_car_data().add_distance()  # per-lap DataFrame

    This bridge accepts either:
      1. a single lap's telemetry (convert -> pad to a 1-lap session), or
      2. a list of lap telemetry DataFrames (one per lap -> multi-lap session).
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ API
    def convert_lap(self, lap_telemetry: pd.DataFrame) -> pd.DataFrame:
        """Convert ONE FastF1 lap telemetry frame to one PitMind lap frame."""
        return self._convert_lap(lap_telemetry)

    def convert_session(self, lap_telemetry: list[pd.DataFrame]) -> pd.DataFrame:
        """Convert a list of lap telemetry frames into a multi-lap session.

        Lap numbers are assigned in the order given (1, 2, ...). Sector is
        derived per lap from the dataset's `SectorNumber` channel when present,
        else defaulted to 1.
        """
        frames = [self._convert_lap(lt, lap_no=i + 1) for i, lt in enumerate(lap_telemetry)]
        if not frames:
            return pd.DataFrame(columns=BRIDGE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # -------------------------------------------------------------- internals
    def _convert_lap(self, lt: pd.DataFrame, lap_no: int = 1) -> pd.DataFrame:
        if lt is None or len(lt) < 2:
            return pd.DataFrame(columns=BRIDGE_COLUMNS)

        x = pd.to_numeric(lt["X"], errors="coerce").to_numpy(dtype=float) / 10.0
        y = pd.to_numeric(lt["Y"], errors="coerce").to_numpy(dtype=float) / 10.0
        z = pd.to_numeric(lt["Z"], errors="coerce").to_numpy(dtype=float) / 10.0

        n = len(lt)
        timestamp = _as_timestamp(lt, n)
        speed = pd.to_numeric(lt["Speed"], errors="coerce").to_numpy(dtype=float)
        throttle = pd.to_numeric(lt["Throttle"], errors="coerce").to_numpy(dtype=float) / 100.0
        brake = pd.to_numeric(lt["Brake"], errors="coerce").fillna(0).to_numpy(dtype=float)
        brake = (brake > 0.0).astype(float)
        gear = pd.to_numeric(lt["nGear"], errors="coerce").fillna(0).to_numpy(dtype=int)
        rpm = pd.to_numeric(lt["RPM"], errors="coerce").to_numpy(dtype=float)

        sector = _as_sector(lt, n)

        track_position = _synthesize_track_position(x, y, n)

        return pd.DataFrame(
            {
                "timestamp": timestamp,
                "lap_number": np.full(n, int(lap_no), dtype=int),
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


# ------------------------------------------------------------------ helpers
def _as_timestamp(lt: pd.DataFrame, n: int) -> np.ndarray:
    """timestamp from session time if present, else a 0..n-1 ramp."""
    if "Time" in lt.columns:
        t = pd.to_numeric(lt["Time"], errors="coerce").to_numpy(dtype=float)
        t = np.where(np.isnan(t), np.arange(n, dtype=float), t)
        return t
    return np.arange(n, dtype=float)


def _as_sector(lt: pd.DataFrame, n: int) -> np.ndarray:
    if "SectorNumber" in lt.columns:
        s = pd.to_numeric(lt["SectorNumber"], errors="coerce").fillna(1).to_numpy(dtype=int)
        return np.clip(s, 1, 3).astype(int)
    return np.ones(n, dtype=int)


def _synthesize_track_position(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Normalized cumulative arc length of the x/y path in [0, 1)."""
    if n < 2:
        return np.zeros(n, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    if np.all(~np.isfinite(seg)) or seg.sum() <= 0:
        return np.linspace(0.0, 1.0, n, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    return (cum / total).astype(float)


def bridge_to_csv(session: pd.DataFrame, path: str) -> str:
    """Write a bridged session DataFrame to CSV in the contract schema."""
    session.to_csv(path, index=False)
    return path
