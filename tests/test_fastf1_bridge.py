"""Offline tests for the FastF1 -> PitMind bridge.

These run WITHOUT fastf1 installed or any network access. We build a synthetic
FastF1-shaped telemetry DataFrame (the exact columns/units FastF1 emits:
Speed in km/h, Throttle 0-100, Brake bool, nGear, RPM, X/Y/Z in 1/10 m) and
assert the bridge converts it to the 13-column PitMind contract correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1.fastf1_bridge import (
    BRIDGE_COLUMNS,
    FastF1Bridge,
    _synthesize_track_position,
)


def _fake_lap_tel(n: int = 100, x0: float = 0.0, y0: float = 0.0, t0: float = 0.0) -> pd.DataFrame:
    """A synthetic FastF1-shaped telemetry frame (contract *input* units)."""
    t = np.linspace(0.0, 90.0, n) + t0
    # a fake "oval-ish" closed path so arc length is well-defined
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = 100.0
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    z = np.zeros(n)
    return pd.DataFrame(
        {
            "Time": t,
            "Speed": np.linspace(80.0, 260.0, n),          # km/h
            "Throttle": np.linspace(0.0, 100.0, n),        # 0-100 %
            "Brake": (np.sin(theta * 3) > 0.5).astype(bool),  # boolean
            "nGear": np.full(n, 6, dtype=int),
            "RPM": np.linspace(5000.0, 11500.0, n),
            "X": (x * 10).round().astype(int),             # 1/10 m
            "Y": (y * 10).round().astype(int),             # 1/10 m
            "Z": (z * 10).round().astype(int),             # 1/10 m
            "SectorNumber": np.full(n, 2, dtype=int),
        }
    )


# ------------------------------------------------------------- helpers ----
def test_bridge_columns_match_contract():
    assert BRIDGE_COLUMNS == [
        "timestamp", "lap_number", "sector", "track_position",
        "speed_kmh", "throttle", "brake", "steering", "gear", "rpm", "x", "y", "z",
    ]


def test_single_lap_conversion_shape_and_units():
    lt = _fake_lap_tel(n=120)
    out = FastF1Bridge().convert_lap(lt)
    assert list(out.columns) == BRIDGE_COLUMNS
    assert len(out) == 120

    # unit conversions
    assert np.isclose(out["throttle"].iloc[0], 0.0)      # 0 % -> 0
    assert np.isclose(out["throttle"].iloc[-1], 1.0)     # 100 % -> 1
    assert set(np.unique(out["brake"])) <= {0.0, 1.0}    # bool -> 0/1
    assert set(np.unique(out["gear"])) == {6}
    # X/Y/Z divided by 10 (1/10 m -> m): max radius ~100 m
    assert np.abs(out["x"]).max() < 105
    assert np.isclose(out["speed_kmh"].iloc[0], 80.0)
    assert np.isclose(out["rpm"].iloc[-1], 11500.0)


def test_steering_filled_with_nan_on_f1():
    lt = _fake_lap_tel(n=60)
    out = FastF1Bridge().convert_lap(lt)
    assert out["steering"].isna().all()
    assert np.isnan(out["steering"].iloc[0])


def test_track_position_is_normalized_arc_length():
    lt = _fake_lap_tel(n=200)
    out = FastF1Bridge().convert_lap(lt)
    tp = out["track_position"].to_numpy()
    # starts near 0, ends near 1, monotonic non-decreasing
    assert tp[0] == pytest.approx(0.0, abs=1e-6)
    assert tp[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(tp) >= -1e-9)


def test_track_position_handles_flat_or_missing_path():
    # constant coords -> zero arc length -> fallback ramp
    n = 50
    lt = _fake_lap_tel(n=n)
    lt["X"] = 1230  # all identical
    lt["Y"] = 4560
    out = FastF1Bridge().convert_lap(lt)
    tp = out["track_position"].to_numpy()
    assert tp[0] == pytest.approx(0.0)
    # fallback ramp is linspace(0,1,endpoint=False) -> last = (n-1)/n
    assert tp[-1] == pytest.approx((n - 1) / n, abs=1e-6)
    assert np.all(np.diff(tp) >= -1e-9)


def test_synthesize_track_position_direct():
    x = np.array([0.0, 0.0, 3.0, 6.0])
    y = np.array([0.0, 4.0, 4.0, 4.0])
    tp = _synthesize_track_position(x, y, 4)
    # cumulative: 0,4,3,3 -> total 10 -> normalized 0,0.4,0.7,1.0
    assert tp[0] == pytest.approx(0.0)
    assert tp[1] == pytest.approx(0.4)
    assert tp[2] == pytest.approx(0.7)
    assert tp[3] == pytest.approx(1.0)


def test_multi_lap_session_assigns_lap_numbers_and_concatenates():
    bridge = FastF1Bridge()
    frames = [_fake_lap_tel(n=80, t0=0.0), _fake_lap_tel(n=70, t0=100.0), _fake_lap_tel(n=60, t0=200.0)]
    session = bridge.convert_session(frames)
    assert len(session["lap_number"].unique()) == 3
    assert set(session["lap_number"].unique()) == {1, 2, 3}
    assert len(session) == 80 + 70 + 60


def test_empty_session_returns_empty_contract():
    session = FastF1Bridge().convert_session([])
    assert list(session.columns) == BRIDGE_COLUMNS
    assert len(session) == 0


def test_lap_number_column_from_input_frames_is_overridden():
    # even if the input carries its own lap marker, bridge assigns 1..N
    lt = _fake_lap_tel(n=50)
    lt["LapNumber"] = 7
    session = FastF1Bridge().convert_session([lt])
    assert set(session["lap_number"].unique()) == {1}


def test_prunes_excess_steering_when_capability_off():
    """F1 has no steering channel -> EXCESS_STEERING must never fire."""
    from pitmind import mistakes
    from pitmind.config import Config

    # hand-build a minimal feature table row with a large steering_max (would
    # trip EXCESS_STEERING if the flag weren't honoured)
    table = pd.DataFrame([
        {
            "lap": 1, "corner": 1, "name": "T1",
            "delta_brake_point_m": np.nan, "delta_apex_speed_kmh": np.nan,
            "delta_throttle_on_s": np.nan, "delta_exit_speed_kmh": np.nan,
            "steering_max": 0.9,
        }
    ])
    cfg = Config.from_file()

    with_cap = mistakes.detect_mistakes(table, cfg, capabilities={"steering": False})
    assert not any(m.mistake_type == mistakes.MistakeType.EXCESS_STEERING for m in with_cap)

    with_steering = mistakes.detect_mistakes(table, cfg)
    assert any(m.mistake_type == mistakes.MistakeType.EXCESS_STEERING for m in with_steering)
