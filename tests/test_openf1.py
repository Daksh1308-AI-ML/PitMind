"""Tests for f1.openf1 — OpenF1 client + bridge to the PitMind contract (M5/C1).

No network: the bridge is tested against a committed JSON fixture, and the HTTP
fetch path is covered with a mocked urlopen. CI stays offline.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from f1 import openf1
from f1.fastf1_bridge import BRIDGE_COLUMNS

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openf1_car_data_sample.json")


@pytest.fixture(scope="module")
def car_data():
    with open(FIXTURE, encoding="utf-8") as fh:
        return pd.DataFrame(json.load(fh))


def test_contract_columns_present(car_data):
    out = openf1.to_contract(car_data)
    assert list(out.columns) == BRIDGE_COLUMNS


def test_contract_brake_boolean_non_steering(car_data):
    out = openf1.to_contract(car_data)
    # F1 has no steering -> NaN sentinel
    assert out["steering"].isna().all()
    # brake is boolean 0/1
    assert set(out["brake"].unique()) <= {0.0, 1.0}


def test_speed_kmh_unchanged(car_data):
    out = openf1.to_contract(car_data)
    assert out["speed_kmh"].iloc[0] == pytest.approx(298.4)


def test_throttle_normalised_to_unit(car_data):
    out = openf1.to_contract(car_data)
    # 100 % -> 1.0, 55 % -> 0.55
    assert out["throttle"].iloc[0] == pytest.approx(1.0)
    assert out["throttle"].iloc[2] == pytest.approx(0.55, abs=1e-6)


def test_track_position_in_unit_interval(car_data):
    out = openf1.to_contract(car_data)
    assert out["track_position"].min() >= 0.0
    assert out["track_position"].max() <= 1.0
    assert np.all(np.diff(out["track_position"]) >= 0)  # monotonic arc


def test_sector_split_into_thirds(car_data):
    out = openf1.to_contract(car_data)
    assert set(out["sector"].unique()) <= {1, 2, 3}


def test_fetch_car_data_uses_expected_url(monkeypatch):
    captured = {}

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'[{"date":"2024-09-01T13:00:00.000Z","session_key":9437,"driver_number":1,"rpm":1,"speed":10,"n_gear":1,"throttle":100,"brake":0,"x":1,"y":2,"z":3}]'

    def _fake_urlopen(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResp()

    monkeypatch.setattr(openf1.urllib.request, "urlopen", _fake_urlopen)
    df = openf1.fetch_car_data(9437, 1, base_url="http://localhost:1/v1", timeout_s=5)
    assert captured["url"] == "http://localhost:1/v1/car_data?session_key=9437&driver_number=1"
    assert captured["timeout"] == 5
    assert df["driver_number"].iloc[0] == 1
