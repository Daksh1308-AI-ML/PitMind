"""Smoke tests for project setup + synthetic telemetry generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitmind.config import Config, REQUIRED_TELEMETRY_COLUMNS
from synthetic import generator as gen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def session() -> pd.DataFrame:
    return gen.generate_session(laps=4)[0]


class TestConfig:
    def test_defaults_load(self):
        cfg = Config.from_file()
        assert cfg.detection.sample_rate_hz == 60
        assert cfg.ranges.brake_point_delta_m["potential"] == 10.0
        assert cfg.timeloss_mode == "kinematic"

    def test_required_columns_defined(self):
        assert "track_position" in REQUIRED_TELEMETRY_COLUMNS
        assert len(REQUIRED_TELEMETRY_COLUMNS) >= 12


class TestSyntheticSchema:
    def test_columns(self, session):
        for col in REQUIRED_TELEMETRY_COLUMNS:
            assert col in session.columns, f"missing column {col}"

    def test_lap_count(self, session):
        assert session["lap_number"].nunique() == 4

    def test_track_position_in_unit_range(self, session):
        tp = session["track_position"]
        assert tp.min() >= 0.0 and tp.max() <= 1.0

    def test_lap_starts_near_line(self, session):
        lap1 = session[session.lap_number == 1]
        assert lap1["track_position"].iloc[0] == pytest.approx(0.0, abs=0.01)


class TestSyntheticPhysics:
    def test_braking_zones_exist(self, session):
        any_brake = (session["brake"] > 0.3).any()
        assert any_brake

    def test_speed_bounds(self, session):
        assert session["speed_kmh"].min() > 40
        # V_MAX is 310 but acceleration on straights can briefly exceed it
        assert session["speed_kmh"].max() <= 430

    def test_clean_lap_is_fastest(self, session):
        # clean lap (no injected mistakes) should be fastest of the session
        per_lap_time = session.groupby("lap_number")["timestamp"].agg(lambda x: x.max() - x.min())
        clean = [1, 2]  # generator marks first two laps clean
        dirty = [l for l in per_lap_time.index if l not in clean]
        assert per_lap_time.loc[clean].min() < per_lap_time.loc[dirty].min()

    def test_angles_sum_to_closed_loop(self):
        track_data = gen.build_track()
        corners = track_data.corners
        L = track_data.length_m
        assert L > 4000  # Monza is ~5786m, Spa ~6977m
        total = abs(sum(c.angle_rad for c in corners))
        # Real circuits from GeoJSON don't sum exactly to 2π; allow ~0.15 rad (~8.5°) tolerance
        assert total == pytest.approx(2 * np.pi, abs=0.15)


class TestGeneratedFiles:
    def test_seed_csv_and_ground_truth_written(self):
        df = pd.read_csv(DATA_DIR / "synthetic_generic_f1.csv")
        gt = json.loads((DATA_DIR / "synthetic_generic_f1_ground_truth.json").read_text(encoding="utf-8"))
        assert df["lap_number"].nunique() == 12
        assert len(gt["laps"]) == 12
        # Track kind is now configurable (default Monza = 8 corners)
        assert gt["track"]["corners"] >= 4