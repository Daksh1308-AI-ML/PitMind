"""Tests for preprocess + segmentation on synthetic telemetry."""

from __future__ import annotations

import pandas as pd
import pytest

from pitmind import preprocess, segmentation
from pitmind.config import Config


@pytest.fixture(scope="module")
def session() -> pd.DataFrame:
    from synthetic import generator as gen
    return gen.generate_session(laps=6)[0]


class TestPreprocess:
    def test_resample_rates_conserved(self, session):
        cfg = Config.from_file()
        out = preprocess.preprocess(session, cfg)
        # per-lap sample intervals ~ 1/60 s
        dts = out.groupby("lap_number")["timestamp"].diff().dropna()
        assert dts.median() == pytest.approx(1 / 60.0, abs=0.01)

    def test_required_columns_survive(self, session):
        out = preprocess.preprocess(session, Config.from_file())
        assert set(["timestamp", "track_position", "speed_kmh"]).issubset(out.columns)

    def test_missing_track_position_raises(self, session):
        bad = session.drop(columns=["track_position"])
        with pytest.raises(ValueError):
            preprocess.preprocess(bad, Config.from_file())


class TestSegmentation:
    def test_split_finds_expected_lap_count(self, session):
        laps = segmentation.valid_laps(session)
        assert len(laps) == 6

    def test_each_valid_lap_covers_full_circle(self, session):
        for lap in segmentation.valid_laps(session):
            assert (lap["track_position"].max() - lap["track_position"].min()) >= segmentation.MIN_COVERAGE

    def test_drops_truncated_lap(self):
        # an out-lap chunk covering only part of the circle must be rejected
        from synthetic import generator as gen
        session = gen.generate_session(laps=3)[0]
        full = session[session.lap_number == 1]
        truncated = full[full["track_position"] <= 0.6].reset_index(drop=True)
        assert segmentation.is_valid_lap(truncated) is False
        assert len(segmentation.valid_laps(truncated)) == 0

    def test_lap_table_shape(self, session):
        table = segmentation.valid_lap_table(session)
        assert list(table.columns) == [
            "lap", "duration_s", "n_samples", "min_speed_kmh", "max_speed_kmh", "avg_speed_kmh",
        ]
        assert len(table) == 6

    def test_duration_plausible(self, session):
        table = segmentation.valid_lap_table(session)
        # Monza laps are ~88-89s; allow 70-120s for different tracks
        assert table["duration_s"].between(70, 120).all()
