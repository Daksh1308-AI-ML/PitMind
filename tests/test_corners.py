"""Tests for automatic corner detection + per-corner event extraction."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pitmind import corners, events, segmentation
from pitmind.config import Config
from synthetic import generator as gen

A_LAT = gen.A_LAT


@pytest.fixture(scope="module")
def data():
    session, gt = gen.generate_session(laps=6)
    return session, gt


@pytest.fixture(scope="module")
def detected(data):
    session, gt = data
    cfg = Config.from_file()
    lap1 = session[session.lap_number == 1].reset_index(drop=True)
    return corners.detect_corners(lap1, cfg), corners.track_length_m(lap1), cfg


class TestCornerDetection:
    def test_expected_number(self, detected):
        corners_found, L, _ = detected
        assert len(corners_found) == 6

    def test_ordered_along_lap(self, detected):
        cs, _, _ = detected
        starts = [c.start_tp for c in cs]
        assert starts == sorted(starts)

    def test_position_matches_ground_truth(self, data, detected):
        session, gt = data
        cs, L, _ = detected
        gt_corners = gt["laps"][0]["corners"]
        assert len(gt_corners) == len(cs)
        for det, g in zip(cs, gt_corners):
            assert abs(det.start_tp * L - g["start_s"]) < 250

    def test_total_turning_circuit(self, detected):
        cs, _, _ = detected
        total = sum(c.angle_deg for c in cs)
        assert total == pytest.approx(360.0, abs=15.0)

    def test_clean_lap_apex_speeds_near_limits(self, data, detected):
        session, _ = data
        cs, L, cfg = detected
        lap1 = session[session.lap_number == 1].reset_index(drop=True)
        ev = events.extract_corner(lap1, cs[0], L, cfg)  # sanity: first corner
        limit = math.sqrt(A_LAT * cs[0].radius_m) * 3.6
        assert ev.apex_speed_kmh == pytest.approx(limit, rel=0.15)
        # check limits for all corners against known radii
        for det in cs:
            limit = math.sqrt(A_LAT * det.radius_m) * 3.6
            assert limit > 70  # plausible corner speed


class TestEventExtraction:
    def test_table_shape(self, data, detected):
        session, _ = data
        cs, L, cfg = detected
        laps = segmentation.valid_laps(session)
        table = events.corner_features_table(laps, cs, L, cfg)
        assert len(table) == 6 * 6  # corner x lap

    def test_brake_point_positive(self, data, detected):
        session, _ = data
        cs, L, cfg = detected
        table = events.corner_features_table(segmentation.valid_laps(session), cs, L, cfg)
        healthy = table.dropna(subset=["brake_point_m"])
        assert (healthy["brake_point_m"] > 50).all()
        assert (healthy["brake_point_m"] < 900).all()

    def test_mistakes_are_detected_by_metrics(self, data, detected):
        session, gt = data
        cs, L, cfg = detected
        laps = segmentation.valid_laps(session)
        table = events.corner_features_table(laps, cs, L, cfg)

        # reference = clean lap 1
        refs = table[table.lap == 1].set_index("corner")

        early_ok = low_apex_ok = late_thr_ok = 0
        early_n = low_apex_n = late_thr_n = 0
        for gl in gt["laps"]:
            for gc in gl["corners"]:
                ci = int(gc["corner"][1:]) - 1
                row = table[(table.lap == gl["lap"]) & (table.corner == ci)]
                assert len(row) == 1
                row = row.iloc[0]
                if "brake_shift_m" in gc and gc["brake_shift_m"] > 0:
                    early_n += 1
                    if row["brake_point_m"] > refs.loc[ci, "brake_point_m"] + gc["brake_shift_m"] * 0.4:
                        early_ok += 1
                if "apex_deficit_kmh" in gc:
                    low_apex_n += 1
                    if row["apex_speed_kmh"] < refs.loc[ci, "apex_speed_kmh"] - gc["apex_deficit_kmh"] * 0.4:
                        low_apex_ok += 1
                if "throttle_delay_s" in gc:
                    late_thr_n += 1
                    if row["throttle_on_s"] > refs.loc[ci, "throttle_on_s"] + gc["throttle_delay_s"] * 0.4:
                        late_thr_ok += 1
        assert early_n > 0 and early_ok >= early_n * 0.8
        assert low_apex_n > 0 and low_apex_ok >= low_apex_n * 0.8
        assert late_thr_n > 0 and late_thr_ok >= late_thr_n * 0.8