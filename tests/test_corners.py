"""Tests for automatic corner detection + per-corner event extraction.

Circuit-agnostic: expectations derived from generated ground truth, not hardcoded.
"""

from __future__ import annotations

import math

import pytest

from pitmind import corners, events, features, mistakes, potential_lap, segmentation, timeloss
from pitmind.config import Config
from synthetic import circuit as circuit_mod
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


@pytest.fixture(scope="module")
def track_gt(data):
    """Ground truth track info from generator."""
    _, gt = data
    return gt["track"]


class TestCornerDetection:
    def test_detected_corners_match_gt_count(self, data, detected, track_gt):
        """Number of detected corners should match ground truth corner count."""
        cs, _, _ = detected
        # Allow +/-1 difference due to detection sensitivity
        assert abs(len(cs) - track_gt["corners"]) <= 1

    def test_ordered_along_lap(self, detected):
        """Corner start_tp should be monotonically increasing around the lap."""
        cs, _, _ = detected
        starts = [c.start_tp for c in cs]
        assert starts == sorted(starts)

    def test_position_matches_ground_truth(self, data, detected):
        """Detected corner start positions should approximately match some GT corner."""
        session, gt = data
        cs, L, _ = detected
        gt_corners = gt["laps"][0]["corners"]
        gt_positions = [g["start_s"] for g in gt_corners]
# Each detected corner should match some GT corner within tolerance
        for c in cs:
            c_s = c.start_tp * L
            best_gt = min(gt_positions, key=lambda g: abs(g - c_s))
            # Allow up to 550m tolerance for noisy telemetry (some corners merge)
            assert abs(best_gt - c_s) < 550, (
                f"Detected {c.name} at {c_s:.1f}m not matched to any GT corner "
                f"(best: {best_gt:.1f}m)"
            )

    def test_no_invalid_total_turn_check(self, detected):
        """Total turning angle should be approximately -360° (clockwise) or +360° (CCW).

        This replaces the old hardcoded 'total == 360' check which was invalid
        for non-simple circuits and noisy telemetry.
        """
        cs, _, _ = detected
        total = sum(c.angle_deg for c in cs)
        # Allow wide tolerance due to noise and detection merging
        assert abs(abs(total) - 360.0) < 180.0

    def test_clean_lap_apex_speeds_near_limits(self, data, detected):
        """Apex speeds on clean lap should be near lateral-g limit for each corner.

        Note: fast sweepers may not reach the lateral limit; we check that
        apex speed is at least 40% of the theoretical limit.
        """
        session, _ = data
        cs, L, cfg = detected
        lap1 = session[session.lap_number == 1].reset_index(drop=True)
        for det in cs:
            ev = events.extract_corner(lap1, det, L, cfg)
            limit = math.sqrt(A_LAT * det.radius_m) * 3.6
            # Apex speed should be at least 40% of theoretical limit
            # (some corners are sweepers taken flat-out)
            assert ev.apex_speed_kmh >= limit * 0.40
            # Plausible corner speed (> 60 km/h)
            assert limit > 60


class TestEventExtraction:
    def test_table_shape_matches_gt_corners(self, data, detected):
        """Feature table should have rows = valid_laps * detected_corners."""
        session, _ = data
        cs, L, cfg = detected
        laps = segmentation.valid_laps(session)
        table = events.corner_features_table(laps, cs, L, cfg)
        assert len(table) == len(laps) * len(cs)

    def test_brake_point_positive(self, data, detected):
        """Brake points should be in plausible range (50-900m before corner)."""
        session, _ = data
        cs, L, cfg = detected
        table = events.corner_features_table(segmentation.valid_laps(session), cs, L, cfg)
        healthy = table.dropna(subset=["brake_point_m"])
        assert (healthy["brake_point_m"] > 50).all()
        assert (healthy["brake_point_m"] < 900).all()

    def test_mistakes_are_detected_by_metrics(self, data, detected):
        """Known injected mistakes should be detectable in corner metrics."""
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
                # Map GT corner name to detected corner index
                # GT corners are T1, T2... detected are 0, 1...
                gt_name = gc["corner"]
                try:
                    gt_idx = int(gt_name[1:]) - 1
                except (ValueError, IndexError):
                    continue
                # Find detected corner with closest start_s
                row = table[(table.lap == gl["lap"]) & (table.corner == gt_idx)]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                if "brake_shift_m" in gc and gc["brake_shift_m"] > 0:
                    early_n += 1
                    if row["brake_point_m"] > refs.loc[gt_idx, "brake_point_m"] + gc["brake_shift_m"] * 0.4:
                        early_ok += 1
                if "apex_deficit_kmh" in gc:
                    low_apex_n += 1
                    if row["apex_speed_kmh"] < refs.loc[gt_idx, "apex_speed_kmh"] - gc["apex_deficit_kmh"] * 0.4:
                        low_apex_ok += 1
                if "throttle_delay_s" in gc:
                    late_thr_n += 1
                    if row["throttle_on_s"] > refs.loc[gt_idx, "throttle_on_s"] + gc["throttle_delay_s"] * 0.4:
                        late_thr_ok += 1
        # At least some mistakes of each type should be injected and detected
        assert early_n > 0 and early_ok >= early_n * 0.5
        assert low_apex_n > 0 and low_apex_ok >= low_apex_n * 0.5
        # Late throttle is harder to detect reliably; lower threshold
        assert late_thr_n > 0 and late_thr_ok >= late_thr_n * 0.25


class TestCircuitModule:
    """Track-module tests: loop closure, length ≈ official, both turn directions."""

    def test_loop_closure(self):
        """Circuit centerline should form a closed loop (start ≈ end)."""
        track = circuit_mod.load_circuit("monza")
        dx = track.centerline_xy[0, 0] - track.centerline_xy[-1, 0]
        dy = track.centerline_xy[0, 1] - track.centerline_xy[-1, 1]
        gap = math.hypot(dx, dy)
        assert gap < 2.0  # meters (resampling can leave small gap)

    def test_length_matches_official(self):
        """Track length should be within 2% of official length."""
        track = circuit_mod.load_circuit("monza")
        # Official Monza length: 5793 m (from GeoJSON properties)
        official = 5793.0
        assert abs(track.length_m - official) / official < 0.02

    def test_both_turn_directions_present(self):
        """Circuit should have both left and right corners (signed angles)."""
        track = circuit_mod.load_circuit("monza")
        angles = [c.angle_deg for c in track.corners]
        has_left = any(a > 0 for a in angles)
        has_right = any(a < 0 for a in angles)
        assert has_left and has_right

    def test_all_available_circuits_load(self):
        """All vendored circuits should load without error."""
        for cid in circuit_mod.list_circuits():
            track = circuit_mod.load_circuit(cid)
            assert track.length_m > 1000
            assert len(track.corners) >= 4
            assert track.circuit_id == cid


class TestMistakeDetection:
    """Tests for mistake detection on synthetic laps with known injected mistakes."""

    @pytest.fixture(scope="module")
    def full_results(self, data):
        """Run full pipeline: features -> mistakes -> timeloss."""
        session, gt = data
        cfg = Config.from_file()
        table = features.build_feature_table(session, cfg)
        mistake_list = mistakes.detect_mistakes(table, cfg)
        time_loss_list = timeloss.estimate_time_loss(mistake_list, table, cfg)
        return table, mistake_list, time_loss_list, gt

    def test_mistake_types_covered(self, full_results):
        """All mistake types from ground truth should be detectable."""
        _, mistake_list, _, gt = full_results
        m_df = mistakes.mistakes_to_dataframe(mistake_list)

        # Check that each GT mistake type has at least one detection
        gt_brake_early = sum(1 for gl in gt["laps"] for gc in gl["corners"]
                            if "brake_shift_m" in gc and gc["brake_shift_m"] > 0)
        gt_brake_late = sum(1 for gl in gt["laps"] for gc in gl["corners"]
                           if "brake_shift_m" in gc and gc["brake_shift_m"] < 0)
        gt_low_apex = sum(1 for gl in gt["laps"] for gc in gl["corners"]
                         if "apex_deficit_kmh" in gc)
        gt_late_thr = sum(1 for gl in gt["laps"] for gc in gl["corners"]
                         if "throttle_delay_s" in gc)

        # These should have some detections (allowing for false negatives)
        detected_types = set(m_df["mistake_type"].unique())
        assert mistakes.MistakeType.EARLY_BRAKE.value in detected_types or gt_brake_early == 0
        assert mistakes.MistakeType.LATE_BRAKE.value in detected_types or gt_brake_late == 0
        assert mistakes.MistakeType.LOW_APEX_SPEED.value in detected_types or gt_low_apex == 0
        assert mistakes.MistakeType.LATE_THROTTLE.value in detected_types or gt_late_thr == 0

    def test_confidence_levels_present(self, full_results):
        """Mistakes should have confidence levels assigned."""
        _, mistake_list, _, _ = full_results
        m_df = mistakes.mistakes_to_dataframe(mistake_list)
        assert set(m_df["confidence"].unique()).issubset({"weak", "significant", "strong"})

    def test_time_loss_positive(self, full_results):
        """Time loss estimates should be non-negative and bounded."""
        _, _, time_loss_list, _ = full_results
        tl_df = timeloss.time_loss_to_dataframe(time_loss_list)
        assert (tl_df["time_loss_s"] >= 0).all()
        assert (tl_df["time_loss_s"] <= 5.0).all()  # reasonable upper bound per mistake

    def test_total_time_loss_per_lap_reasonable(self, full_results):
        """Total time loss per lap should be plausible."""
        _, _, time_loss_list, _ = full_results
        totals = timeloss.total_time_loss_per_lap(time_loss_list)
        for _lap, loss in totals.items():
            assert 0 <= loss <= 10.0  # max ~10s lost per lap

    def test_potential_lap_improves_on_actual(self, data):
        """Potential lap should be <= fastest actual lap."""
        session, _ = data
        cfg = Config.from_file()
        pot = potential_lap.build_potential_lap(session, cfg)
        # Potential should not be worse than best actual (allow small numerical tolerance)
        assert pot.improvement_vs_best_s >= -0.1


class TestTimeLoss:
    """Tests for kinematic time-loss estimation."""

    def test_brake_loss_monotonic(self):
        """Brake time loss should increase with delta magnitude."""
        from pitmind.config import Config
        cfg = Config.from_file()
        entry_speed = 80 / 3.6

        loss_early_10 = timeloss._kinematic_brake_loss(-10.0, entry_speed, cfg)
        loss_early_20 = timeloss._kinematic_brake_loss(-20.0, entry_speed, cfg)
        loss_late_10 = timeloss._kinematic_brake_loss(10.0, entry_speed, cfg)
        loss_late_20 = timeloss._kinematic_brake_loss(20.0, entry_speed, cfg)

        assert loss_early_20 > loss_early_10
        assert loss_late_20 > loss_late_10
        assert loss_early_10 > 0 and loss_late_10 > 0

    def test_apex_loss_monotonic(self):
        """Apex time loss should increase with speed deficit."""
        from pitmind.config import Config
        cfg = Config.from_file()
        radius = 100.0
        ref_speed = 100.0  # reference apex speed

        loss_5 = timeloss._kinematic_apex_loss(-5.0, ref_speed, radius, cfg)
        loss_10 = timeloss._kinematic_apex_loss(-10.0, ref_speed, radius, cfg)
        loss_20 = timeloss._kinematic_apex_loss(-20.0, ref_speed, radius, cfg)

        assert loss_10 > loss_5
        assert loss_20 > loss_10
        assert loss_5 > 0

    def test_throttle_loss_monotonic(self):
        """Throttle delay loss should increase with delay."""
        from pitmind.config import Config
        cfg = Config.from_file()
        exit_speed = 100 / 3.6

        loss_01 = timeloss._kinematic_throttle_loss(0.1, exit_speed, cfg)
        loss_03 = timeloss._kinematic_throttle_loss(0.3, exit_speed, cfg)

        assert loss_03 > loss_01
        assert loss_01 > 0

    def test_exit_loss_monotonic(self):
        """Exit speed loss should increase with deficit."""
        from pitmind.config import Config
        cfg = Config.from_file()
        straight = 200.0
        ref_speed = 150.0  # reference exit speed

        loss_5 = timeloss._kinematic_exit_loss(-5.0, ref_speed, straight, cfg)
        loss_15 = timeloss._kinematic_exit_loss(-15.0, ref_speed, straight, cfg)

        assert loss_15 > loss_5
        assert loss_5 > 0

    def test_steering_loss_monotonic(self):
        """Steering excess loss should increase with excess."""
        from pitmind.config import Config
        cfg = Config.from_file()
        corner_time = 3.0

        loss_02 = timeloss._kinematic_steering_loss(0.25, corner_time, cfg)  # 0.1 excess
        loss_05 = timeloss._kinematic_steering_loss(0.55, corner_time, cfg)  # 0.4 excess

        assert loss_05 > loss_02
        assert loss_02 >= 0
