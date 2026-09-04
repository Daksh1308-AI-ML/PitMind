"""Tests for tools.tune in F1 mode (todo.md M2).

M2 is about making the engine *trust* real-F1-shaped data: bridging a FastF1
session to the contract, then running the pipeline with F1's capability limits
(no steering channel, boolean brake) and sanity-checking corner count + time-loss
ranges offline on a synthetic Monza lap set.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from pitmind.config import Config
from pitmind import mistakes
from tools import tune

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "f1_monza_laps.csv")


@pytest.fixture(scope="module")
def f1_df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def cfg():
    return Config.from_file()


def _all_capabilities():
    # repeat the capability set used by --f1
    return {"steering": False}


def test_f1_fixture_is_real_f1_shaped(f1_df):
    # steering absent/NaN + boolean brake = the F1 contract reality
    assert f1_df["steering"].isna().all()
    assert set(f1_df["brake"].unique()) <= {0.0, 1.0}
    assert f1_df["lap_number"].nunique() >= 1


def test_f1_pipeline_prunes_steering_keeps_f1_mistakes(f1_df, cfg):
    bundle = tune.run_pipeline(f1_df, cfg, capabilities=_all_capabilities())
    types = {m.mistake_type for m in bundle["mistakes"]}
    # F1 has no steering -> EXCESS_STEERING must never fire
    assert mistakes.MistakeType.EXCESS_STEERING not in types
    # ...but the F1-relevant classes still do (design.md M2)
    f1_expected = {
        mistakes.MistakeType.EARLY_BRAKE,
        mistakes.MistakeType.LATE_BRAKE,
        mistakes.MistakeType.LOW_APEX_SPEED,
        mistakes.MistakeType.LATE_THROTTLE,
        mistakes.MistakeType.SLOW_EXIT,
    }
    assert f1_expected & types == f1_expected


def test_f1_pipeline_corner_count_monza(f1_df, cfg):
    bundle = tune.run_pipeline(f1_df, cfg, capabilities=_all_capabilities())
    # Monza resolves to ~7 corners even at F1's coarse ~4 Hz sampling
    assert abs(len(bundle["corners"]) - 7) <= 1


def test_f1_sanity_check_passes_on_monza(f1_df, cfg):
    bundle = tune.run_pipeline(f1_df, cfg, capabilities=_all_capabilities())
    result = tune.sanity_check_f1(bundle, expected_corners=7)
    assert result["checks"]["corner_count"] is True
    assert result["checks"]["steering_pruned"] is True
    assert result["checks"]["time_loss_range"] is True
    assert all(result["checks"].values())


def test_f1_sanity_check_flags_wrong_corner_count(f1_df, cfg):
    bundle = tune.run_pipeline(f1_df, cfg, capabilities=_all_capabilities())
    # ask for a wildly wrong corner count -> the check must fail (catch regressions)
    result = tune.sanity_check_f1(bundle, expected_corners=99, corner_tol=1)
    assert result["checks"]["corner_count"] is False


def test_f1_sanity_check_flags_steering_if_not_pruned(f1_df, cfg):
    # running WITHOUT the F1 capability is a mistake for F1 data: steering is
    # all-NaN. sanity_check must still report steering_pruned False.
    bundle = tune.run_pipeline(f1_df, cfg)  # no capabilities -> steering assumed present
    result = tune.sanity_check_f1(bundle, expected_corners=7)
    assert result["checks"]["steering_pruned"] is False


def test_f1_cli_mode_runs_end_to_end(tmp_path, f1_df, cfg):
    # exercise the CLI path (no --write, so nothing touches config.yaml)
    rc = tune.main([DATA, "--f1"])
    assert rc == 0
