"""Tests for tools.tune in F1 mode (todo.md M2), parametrized over circuits (M5/C2).

M2 is about making the engine *trust* real-F1-shaped data: bridging a FastF1
session to the contract, then running the pipeline with F1's capability limits
(no steering channel, boolean brake) and sanity-checking corner count + time-loss
ranges offline. C2 extends this across multiple circuits (monza/spa/silverstone/
imola) so a change that fixes one track can't silently regress another.

Each fixture is generated in-session with the vendored synthetic generator + the
real FastF1 bridge, so no CSV and no network is needed offline.
"""

from __future__ import annotations

import pytest

from pitmind import mistakes
from pitmind.config import Config
from tools import tune
from tools.fixture_f1 import generate_f1_lap_set

# circuit id -> the corner count the coarse ~4 Hz F1 sampling resolves to
TRACKS = {
    "monza": (7, 42),
    "spa": (13, 42),
    "silverstone": (9, 42),
    "imola": (11, 42),
}


@pytest.fixture()
def cfg():
    return Config.from_file()


def _f1_capabilities():
    return {"steering": False}


@pytest.mark.parametrize("track", list(TRACKS.keys()))
def test_f1_fixture_is_real_f1_shaped(track):
    df = generate_f1_lap_set(track, laps=8, rng_seed=TRACKS[track][1])
    assert df["steering"].isna().all()
    assert set(df["brake"].unique()) <= {0.0, 1.0}
    assert df["lap_number"].nunique() >= 1


@pytest.mark.parametrize("track", list(TRACKS.keys()))
def test_f1_pipeline_prunes_steering_keeps_f1_mistakes(track, cfg):
    df = generate_f1_lap_set(track, laps=8, rng_seed=TRACKS[track][1])
    bundle = tune.run_pipeline(df, cfg, capabilities=_f1_capabilities())
    types = {m.mistake_type for m in bundle["mistakes"]}
    assert mistakes.MistakeType.EXCESS_STEERING not in types
    # the design.md M2 F1-relevant classes must be represented on every circuit
    f1_expected = {
        mistakes.MistakeType.EARLY_BRAKE,
        mistakes.MistakeType.LATE_BRAKE,
        mistakes.MistakeType.LOW_APEX_SPEED,
        mistakes.MistakeType.SLOW_EXIT,
    }
    assert f1_expected <= types


@pytest.mark.parametrize("track", list(TRACKS.keys()))
def test_f1_pipeline_corner_count_matches_track(track, cfg):
    df = generate_f1_lap_set(track, laps=8, rng_seed=TRACKS[track][1])
    bundle = tune.run_pipeline(df, cfg, capabilities=_f1_capabilities())
    expected = TRACKS[track][0]
    assert abs(len(bundle["corners"]) - expected) <= 1


@pytest.mark.parametrize("track", list(TRACKS.keys()))
def test_f1_sanity_check_passes_on_every_track(track, cfg):
    df = generate_f1_lap_set(track, laps=8, rng_seed=TRACKS[track][1])
    bundle = tune.run_pipeline(df, cfg, capabilities=_f1_capabilities())
    result = tune.sanity_check_f1(bundle, expected_corners=TRACKS[track][0])
    assert all(result["checks"].values())


def test_f1_sanity_check_flags_wrong_corner_count(cfg):
    df = generate_f1_lap_set("monza", laps=8, rng_seed=42)
    bundle = tune.run_pipeline(df, cfg, capabilities=_f1_capabilities())
    result = tune.sanity_check_f1(bundle, expected_corners=99, corner_tol=1)
    assert result["checks"]["corner_count"] is False


def test_f1_sanity_check_flags_steering_if_not_pruned(cfg):
    df = generate_f1_lap_set("monza", laps=8, rng_seed=42)
    bundle = tune.run_pipeline(df, cfg)  # no capabilities -> steering assumed present
    result = tune.sanity_check_f1(bundle, expected_corners=7)
    assert result["checks"]["steering_pruned"] is False


def test_f1_cli_mode_runs_end_to_end(tmp_path, cfg):
    df = generate_f1_lap_set("monza", laps=8, rng_seed=42)
    path = tmp_path / "monza.csv"
    df.to_csv(path, index=False)
    assert tune.main([str(path), "--f1"]) == 0
