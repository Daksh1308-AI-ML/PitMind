"""Tests for tools.tune (pipeline validation + threshold recommendations)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from pitmind.config import Config
from tools import tune

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_generic_f1.csv")


@pytest.fixture()
def session_df():
    return pd.read_csv(DATA)


@pytest.fixture()
def cfg(tmp_path):
    # use a throwaway config so --write never touches the project config
    from shutil import copyfile
    tmp = tmp_path / "config.yaml"
    copyfile(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), tmp)
    return Config.from_file(tmp)


def test_run_pipeline_produces_bundle(session_df, cfg):
    bundle = tune.run_pipeline(session_df, cfg)
    assert bundle["n_laps"] >= 1
    assert len(bundle["corners"]) >= 1
    assert "mistakes" in bundle
    assert "summary" in bundle
    assert "time_losses" in bundle
    assert len(bundle["lap_times"]) == bundle["n_laps"]


def test_pipeline_total_time_loss_non_negative(session_df, cfg):
    bundle = tune.run_pipeline(session_df, cfg)
    assert bundle["total_time_loss_s"] >= 0.0


def test_suggest_thresholds_shape(session_df, cfg):
    bundle = tune.run_pipeline(session_df, cfg)
    sugg = tune.suggest_thresholds(bundle)
    # every metric in the map either produced a recommendation or was skipped
    for _col, info in sugg.items():
        for key in ("current", "recommended", "flag_rate", "p50", "p85"):
            assert key in info
        assert info["flag_rate"] is None or 0.0 <= info["flag_rate"] <= 1.0


def test_print_report_runs(session_df, cfg, capsys):
    bundle = tune.run_pipeline(session_df, cfg)
    tune.print_report(bundle)
    out = capsys.readouterr().out
    assert "Validation report" in out or "PitMind validation report" in out
    assert "Threshold diagnostics" in out


def test_apply_thresholds_writes_to_file(tmp_path, session_df, cfg):
    bundle = tune.run_pipeline(session_df, cfg)
    sugg = tune.suggest_thresholds(bundle)
    recs = {info["path"]: info["recommended"] for info in sugg.values()}
    target = tmp_path / "out.yaml"
    from shutil import copyfile
    copyfile(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), target)
    tune.apply_thresholds(recs, config_path=target)
    # file should still parse as YAML and keep the dict threshold intact
    import yaml
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(raw["ranges"]["brake_point_delta_m"], dict)
