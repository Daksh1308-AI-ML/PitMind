"""Tests for f1.live — the live race-engineer loop (todo.md M4).

No fastf1 / network: the source is injected as a callable over offline F1
fixtures, exercising the same poll -> analyse -> emit contract the real
FastF1/OpenF1 source uses.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from pitmind.config import Config
from f1 import live

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "f1_monza_laps.csv")


@pytest.fixture(scope="module")
def f1_df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def cfg():
    return Config.from_file()


def test_analyze_session_f1_capabilities(f1_df, cfg):
    # the F1 slice goes through the pipeline with steering pruned
    lap = live.analyze_session(f1_df, cfg)
    assert lap.lap_number == 1
    assert lap.total_time_loss_s >= 0
    assert lap.n_directives >= 0
    assert isinstance(lap.summary, str) and "issues" in lap.summary


def test_analyze_session_single_valid_lap_has_zero_delta_loss(f1_df, cfg):
    # a lone clean lap has no reference to diff against -> no time loss, no failure
    sub = f1_df[f1_df["lap_number"] == 5].copy()
    lap = live.analyze_session(sub, cfg)
    assert lap.lap_number == 5
    assert lap.total_time_loss_s == 0
    assert lap.n_mistakes == 0


def test_engineer_loop_emits_each_new_slice(f1_df, cfg):
    # two whole laps as a streaming source -> exactly two emits (lap 5 then 6)
    slices = [f1_df[f1_df["lap_number"] == n].copy() for n in (5, 6)]
    state = {"i": 0}

    def source():
        if state["i"] < len(slices):
            s = slices[state["i"]]
            state["i"] += 1
            return s
        return None

    emitted = []
    live.engineer_loop(source, emitted.append, cfg, max_ticks=4)
    assert [e.lap_number for e in emitted] == [5, 6]


def test_engineer_loop_skips_no_new_data(f1_df, cfg):
    # a dead source (always None) must emit nothing and terminate cleanly
    emitted = []

    def source():
        return None

    live.engineer_loop(source, emitted.append, cfg, max_ticks=3)
    assert emitted == []


def test_engineer_loop_handles_failed_slice(f1_df, cfg):
    # a corrupt slice must emit an error lap, not crash the loop
    bad = f1_df.copy()
    bad["lap_number"] = None
    calls = {"n": 0}

    def source():
        calls["n"] += 1
        return bad if calls["n"] == 1 else None

    emitted = []
    live.engineer_loop(source, emitted.append, cfg, max_ticks=3)
    assert len(emitted) == 1
    assert emitted[0].n_directives == 0
    assert "failed" in emitted[0].summary.lower()


def test_priority_severity_map():
    assert live._severity(1) == "HIGH"
    assert live._severity(2) == "MED"
    assert live._severity(3) == "LOW"


def test_f1_capabilities_default_steering_off():
    assert live.F1_CAPABILITIES == {"steering": False}
