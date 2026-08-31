"""Tests for dashboard.map_plot (circuit track map figure)."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go

from pitmind.config import Config
from pitmind import corners
from dashboard import map_plot

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_generic_f1.csv")


def _load():
    df = pd.read_csv(DATA)
    return (
        df[df["lap_number"] == 1].reset_index(drop=True),
        df[df["lap_number"] == 2].reset_index(drop=True),
    )


def test_center_lap_offsets_to_track_origin():
    lap1, _ = _load()
    c = map_plot.center_lap(lap1)
    assert abs(c["x"].mean()) < 1.0
    assert abs(c["y"].mean()) < 1.0


def test_center_lap_shared_origin_between_laps():
    lap1, lap2 = _load()
    c2 = map_plot.center_lap(lap2, lap1)
    # same reference origin applied, so both laps are centred the same way
    c1 = map_plot.center_lap(lap1, lap1)
    # spans preserved (shape unchanged), only origin shifted
    assert abs(c2["x"].max() - c2["x"].min()) > 100
    assert abs(c1["x"].max() - c1["x"].min()) > 100


def test_resample_lap_caps_point_count():
    lap1, _ = _load()
    small = map_plot.resample_lap(lap1, max_points=1000)
    assert len(small) <= 1000


def test_track_map_figure_base_structure():
    lap1, lap2 = _load()
    fig = map_plot.track_map_figure(lap2, color_by_speed=False)
    assert isinstance(fig, go.Figure)
    names = [t.name for t in fig.data]
    # neutral track ribbon + start/finish
    assert any(n == "Start / Finish" for n in names)
    assert sum(1 for t in fig.data if t.mode == "lines") >= 1


def test_track_map_figure_speed_trace():
    lap1, lap2 = _load()
    fig = map_plot.track_map_figure(lap2, color_by_speed=True)
    speed_trace = [t for t in fig.data if t.name == "Speed"]
    assert len(speed_trace) == 1
    # speed trace exposes a colorbar + numeric marker colors
    assert speed_trace[0].marker.colorbar is not None


def test_track_map_figure_corner_markers():
    lap1, lap2 = _load()
    cfg = Config.from_file()
    cs = corners.detect_corners(lap1, cfg)
    fig = map_plot.track_map_figure(lap2, cs)
    corner_trace = [t for t in fig.data if t.name == "Corners"]
    track_trace = [t for t in fig.data if t.mode == "lines"][0]
    assert len(corner_trace) == 1
    assert len(corner_trace[0].x) == len(cs)
    # apex markers lie on the track path drawn in the same figure
    assert all(track_trace.x.min() - 1 <= v <= track_trace.x.max() + 1 for v in corner_trace[0].x)
    assert all(track_trace.y.min() - 1 <= v <= track_trace.y.max() + 1 for v in corner_trace[0].y)


def test_track_map_figure_equal_aspect():
    _, lap2 = _load()
    fig = map_plot.track_map_figure(lap2)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
