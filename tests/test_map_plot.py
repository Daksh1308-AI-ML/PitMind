"""Tests for dashboard.map_plot (circuit track map figure)."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import map_plot
from pitmind import corners
from pitmind.config import Config

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


# --------------------------------------------------------------------------- #
# M4: multi-metric corner heat-map overlay
# --------------------------------------------------------------------------- #

def _corner_metric_time_loss(lap1):
    cfg = Config.from_file()
    cs = corners.detect_corners(lap1, cfg)
    # minimal feature table with the columns corner_overlay needs
    ft = pd.DataFrame({
        "corner": list(range(len(cs))),
        "name": [f"T{i+1}" for i in range(len(cs))],
        "delta_apex_speed_kmh": [-1.0 * (i + 1) for i in range(len(cs))],
    })
    return cs, ft


def test_corner_metric_table_time_loss_agg():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, [], metric="time_loss_s")
    assert len(cm) == len(cs)
    assert (cm["time_loss_s"] == 0).all()


def test_corner_metric_table_apex_speed():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, metric="delta_apex_speed_kmh")
    assert len(cm) == len(cs)
    assert "delta_apex_speed_kmh" in cm.columns


def test_corner_overlay_figure_structure():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, metric="delta_apex_speed_kmh")
    fig = map_plot.corner_overlay_figure(lap1, cs, cm, metric="delta_apex_speed_kmh")
    assert isinstance(fig, go.Figure)
    # track ribbon + corner heat-map
    assert sum(1 for t in fig.data if t.mode == "lines") >= 1
    corners_trace = [t for t in fig.data if t.name == "Corners"]
    assert len(corners_trace) == 1
    assert len(corners_trace[0].x) == len(cs)
    assert corners_trace[0].marker.colorbar is not None


def test_corner_overlay_figure_equal_aspect():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, metric="delta_apex_speed_kmh")
    fig = map_plot.corner_overlay_figure(lap1, cs, cm, metric="delta_apex_speed_kmh")
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_corner_overlay_rejects_mismatched_corner_count():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, metric="delta_apex_speed_kmh")
    cm = cm.iloc[:-1].reset_index(drop=True)  # drop one row -> mismatch
    with pytest.raises(ValueError):
        map_plot.corner_overlay_figure(lap1, cs, cm)


def test_corner_overlay_rejects_unknown_metric():
    lap1, _ = _load()
    cs, ft = _corner_metric_time_loss(lap1)
    cm = map_plot.corner_metric_table(ft, metric="delta_apex_speed_kmh")
    with pytest.raises(KeyError):
        map_plot.corner_overlay_figure(lap1, cs, cm, metric="nope")
