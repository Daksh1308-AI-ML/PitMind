"""Circuit track map helper for the PitMind dashboard.

Draws the actual circuit layout from a lap's (x, y) world coordinates: a
heat-map ribbon colored by speed, with detected-corner apex markers and a
start/finish line. No GeoJSON is used — the shape comes from the driven path,
so it works for both synthetic and real recorded laps (architect.md rule 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

LINE_COLOR = "steelblue"
CORNER_COLOR = "crimson"
FINISH_COLOR = "limegreen"


def center_lap(lap: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a copy of ``lap`` with x/y re-centred to meters from track centre.

    Offsets are the mean of ``ref`` (defaults to ``lap`` itself), so multiple
    laps share one origin. Raw ACC/GeoJSON coords are UTM-style (~1e6) and make
    ugly axis ticks; re-centring reads in small meter numbers without changing
    the lap's shape.
    """
    base = lap if ref is None else ref
    mx = float(base["x"].mean())
    my = float(base["y"].mean())
    out = lap.copy()
    out["x"] = out["x"] - mx
    out["y"] = out["y"] - my
    return out


def resample_lap(lap: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    """Downsample a lap to at most ``max_points`` rows (uniform by track_position)."""
    if len(lap) <= max_points:
        return lap.reset_index(drop=True)
    tp = lap["track_position"].to_numpy(dtype=float)
    grid = np.linspace(tp.min(), tp.max(), max_points)
    cols: dict[str, np.ndarray] = {}
    for col in lap.columns:
        cols[col] = np.interp(grid, tp, lap[col].to_numpy(dtype=float))
    cols["track_position"] = grid
    return pd.DataFrame(cols)


def track_map_figure(
    lap: pd.DataFrame,
    corners: list | None = None,
    *,
    color_by_speed: bool = True,
    title: str | None = None,
    ref: pd.DataFrame | None = None,
) -> go.Figure:
    """Build a Plotly figure of a lap's circuit layout.

    Args:
        lap: Single-lap telemetry with (x, y, speed_kmh, track_position).
        corners: Optional list of corners.detect_corners CornerRegion, used to
                 draw apex markers (A: marked with T1..Tn).
        color_by_speed: Color the track ribbon by speed_kmh (heat-map).
        title: Optional figure title.
        ref: Reference lap used to centre the map (defaults to ``lap``).
    """
    lap = resample_lap(lap)
    centre_df = center_lap(lap, ref)

    tp = centre_df["track_position"].to_numpy()
    order = np.argsort(tp)
    xs = centre_df["x"].to_numpy()[order]
    ys = centre_df["y"].to_numpy()[order]
    speed = centre_df["speed_kmh"].to_numpy()[order]

    fig = go.Figure()

    # Neutral ribbon underneath
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=LINE_COLOR, width=4),
            hoverinfo="skip",
            name=title or "Track",
        )
    )

    if color_by_speed:
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    color=speed,
                    colorscale="Turbo",
                    size=5,
                    line=dict(width=0),
                    colorbar=dict(title="km/h"),
                ),
                hoverinfo="skip",
                name="Speed",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers",
            marker=dict(color=FINISH_COLOR, size=12, symbol="square",
                        line=dict(color="white", width=1)),
            name="Start / Finish",
        )
    )

    if corners:
        ax: list[float] = []
        ay: list[float] = []
        names: list[str] = []
        for c in corners:
            i = int(np.argmin(np.abs(tp - c.apex_tp)))
            ax.append(float(xs[i]))
            ay.append(float(ys[i]))
            names.append(c.name)
        fig.add_trace(
            go.Scatter(
                x=ax, y=ay, mode="markers+text", text=names,
                textposition="top center", textfont=dict(color=CORNER_COLOR),
                marker=dict(color=CORNER_COLOR, size=10, symbol="circle",
                            line=dict(color="white", width=1)),
                name="Corners",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="meters",
        yaxis_title="meters",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------- #
# Multi-metric corner heat-map overlay (M4)
# --------------------------------------------------------------------------- #

# default metric: column -> (title, higher_is_worse)
CORNER_METRICS: dict[str, tuple[str, bool]] = {
    "time_loss_s": ("Time loss (s)", True),
    "delta_apex_speed_kmh": ("Apex speed delta (km/h)", True),   # more negative = slower
    "delta_brake_point_m": ("Brake point delta (m)", True),       # |delta| = bigger deviation
    "corner_time_s": ("Corner time (s)", True),
}


def corner_metric_table(
    feature_table: pd.DataFrame,
    time_loss: list | None = None,
    metric: str = "time_loss_s",
) -> pd.DataFrame:
    """Build a one-row-per-corner metric frame for `corner_overlay_figure`.

    If ``metric`` is ``time_loss_s``, `time_loss` (a list of timeloss.TimeLoss,
    e.g. from a single lap) is summed per corner. Otherwise the metric is taken
    as a column of ``feature_table`` (meaned across laps, in corner order).

    Returns a DataFrame with columns: ``corner``, ``name``, and ``metric``.
    """
    if metric == "time_loss_s":
        from pitmind import timeloss
        tiles = timeloss.time_loss_to_dataframe(time_loss or [])
        if tiles.empty:
            corners_in = sorted(feature_table["corner"].unique())
            return pd.DataFrame({
                "corner": corners_in,
                "name": [_corner_name(feature_table, c) for c in corners_in],
                "time_loss_s": [0.0] * len(corners_in),
            })
        agg = tiles.groupby("corner")["time_loss_s"].sum().reset_index()
        agg["name"] = agg["corner"].apply(lambda c: _corner_name(feature_table, int(c)))
        return agg[["corner", "name", "time_loss_s"]]

    if metric not in feature_table.columns:
        raise KeyError(f"metric '{metric}' not in feature table")
    corners_in = sorted(feature_table["corner"].unique())
    rows = []
    for c in corners_in:
        seg = feature_table[feature_table["corner"] == c]
        rows.append({
            "corner": c,
            "name": _corner_name(feature_table, c),
            metric: float(seg[metric].mean()),
        })
    return pd.DataFrame(rows)


def _corner_name(feature_table: pd.DataFrame, corner: int) -> str:
    seg = feature_table[feature_table["corner"] == corner]
    if "name" in seg.columns and len(seg):
        return str(seg["name"].iloc[0])
    return f"T{corner + 1}"


def _metric_values(corner_metric: pd.DataFrame, metric: str) -> np.ndarray:
    """Extract + absolutise per-corner metric values for colour/size."""
    if metric not in corner_metric.columns:
        raise KeyError(f"metric '{metric}' not in corner table; have "
                       f"{list(corner_metric.columns)}")
    vals = corner_metric[metric].to_numpy(dtype=float)
    # brake-point delta is signed; "worse" = bigger absolute deviation
    if metric in ("delta_brake_point_m", "delta_apex_speed_kmh", "delta_exit_speed_kmh"):
        return np.abs(np.nan_to_num(vals))
    return np.nan_to_num(vals)


def corner_overlay_figure(
    lap: pd.DataFrame,
    corners: list,
    corner_metric: pd.DataFrame,
    metric: str = "time_loss_s",
    *,
    title: str | None = None,
    ref: pd.DataFrame | None = None,
    size_scale: float = 1.0,
) -> go.Figure:
    """A corner heat-map overlay: each corner is a bubble coloured + sized by a metric.

    Used to visualise where time/performance is lost on the track (M4). The
    base track shape comes from the lap's x/y path (track-agnostic, no GeoJSON);
    the corner markers are coloured by ``corner_metric[metric]``.

    Args:
        lap: Single-lap telemetry (x, y, speed_kmh, track_position).
        corners: Detected CornerRegion list (must align with corner_metric rows by index).
        corner_metric: Feature-table-style rows, one per corner (same order as corners),
                       containing at least ``metric`` and a ``name``/``corner`` column.
        metric: Column of corner_metric to color/size by (see CORNER_METRICS).
        title: Figure title.
        ref: Reference lap for centring (defaults to ``lap``).
        size_scale: Overall bubble-size multiplier.
    """
    if len(corners) != len(corner_metric):
        raise ValueError(f"corners ({len(corners)}) must match corner_metric rows "
                         f"({len(corner_metric)})")

    lap = resample_lap(lap)
    centre_df = center_lap(lap, ref)
    tp = centre_df["track_position"].to_numpy()
    order = np.argsort(tp)
    xs = centre_df["x"].to_numpy()[order]
    ys = centre_df["y"].to_numpy()[order]

    _, higher_is_worse = CORNER_METRICS.get(metric, (metric, True))
    values = _metric_values(corner_metric, metric)
    # normalise bubble size so the largest metric value maps to a clear size delta
    vmax = float(values.max()) if len(values) and values.max() > 0 else 1.0
    sizes = 10 + (values / vmax) * (25 * size_scale)

    ax: list[float] = []
    ay: list[float] = []
    names: list[str] = []
    for c in corners:
        i = int(np.argmin(np.abs(tp - c.apex_tp)))
        ax.append(float(xs[i]))
        ay.append(float(ys[i]))
        nm = corner_metric.get("name")
        name = str(nm.iloc[c.index]) if nm is not None and c.index < len(nm) else c.name
        names.append(name)

    # neutral track ribbon
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=LINE_COLOR, width=3), hoverinfo="skip", name="Track",
    ))

    fig.add_trace(go.Scatter(
        x=ax, y=ay, mode="markers+text", text=names, textposition="top center",
        marker=dict(
            color=values, colorscale="RdYlGn_r" if higher_is_worse else "RdYlGn",
            size=sizes,
            colorbar=dict(title=CORNER_METRICS.get(metric, (metric, True))[0]),
            line=dict(color="white", width=1),
            showscale=True,
        ),
        name="Corners",
        customdata=values,
        hovertemplate="%{text}<br>%{customdata:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title or f"Corner heat-map: {metric}",
        xaxis_title="meters", yaxis_title="meters", height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
