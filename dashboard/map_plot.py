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
