#!/usr/bin/env python3
"""
Circuit loader: GeoJSON -> projected meters -> ~1m arc grid -> corner regions.

Reuses chord-curvature logic from pitmind.corners for consistency.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CIRCUITS_DIR = Path(__file__).parent.parent / "data" / "circuits"

GRID_N = 4000
CHORD_W = 24
DEFAULT_CURV_THRESHOLD = 0.0008
DEFAULT_MIN_ANGLE_DEG = 10.0
DEFAULT_MERGE_M = 40.0


@dataclass(frozen=True)
class Corner:
    index: int
    name: str
    start_tp: float
    end_tp: float
    apex_tp: float
    angle_deg: float
    radius_m: float
    min_speed_kmh: float
    center_tp: float


@dataclass(frozen=True)
class Track:
    circuit_id: str
    name: str
    length_m: float
    centerline_xy: np.ndarray
    curvature: np.ndarray
    turn_angle: np.ndarray
    corners: tuple[Corner, ...]
    tp_grid: np.ndarray


def _equirectangular_project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_mean = np.mean(lat)
    cos_lat = math.cos(math.radians(lat_mean))
    R = 6371000.0
    x = R * np.radians(lon) * cos_lat
    y = R * np.radians(lat)
    return x, y


def _resample_to_grid(x: np.ndarray, y: np.ndarray, n: int = GRID_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.hypot(dx, dy)
    cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = cum_len[-1]
    target_cum = np.linspace(0.0, total_len, n, endpoint=False)
    x_new = np.interp(target_cum, cum_len, x)
    y_new = np.interp(target_cum, cum_len, y)
    tp_grid = target_cum / total_len
    return x_new, y_new, tp_grid


def _ensure_closed_loop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-6:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    return x, y


def _chord_curvature(x: np.ndarray, y: np.ndarray, w: int = CHORD_W) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute curvature and heading using chord method.
    Returns (curvature, turn_angle, heading) where:
    - curvature = turn / (2 * w * ds) (signed rad/m)
    - turn_angle = heading_forward - heading_backward (wrapped to [-pi, pi])
    - heading = heading_forward (direction of chord i to i+w)
    """
    n = len(x)
    curvature = np.zeros(n)
    turn_angle = np.zeros(n)
    heading = np.zeros(n)
    ds = np.mean(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    for i in range(n):
        im = (i - w) % n
        ip = (i + w) % n
        dx1 = x[i] - x[im]
        dy1 = y[i] - y[im]
        dx2 = x[ip] - x[i]
        dy2 = y[ip] - y[i]

        ang1 = math.atan2(dy1, dx1)  # backward chord direction
        ang2 = math.atan2(dy2, dx2)  # forward chord direction

        turn = ang2 - ang1
        turn = (turn + math.pi) % (2.0 * math.pi) - math.pi

        norm1 = math.hypot(dx1, dy1)
        norm2 = math.hypot(dx2, dy2)
        if norm1 > 1e-9 and norm2 > 1e-9:
            curvature[i] = turn / (2.0 * w * ds)
            turn_angle[i] = turn
            heading[i] = ang2

    return curvature, turn_angle, heading


def _detect_corner_regions(
    tp_grid: np.ndarray,
    curvature: np.ndarray,
    turn_angle: np.ndarray,
    heading: np.ndarray,
    length_m: float,
    curv_threshold: float = DEFAULT_CURV_THRESHOLD,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    merge_m: float = DEFAULT_MERGE_M,
) -> list[Corner]:
    n = len(curvature)
    engaged = np.abs(curvature) > curv_threshold

    bounds = []
    in_run = False
    for i in range(n):
        if engaged[i] and not in_run:
            start = i
            in_run = True
        elif not engaged[i] and in_run:
            bounds.append((start, i))
            in_run = False
    if in_run:
        bounds.append((start, n))

    if not bounds:
        return []

    merged = []
    for b in bounds:
        if merged and (tp_grid[b[0]] - tp_grid[merged[-1][1]]) * length_m < merge_m:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)

    corners = []
    min_angle = math.radians(min_angle_deg)
    for idx, (si, ei) in enumerate(merged):
        if ei <= si:
            continue
        region_curv = curvature[si:ei]
        # Net heading change from start to end of corner region
        # heading at si is backward chord direction, at ei-1 is forward chord direction
        # For net turn, use forward heading at end minus backward heading at start
        h_start = math.atan2(
            np.sin(math.atan2(
                np.sin(turn_angle[si]) if False else 0, 0)) if False else 0, 0
        )
        # Simpler: net turn = angle between chord at ei-1 (forward) and chord at si (backward)
        # Actually: heading at si = backward chord direction at si
        # heading at ei-1 = forward chord direction at ei-1
        # But we need consistent reference. Use forward heading at both ends.
        total_turn = heading[ei - 1] - heading[si]
        total_turn = (total_turn + math.pi) % (2.0 * math.pi) - math.pi

        if abs(total_turn) < min_angle:
            continue
        pk = int(np.argmax(np.abs(region_curv)))
        apex_tp = float(tp_grid[si + pk])
        start_tp = float(tp_grid[si])
        end_tp = float(tp_grid[ei - 1])
        radius = 1.0 / max(np.abs(region_curv).max(), 1e-9)
        min_speed = math.sqrt(14.0 * radius) * 3.6
        corners.append(Corner(
            index=idx,
            name=f"T{idx + 1}",
            start_tp=start_tp,
            end_tp=end_tp,
            apex_tp=apex_tp,
            angle_deg=math.degrees(total_turn),
            radius_m=radius,
            min_speed_kmh=min_speed,
            center_tp=((start_tp + end_tp) / 2.0) % 1.0,
        ))

    return corners


def load_circuit(
    circuit_id: str,
    curv_threshold: float = DEFAULT_CURV_THRESHOLD,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    merge_m: float = DEFAULT_MERGE_M,
) -> Track:
    path = CIRCUITS_DIR / f"{circuit_id}.geojson"
    if not path.exists():
        available = [p.stem for p in CIRCUITS_DIR.glob("*.geojson")]
        raise FileNotFoundError(f"Circuit '{circuit_id}' not found. Available: {available}")

    with open(path) as f:
        data = json.load(f)

    coords = np.array(data["features"][0]["geometry"]["coordinates"])
    lon = coords[:, 0]
    lat = coords[:, 1]
    x, y = _equirectangular_project(lon, lat)
    x, y = _ensure_closed_loop(x, y)
    x, y, tp_grid = _resample_to_grid(x, y, GRID_N)

    curvature, turn_angle, heading = _chord_curvature(x, y)
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    length_m = float(np.sum(np.hypot(dx, dy)))

    corners = _detect_corner_regions(tp_grid, curvature, turn_angle, heading, length_m,
                                     curv_threshold, min_angle_deg, merge_m)

    return Track(
        circuit_id=circuit_id,
        name=data["features"][0]["properties"]["Name"],
        length_m=length_m,
        centerline_xy=np.column_stack((x, y)),
        curvature=curvature,
        turn_angle=turn_angle,
        corners=tuple(corners),
        tp_grid=tp_grid,
    )


def list_circuits() -> list[str]:
    return sorted(p.stem for p in CIRCUITS_DIR.glob("*.geojson"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Circuit loader / info")
    parser.add_argument("--list", action="store_true", help="List available circuits")
    parser.add_argument("--info", type=str, help="Show info for circuit")
    args = parser.parse_args()

    if args.list:
        for cid in list_circuits():
            print(cid)
        return

    if args.info:
        track = load_circuit(args.info)
        print(f"Circuit: {track.name} ({track.circuit_id})")
        print(f"Length: {track.length_m:.1f} m")
        print(f"Grid points: {len(track.tp_grid)}")
        print(f"Corners detected: {len(track.corners)}")
        for c in track.corners:
            print(f"  {c.name}: angle={c.angle_deg:.1f}deg radius={c.radius_m:.1f}m "
                  f"min_speed={c.min_speed_kmh:.1f}kmh apex_tp={c.apex_tp:.3f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()