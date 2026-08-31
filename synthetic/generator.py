"""Synthetic ACC-style telemetry generator.

Emits CSVs matching the "contract" schema (see design.md and config.py). Laps are
driven by a kinematic car model around a generic F1-style circuit. Individual
laps/corners can be degraded with *known, labeled* mistakes (early/late braking,
low apex speed, late throttle, slow exit) so the analysis pipeline and tests have
ground truth.

Usage:
    uv run python synthetic/generator.py            # default 12-lap session
    uv run python synthetic/generator.py --laps 5    # fewer laps
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_CSV = DATA_DIR / "synthetic_generic_f1.csv"
OUT_JSON = DATA_DIR / "synthetic_generic_f1_ground_truth.json"

SAMPLE_RATE = 60.0  # Hz
DT = 1.0 / SAMPLE_RATE

V_MAX_KMH = 310.0
A_ACCEL = 8.0    # m/s^2 full-throttle accel
A_BRAKE = 12.0   # m/s^2 braking decel
A_LAT = 14.0     # m/s^2 lateral acceleration cap -> corner speed limit

# (angle_deg, radius_m); angles sum to exactly 360 -> closed loop
CORNERS = [
    (90.0, 55.0),   # hairpin 1
    (30.0, 180.0),  # fast sweeper
    (55.0, 110.0),  # medium right
    (60.0, 100.0),  # medium left
    (110.0, 45.0),  # hairpin 2
    (15.0, 200.0),  # quick kink
]
STRAIGHTS = [2600.0, 800.0, 1500.0, 350.0, 1200.0, 600.0]

GEAR_BANDS = [0.0, 65.0, 115.0, 165.0, 215.0, 265.0, V_MAX_KMH + 1.0]  # km/h
RPM_IDLE, RPM_PEAK = 6500.0, 9500.0

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# Track


@dataclass
class Corner:
    index: int
    start_s: float
    end_s: float
    radius: float
    angle_rad: float
    dir_: int
    limit_speed: float  # m/s cornering speed cap

    @property
    def length(self) -> float:
        return self.end_s - self.start_s

    @property
    def name(self) -> str:
        return f"T{self.index + 1}"


def build_track(corners=(CORNERS), straights=(STRAIGHTS)):
    """Return (corners, L, centerline_x, centerline_y).

    Track is a closed loop: straights and corners alternate, starting with the
    start/finish straight. All corners are right-handers (dir = -1) -> net heading
    change is -2*pi. The centerline is interpolable by arc length s in [0, L).
    """
    assert len(corners) == len(straights)

    n = len(corners)
    s, heading = 0.0, 0.0
    xs, ys = [0.0], [0.0]
    vertices_s: list[float] = [0.0]
    corner_list: list[Corner] = []

    for i in range(n):
        length = straights[i]
        s += length
        xs.append(xs[-1] + length * math.cos(heading))
        ys.append(ys[-1] + length * math.sin(heading))
        vertices_s.append(s)

        angle_deg, radius = corners[i]
        angle = math.radians(angle_deg)
        dir_ = -1  # right-handed
        a = angle * dir_
        start_s = s
        h1 = heading + a
        xs.append(xs[-1] + radius * (math.sin(h1) - math.sin(heading)))
        ys.append(ys[-1] + radius * (-math.cos(h1) + math.cos(heading)))
        s += radius * abs(a)
        vertices_s.append(s)
        heading = h1
        corner_list.append(
            Corner(
                index=len(corner_list),
                start_s=start_s,
                end_s=s,
                radius=radius,
                angle_rad=a,
                dir_=dir_,
                limit_speed=math.sqrt(A_LAT * radius),
            )
        )

    # dense centerline sampled at 1m
    ds = 1.0
    s_grid = np.arange(0.0, L := float(s), ds)
    xs = np.array(xs)
    ys = np.array(ys)
    vert = np.array(vertices_s)
    cx = np.interp(s_grid, vert, xs)
    cy = np.interp(s_grid, vert, ys)
    return corner_list, L, cx, cy


def curvature_at(s: float, corners: list[Corner], blend: float = 25.0) -> float:
    """Signed curvature (1/radius, negative for right-handers) at arc length s,
    with linear blends on corner entry/exit."""
    for c in corners:
        start, end = c.start_s, c.end_s
        if start <= s < end:
            ramp = min(1.0, min(s - start, end - s) / blend)
            return (1.0 / c.radius) * c.dir_ * ramp
    for c in corners:
        start, end = c.start_s, c.end_s
        if start - blend < s < start:
            ramp = (s - (start - blend)) / blend
            if ramp > 0:
                return (1.0 / c.radius) * c.dir_ * ramp
        if end < s < end + blend:
            ramp = (end + blend - s) / blend
            if ramp > 0:
                return (1.0 / c.radius) * c.dir_ * ramp
    return 0.0


# --------------------------------------------------------------------------- #
# Lap driving (kinematic plan)


@dataclass
class LapParams:
    """Per-lap driver deviations, keyed by corner index."""

    brake_shift_m: dict[int, float] = field(default_factory=dict)   # + earlier / - later
    apex_deficit_kmh: dict[int, float] = field(default_factory=dict)
    throttle_delay_s: dict[int, float] = field(default_factory=dict)
    exit_deficit_kmh: dict[int, float] = field(default_factory=dict)
    v_max_kmh: float = V_MAX_KMH


def _lead_distance(corners, L, i) -> float:
    d = corners[(i + 1) % len(corners)].start_s - corners[i].end_s
    return d if d >= 0 else d + L


def plan_lap(corners: list[Corner], L: float, params: LapParams) -> dict:
    """Compute the speed plan for one lap.

    Returns a dict with per-corner arrays: brake_start_s, entry_speed, apex_speed,
    throttle_on_s, exit_speed (m/s), plus apex/entry/exit/brake lists.
    """
    n = len(corners)
    v_max = params.v_max_kmh / 3.6

    apex = []
    for c in corners:
        a = c.limit_speed
        a -= params.apex_deficit_kmh.get(c.index, 0.0) / 3.6
        apex.append(max(4.0, a))

    # exit speed from each corner: apex -> full throttle until corner end
    exit_speed = [0.0] * n
    for _ in range(4):
        for i, c in enumerate(corners):
            a = apex[i]
            d_thr = params.throttle_delay_s.get(i, 0.0) * a
            remain = max(0.0, c.length * 0.98 - d_thr)
            v_exit = math.sqrt(max(0.0, a * a + 2 * A_ACCEL * remain))
            v_exit -= params.exit_deficit_kmh.get(i, 0.0) / 3.6
            exit_speed[i] = max(a * 0.5, v_exit)

    # entry/brake start per corner (lead-in straight from previous corner exit)
    entry, brake_start = [0.0] * n, [0.0] * n
    for _ in range(4):
        for i, c in enumerate(corners):
            v_out_prev = exit_speed[(i - 1) % n]
            D = _lead_distance(corners, L, (i - 1) % n)
            x_acc = (apex[i] * apex[i] - v_out_prev * v_out_prev + 2 * A_BRAKE * D) / (2 * (A_ACCEL + A_BRAKE))
            v_peak = min(v_max, math.sqrt(max(0.0, v_out_prev * v_out_prev + 2 * A_ACCEL * x_acc)))
            xd = min(max(0.0, D), (v_peak * v_peak - apex[i] * apex[i]) / (2 * A_BRAKE)) if D > 0 else 0.0
            xd += params.brake_shift_m.get(i, 0.0)
            xd = max(0.0, min(xd, max(0.0, D)))
            brake_start[i] = c.start_s - xd
            entry[i] = math.sqrt(max(0.0, apex[i] * apex[i] + 2 * A_BRAKE * xd))
            entry[i] = min(entry[i], v_peak)

    # throttle-on arc positions
    return {
        "corners": corners,
        "L": L,
        "apex": apex,
        "exit": exit_speed,
        "entry": entry,
        "brake_start": brake_start,
        "brake_shift": params.brake_shift_m,
    }


def build_speed_profile(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_pts, v_pts) breakpoints of the velocity profile, sorted by s.

    Straight-line (s, v) interpolation approximates accel/brake ramps.
    """
    corners = plan["corners"]
    L = plan["L"]
    apex, exit_, entry, brake_start = plan["apex"], plan["exit"], plan["entry"], plan["brake_start"]
    n = len(corners)

    pts = []
    for i, c in enumerate(corners):
        pts.append((brake_start[i], entry[i]))   # braking onset
        pts.append((c.start_s, apex[i]))         # arrive at corner start @ apex
        pts.append((c.start_s + _thr_offset(i, plan), apex[i]))
        pts.append((c.end_s, exit_[i]))          # corner exit

    # dedupe same-s consecutive points keeping max relevance: sort then merge mid-straights
    pts.sort(key=lambda p: (p[0], p[1]))
    merged = []
    for s, v in pts:
        if merged and abs(merged[-1][0] - s) < 1e-9:
            continue
        merged.append((s, v))

    s_arr = np.array([p[0] for p in merged], dtype=float)
    v_arr = np.array([p[1] for p in merged], dtype=float)

    # drop points sitting on the wrap boundaries, then add clean periodic ends
    mask = (s_arr > 1e-6) & (s_arr < L - 1e-6)
    s_arr, v_arr = s_arr[mask], v_arr[mask]
    v0 = float(v_arr[s_arr.argmin()])  # approx main-straight plateau at lap start
    s_arr = np.concatenate([[0.0], s_arr, [L]])
    v_arr = np.concatenate([[v0], v_arr, [v0]])
    return s_arr, v_arr


def _thr_offset(i: int, plan: dict) -> float:
    corners = plan["corners"]
    return plan.get("_thr_delay", {}).get(i, 0.0) * plan["apex"][i]


# --------------------------------------------------------------------------- #
# Sampling


def sample_lap(plan, params, lap_index, lap_start_time, cx, cy, L) -> pd.DataFrame:
    s_arr, v_arr = build_speed_profile(plan)
    corners = plan["corners"]
    n = len(corners)
    brake_ranges = [(plan["brake_start"][i], corners[i].start_s) for i in range(n)]
    apex = plan["apex"]

    rows = []
    s = 0.0
    t = lap_start_time

    while s < L - 1e-6:
        v = float(np.interp(s, s_arr, v_arr))
        in_corner = any(c.start_s <= s < c.end_s for c in corners)

        # brake zone membership
        active = [(bs, be) for bs, be in brake_ranges if bs <= s < be]
        braking = bool(active)

        # throttle
        throttle = 1.0 if s >= L * 0.5 else 1.0  # placeholder overridden below
        if braking:
            throttle = 0.0
        elif in_corner:
            throttle = _throttle_in_corner(s, _corner_at(corners, s), plan)
        else:
            throttle = 1.0

        # brake pressure ramp-up / release on the braking zone
        brake = 0.0
        if active:
            bs, be = active[0]
            sep = max(1e-6, (be - bs))
            brake = min(1.0, (s - bs) / max(1e-6, sep * 0.35))          # quick build
            brake = min(brake, (be - s) / max(1e-6, sep * 0.15))        # trail-off
            brake = float(np.clip(brake, 0.0, 1.0))

        k = curvature_at(s, corners)
        steering = float(np.clip(k * 20.0, -1.0, 1.0))
        speed_kmh = v * 3.6
        gear = int(np.searchsorted(GEAR_BANDS, speed_kmh, side="right"))
        gear = max(1, min(6, gear))
        if speed_kmh < 1.0:
            rpm = RPM_IDLE
        else:
            lo, hi = GEAR_BANDS[gear - 1], GEAR_BANDS[gear]
            rpm = RPM_IDLE + (RPM_PEAK - RPM_IDLE) * float(np.clip((speed_kmh - lo) / (hi - lo), 0.0, 1.0))
        x = float(np.interp(s, np.arange(0.0, L, 1.0)[: len(cx)], cx))
        y = float(np.interp(s, np.arange(0.0, L, 1.0)[: len(cy)], cy))
        sector = int(s / L * 3) + 1 if s / L * 3 < 3 else 3
        rows.append({
            "timestamp": round(t, 4),
            "lap_number": lap_index,
            "sector": sector,
            "track_position": round(s / L, 6),
            "speed_kmh": round(speed_kmh, 3),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
            "steering": round(steering, 4),
            "gear": gear,
            "rpm": round(rpm, 1),
            "x": round(x + RNG.normal(0, 0.3), 2),
            "y": round(y + RNG.normal(0, 0.3), 2),
            "z": round(RNG.normal(0, 0.05), 3),
        })
        s += v * DT
        t += DT
    return pd.DataFrame(rows)


def _corner_at(corners, s):
    for c in corners:
        if c.start_s <= s <= c.end_s:
            return c
    return None


def _thr_offset_for(s, c, plan):
    if c is None:
        return 0.0
    return plan["_thr_delay"].get(c.index, 0.0) * plan["apex"][c.index]


def _throttle_in_corner(s, c, plan):
    if c is None:
        return 0.55
    d_thr = plan["_thr_delay"].get(c.index, 0.0) * plan["apex"][c.index]
    thr_on = c.start_s + d_thr
    if s < thr_on:
        return 0.0
    frac = (s - thr_on) / max(1e-6, c.end_s - thr_on)
    return float(np.clip(0.4 + 0.6 * frac, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Session


def random_lap_params(rng: np.random.Generator, clean: bool) -> LapParams:
    p = LapParams()
    if clean:
        return p
    n_corners = len(CORNERS)
    for i in range(n_corners):
        r = rng.random()
        if r < 0.25:
            p.brake_shift_m[i] = rng.uniform(4.0, 20.0)      # early
        elif r < 0.35:
            p.brake_shift_m[i] = -rng.uniform(3.0, 10.0)     # late
        if rng.random() < 0.25:
            p.apex_deficit_kmh[i] = rng.uniform(5.0, 16.0)
        if rng.random() < 0.2:
            p.throttle_delay_s[i] = rng.uniform(0.1, 0.3)
        if rng.random() < 0.15:
            p.exit_deficit_kmh[i] = rng.uniform(4.0, 12.0)
    return p


def generate_session(laps: int, rng: np.random.Generator = RNG) -> tuple[pd.DataFrame, dict]:
    corners, L, cx, cy = build_track()
    cx, cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)

    frames = []
    ground_truth = {
        "track": {"kind": "generic_f1", "length_m": round(float(L), 1), "corners": len(corners)},
        "laps": [],
    }
    clean_laps = set()

    for lap in range(1, laps + 1):
        clean = lap == 1 or lap == 2
        if clean:
            clean_laps.add(lap)
        params = random_lap_params(rng, clean)
        plan = plan_lap(corners, L, params)
        # stash throttle delay for the sampling helpers
        plan["_thr_delay"] = params.throttle_delay_s
        lap_start = sum(len(f) for f in frames) / SAMPLE_RATE
        df = sample_lap(plan, params, lap, lap_start, cx, cy, L)
        frames.append(df)
        gt_corners = []
        for c in corners:
            gt = {}
            if c.index in params.brake_shift_m:
                gt["brake_shift_m"] = params.brake_shift_m[c.index]
            if c.index in params.apex_deficit_kmh:
                gt["apex_deficit_kmh"] = params.apex_deficit_kmh[c.index]
            if c.index in params.throttle_delay_s:
                gt["throttle_delay_s"] = params.throttle_delay_s[c.index]
            if c.index in params.exit_deficit_kmh:
                gt["exit_deficit_kmh"] = params.exit_deficit_kmh[c.index]
            gt_corners.append({"corner": c.name, "start_s": round(c.start_s, 1), **gt})
        ground_truth["laps"].append({"lap": lap, "clean": clean, "corners": gt_corners})

    session = pd.concat(frames, ignore_index=True)
    return session, ground_truth


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic ACC telemetry")
    ap.add_argument("--laps", type=int, default=12)
    args = ap.parse_args()

    session, gt = generate_session(args.laps)
    DATA_DIR.mkdir(exist_ok=True)
    session.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    cols = ["timestamp", "lap_number", "track_position", "speed_kmh",
            "throttle", "brake", "steering", "gear", "rpm", "x", "y"]
    print(session[cols].head(12).to_string(index=False))
    print(f"\nwrote {OUT_CSV} ({len(session)} rows, {session['lap_number'].nunique()} laps)")
    print(f"wrote {OUT_JSON}")
    print(f"track length ~{gt['track']['length_m']:.0f} m, {gt['track']['corners']} corners")


if __name__ == "__main__":
    main()