#!/usr/bin/env python3
"""PitMind pipeline validator + threshold tuner.

A turnkey tool to validate the analysis pipeline on real (or synthetic) lap data
and suggest `config.yaml` threshold tweaks.

Run the full pipeline (preprocess -> segmentation -> corners -> features ->
mistakes -> time-loss -> potential lap -> coaching) and print a report. Then,
optionally, recommend threshold values so that a target *flag rate* is met and
write them back to config.yaml.

Usage:
    python -m tools.tune data/monza_real.csv                # validate only
    python -m tools.tune data/monza_real.csv --write        # apply suggested thresholds
    python -m tools.tune data/synthetic_generic_f1.csv      # sanity check on synthetic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pitmind import (
    coaching,
    corners,
    features,
    mistakes,
    potential_lap,
    preprocess,
    segmentation,
    timeloss,
)
from pitmind.config import Config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config extras for tuning (percentile-based flag-rate targets)
# ---------------------------------------------------------------------------
def _tune_options(cfg: Config) -> dict:
    """Read optional tuning knobs from config (safe defaults if absent)."""
    t = cfg.tuning or {}
    return {
        "target_flag_rate": float(t.get("target_flag_rate", 0.30)),
        "min_flag_pct": float(t.get("min_flag_pct", 0.10)),
        "max_flag_pct": float(t.get("max_flag_pct", 0.60)),
    }


# ---------------------------------------------------------------------------
# Pipeline run + report
# ---------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, cfg: Config) -> dict:
    """Run the whole PitMind pipeline and return an analysis bundle."""
    clean = preprocess.preprocess(df, cfg)
    laps = segmentation.valid_laps(clean)
    if not laps:
        raise ValueError("No valid laps found in the session (check recording).")

    # corners from the reference/first lap
    lap1 = clean[clean["lap_number"] == 1].reset_index(drop=True) if (clean["lap_number"] == 1).any() else laps[0]
    detected = corners.detect_corners(lap1, cfg)
    L = corners.track_length_m(laps[0])

    table = features.build_feature_table(clean, cfg, detected_corners=detected, L=L)
    mlist = mistakes.detect_mistakes(table, cfg)
    summary = mistakes.summarize_mistakes(mlist)
    tlist = timeloss.estimate_time_loss(mlist, table, cfg)
    total_loss = sum(t.time_loss_s for t in tlist)

    pot = potential_lap.build_potential_lap(clean, cfg)
    directives = coaching.generate_directives(mlist, tlist)

    return {
        "cfg": cfg,
        "clean": clean,
        "laps": laps,
        "n_laps": len(laps),
        "corners": detected,
        "L": L,
        "table": table,
        "mistakes": mlist,
        "summary": summary,
        "time_losses": tlist,
        "total_time_loss_s": total_loss,
        "potential_lap": pot,
        "directives": directives,
        "lap_times": [lap["timestamp"].max() - lap["timestamp"].min() for lap in laps],
    }


def _fmt_time(s: float) -> str:
    m, sec = divmod(float(s), 60.0)
    return f"{int(m)}:{sec:05.2f}"


def print_report(bundle: dict) -> None:
    cfg = bundle["cfg"]
    print("=" * 60)
    print("PitMind validation report")
    print("=" * 60)
    print(f"Valid laps          : {bundle['n_laps']}")
    print(f"Lap length          : {bundle['L']:.0f} m")
    print(f"Corners detected    : {len(bundle['corners'])}")
    for c in bundle["corners"]:
        print(f"   {c.name:>4}  angle={c.angle_deg:5.1f}°  radius={c.radius_m:6.1f}m  "
              f"apex@{c.apex_tp:.3f}")
    lap_times = bundle["lap_times"]
    if lap_times:
        print(f"Lap times           : {_fmt_time(min(lap_times))} best / "
              f"{_fmt_time(max(lap_times))} worst / {_fmt_time(np.mean(lap_times))} mean")

    print("-" * 60)
    print("Mistake summary")
    s = bundle["summary"]
    print(f"  total mistakes    : {s['total']}")
    print(f"  by type           : {s['by_type']}")
    print(f"  by confidence     : {s['by_confidence']}")
    print(f"  total time loss   : {bundle['total_time_loss_s']:.2f} s "
          f"across {bundle['n_laps']} laps")

    pot = bundle["potential_lap"]
    if pot is not None:
        best_actual = min(bundle["lap_times"])
        print(f"  best actual lap   : {_fmt_time(best_actual)}")
        print(f"  potential lap     : {_fmt_time(pot.total_time_s)} "
              f"(gain {pot.improvement_vs_best_s:+.2f}s vs best)")

    print("-" * 60)
    print("Top coaching directives")
    for d in bundle["directives"][:5]:
        print(f"  [P{d.priority}] {d.message}")

    # threshold distribution diagnostics
    print("-" * 60)
    print("Threshold diagnostics (delta distributions, excl. reference lap)")
    _print_threshold_diagnostics(bundle)


# ---------------------------------------------------------------------------
# Threshold diagnostics + tuning recommendations
# ---------------------------------------------------------------------------
THRESHOLD_MAP = [
    # (metric column, config path label, is_signed, reference-zero filter)
    ("delta_brake_point_m", "ranges.brake_point_delta_m", True),
    ("delta_apex_speed_kmh", "ranges.apex_speed_delta_kmh", True),
    ("delta_throttle_on_s", "ranges.throttle_delay_s", True),
    ("delta_exit_speed_kmh", "ranges.exit_speed_delta_kmh", True),
    ("steering_max", "ranges.steering_excess", False),
]


def _nonzero(values: pd.Series) -> pd.Series:
    """."""
    return values.replace(0.0, np.nan).dropna()


def _percentile(values, p: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def suggest_thresholds(bundle: dict) -> dict:
    """Recommend threshold values so the target flag rate is met.

    For signed deltas we look at the magnitude of non-zero samples; the
    recommended threshold keeps roughly `target_flag_rate` of samples flagged.
    Returns {config_dotted_path: recommended_value}.
    """
    table = bundle["table"]
    opts = _tune_options(bundle["cfg"])
    target = opts["target_flag_rate"]
    lo = opts["min_flag_pct"]
    hi = opts["max_flag_pct"]

    suggestions: dict[str, float] = {}
    out = {}
    for col, path, _signed in THRESHOLD_MAP:
        if col not in table.columns:
            continue
        values = _nonzero(table[col])

        # find the |value| threshold that flags ~target of samples
        if values.empty:
            out[col] = {"path": path, "current": None, "recommended": None, "flag_rate": None}
            continue
        mags = np.abs(values.to_numpy(dtype=float))
        current = _current_threshold(bundle["cfg"], path)
        recommended = float(np.percentile(mags, (1.0 - target) * 100.0)) or 0.0001
        flag_rate = float((mags >= recommended).mean()) if recommended else 0.0

        # don't recommend outside the allowed band
        if flag_rate < lo or flag_rate > hi:
            recommended = None  # leave as-is if we can't hit the band

        suggestions[path] = recommended
        out[col] = {
            "path": path,
            "current": current,
            "recommended": recommended,
            "flag_rate": flag_rate,
            "p50": _percentile(mags, 50),
            "p85": _percentile(mags, 85),
        }
    return out


def _current_threshold(cfg: Config, path: str):
    part = path.split(".")
    if len(part) == 3:  # ranges.brake_point_delta_m.{significant,potential,strong}
        d = getattr(cfg.ranges, part[1])
        return d
    if len(part) == 2:
        return getattr(cfg.ranges, part[1])
    return None


def _print_threshold_diagnostics(bundle: dict) -> None:
    out = suggest_thresholds(bundle)
    for col, info in out.items():
        cur = info["current"]
        cur_str = (
            str(cur) if not isinstance(cur, dict)
            else f"sig={cur.get('significant')} pot={cur.get('potential')} strong={cur.get('strong')}"
        )
        rec = info["recommended"]
        rec_str = f"{rec:.2f}" if rec is not None else "n/a"
        flag = info["flag_rate"]
        print(f"  {col:24s} current={cur_str!s:28s} rec={rec_str:>8s} "
              f"flag_rate={flag:.0%} p50={info['p50']:.2f} p85={info['p85']:.2f}")


# ---------------------------------------------------------------------------
# config.yaml write-back
# ---------------------------------------------------------------------------
def apply_thresholds(suggestions: dict, config_path: Path = CONFIG_PATH) -> None:
    """Write recommended thresholds into config.yaml under their dotted paths."""
    import yaml

    text = Path(config_path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}

    written = []
    for path, rec in suggestions.items():
        if rec is None:
            continue
        parts = path.split(".")
        # only handle two-level paths (ranges.<attr>); skip dict thresholds for now
        if len(parts) == 2 and isinstance(rec, (int, float)):
            raw.setdefault(parts[0], {})
            # apply a float (leave dict-valued thresholds alone)
            existing = raw[parts[0]].get(parts[1])
            if isinstance(existing, (int, float)) or existing is None:
                raw[parts[0]][parts[1]] = float(rec)
                written.append(path)

    if not written:
        print("Nothing to write (no float thresholds changed).")
        return
    with Path(config_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    print(f"Wrote thresholds to {config_path}: {written}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="Path to recorded/synthetic telemetry CSV")
    parser.add_argument("--write", action="store_true",
                        help="Apply recommended thresholds to config.yaml")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="config.yaml path (default: project config)")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2
    cfg = Config.from_file(args.config)
    df = pd.read_csv(csv_path)

    try:
        bundle = run_pipeline(df, cfg)
    except ValueError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 3

    print_report(bundle)

    if args.write:
        suggestions = suggest_thresholds(bundle)
        recs = {info["path"]: info["recommended"] for info in suggestions.values()}
        apply_thresholds(recs, config_path=Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
