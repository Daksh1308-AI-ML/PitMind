#!/usr/bin/env python3
"""Generate an offline F1-style telemetry fixture for M2 validation.

Real F1 telemetry (FastF1) can't be fetched in CI/offline, so we synthesize a
faithful stand-in: drive the real Monza circuit with the synthetic generator,
then reshape it into *raw FastF1 form* (Speed km/h, Throttle 0-100 %, Brake as a
boolean, no steering, X/Y/Z in 1/10 m, coarse ~4 Hz broadcast sampling) and run it
through `FastF1Bridge` to produce the 13-column contract CSV the pipeline eats.

This exercises the exact F1 ingest path end-to-end (bridge + pipeline) without
network access, and gives `tools.tune --f1` a Monza lap set whose corner count
(~7) and time-loss ranges we can sanity-check (todo.md M2).

Usage:
    uv run python tools/fixture_f1.py                 # -> data/f1_monza_laps.csv
    uv run python tools/fixture_f1.py --laps 14       # more laps
    uv run python tools/fixture_f1.py --track spa     # any vendored circuit

The output lives at DATA_DIR / f"f1_{track}_laps.csv" (gitignored like other CSVs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# allow running as a script (python tools/fixture_f1.py) not just a module
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

# F1 broadcast sampling is ~4 Hz vs ACC's 60 Hz.
F1_SAMPLE_HZ = 4.0


def _to_fastf1_shape(contract: pd.DataFrame) -> pd.DataFrame:
    """Reshape a contract DataFrame (from the synthetic generator) into raw
    FastF1 telemetry form, keeping the *exact* units FastF1 emits."""
    # subsample to F1's coarse broadcast rate
    step = max(1, round(60.0 / F1_SAMPLE_HZ))
    raw = contract.iloc[::step].copy().reset_index(drop=True)

    f1 = pd.DataFrame(
        {
            "Time": raw["timestamp"],
            "Speed": raw["speed_kmh"],
            "Throttle": raw["throttle"] * 100.0,          # 0-1 -> 0-100 %
            "Brake": raw["brake"] > 0.0,                  # -> boolean
            "nGear": raw["gear"],
            "RPM": raw["rpm"],
            "X": (raw["x"] * 10).round().astype(int),     # meters -> 1/10 m
            "Y": (raw["y"] * 10).round().astype(int),
            "Z": (raw["z"] * 10).round().astype(int),
            "SectorNumber": raw["sector"],
        }
    )
    # FastF1 broadcasts no steering channel -> simply omit it.
    return f1


def generate_f1_lap_set(track: str = "monza", laps: int = 12, rng_seed: int | None = None) -> pd.DataFrame:
    """Produce a bridged F1-style contract CSV for the given circuit/lap count."""
    from synthetic.generator import generate_session, RNG
    from f1.fastf1_bridge import FastF1Bridge

    import numpy as np
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else RNG
    session, _gt = generate_session(laps, track, rng=rng)
    # split into per-lap frames and reshape each to FastF1 form
    raw_frames: list[pd.DataFrame] = []
    for _, lap in session.groupby("lap_number"):
        raw_frames.append(_to_fastf1_shape(lap))

    bridge = FastF1Bridge()
    return bridge.convert_session(raw_frames)


def generate_driver_field(track: str = "monza", laps: int = 8,
                          drivers: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Produce a dict of {driver_code: bridged session} for a small F1 field.

    Each driver is generated with its own RNG seed so their mistake profiles
    differ (mimicking different driving quality), then reshaped + bridged.
    This is what multi-driver comparison (todo.md M3) runs on.

    Args:
        track: Circuit id.
        laps: Laps per driver.
        drivers: {driver_code: rng_seed}. Defaults to a 3-driver field.
    """
    if drivers is None:
        drivers = {"VER": 1, "LEC": 2, "SAI": 3}
    return {
        code: generate_f1_lap_set(track, laps, rng_seed=seed)
        for code, seed in drivers.items()
    }


def main(argv: list[str] | None = None) -> int:
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--laps", type=int, default=12)
    ap.add_argument("--track", type=str, default="monza")
    ap.add_argument("--drivers", type=str, default=None,
                    help='JSON map {code: seed} to also write one CSV per driver '
                         '(multi-driver fixture for M3 comparison). '
                         'e.g. \'{"VER":1,"LEC":2,"SAI":3}\'')
    args = ap.parse_args(argv)

    df = generate_f1_lap_set(args.track, args.laps)
    out = DATA_DIR / f"f1_{args.track}_laps.csv"
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {out} ({len(df)} samples, {df['lap_number'].nunique()} laps, "
          f"{list(df.columns)})")
    print(f"steering is all-NaN: {bool(df['steering'].isna().all())} | "
          f"brake is boolean: {set(df['brake'].unique()) <= {0.0, 1.0}}")

    if args.drivers:
        drivers = json.loads(args.drivers)
        field = generate_driver_field(args.track, args.laps, drivers)
        for code, sdf in field.items():
            p = DATA_DIR / f"f1_{args.track}_driver_{code}.csv"
            sdf.to_csv(p, index=False)
            print(f"wrote {p} ({len(sdf)} samples, {sdf['lap_number'].nunique()} laps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
