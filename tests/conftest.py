"""Auto-generate gitignored data/*.csv fixtures so tests pass on a clean checkout.

`data/*.csv` is deliberately not committed (fixtures are generated), so a fresh
clone (or CI) has no CSVs. Several test modules read them from disk. This
session-scoped autouse fixture regenerates any missing fixture using the same
generators the tooling uses, keeping the tree hermetic for CI and local clones.

IMPORTANT: generation must use LOCAL seeded RNGs, never the shared global
`synthetic.generator.RNG`. Tests rely on that module RNG for determinism (e.g.
`test_corners` calls `generate_session` with the default shared RNG), so
consuming it here would silently shift the whole suite's stream and flip
borderline assertions.

Required fixtures (paths match what the tests read):
  data/synthetic_generic_f1.csv (+ _ground_truth.json)   -> synthetic/generator.py
  data/f1_monza_laps.csv                                 -> tools.fixture_f1.generate_f1_lap_set
  data/f1_monza_driver_{VER,LEC,SAI}.csv                 -> tools.fixture_f1.generate_driver_field
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DRIVERS = {"VER": 1, "LEC": 2, "SAI": 3}
_SEED = 42


def _ensure_synthetic_generic_f1() -> None:
    if (DATA_DIR / "synthetic_generic_f1.csv").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    from synthetic.generator import OUT_CSV, OUT_JSON, generate_session

    session, gt = generate_session(12, "monza", rng=np.random.default_rng(_SEED))
    session.to_csv(str(OUT_CSV), index=False)
    OUT_JSON.write_text(json.dumps(gt, indent=2), encoding="utf-8")


def _ensure_f1_monza_laps() -> None:
    if (DATA_DIR / "f1_monza_laps.csv").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
    from tools.fixture_f1 import generate_f1_lap_set

    generate_f1_lap_set("monza", 12, rng_seed=_SEED).to_csv(
        str(DATA_DIR / "f1_monza_laps.csv"), index=False
    )


def _ensure_f1_monza_drivers() -> None:
    if all((DATA_DIR / f"f1_monza_driver_{code}.csv").exists() for code in DRIVERS):
        return
    DATA_DIR.mkdir(exist_ok=True)
    from tools.fixture_f1 import generate_driver_field

    for code, sdf in generate_driver_field("monza", 8, DRIVERS).items():
        sdf.to_csv(str(DATA_DIR / f"f1_monza_driver_{code}.csv"), index=False)


@pytest.fixture(scope="session", autouse=True)
def _generate_fixtures() -> None:
    _ensure_synthetic_generic_f1()
    _ensure_f1_monza_laps()
    _ensure_f1_monza_drivers()
