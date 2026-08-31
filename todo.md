# PitMind — Progress Tracker

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · note blockers inline.

## 0. Project Setup

- [x] Repo scaffolding: `pyproject.toml`, `config.yaml`, `data/`, `tests/`, venv
- [x] Dependencies installed: numpy, pandas, scipy, streamlit, plotly, pyaccsharedmemory
- [x] `git init` + first commit (pushed to Daksh1308-AI-ML/PitMind)

## 1. Data Layer (doc v0.1)

- [x] Define CSV schema (design.md "the contract")
- [x] `synthetic/generator.py` — kinematic lap models with injected, known mistakes + ground-truth JSON
- [x] Generate seed synthetic laps into `data/`
- [x] Smoke tests for schema + physics (`tests/test_synthetic.py`, 11 passing)
- [x] `pitmind/preprocess.py` — validate → resample to fixed Hz → smooth
- [x] `pitmind/segmentation.py` — lap splitting via `track_position` wrap; drop invalid laps
      (`tests/test_segmentation.py`, 8 passing)
- [~] **Synthetic track → real F1 circuits** (bacinger/f1-circuits, MIT/OSM):
  - [ ] Vendor GeoJSON circuit data into `data/circuits/` — proposal: **Monza (default), Spa,
        Silverstone (+ optional Imola)** — plus `data/circuits/README.md` attribution
  - [ ] `synthetic/circuit.py` — load GeoJSON → project lat/lon → resample ~1 m arc grid
        → auto-derive corner regions (reuse chord-curvature logic) → `Track`/`Corner` list
  - [ ] Refactor `synthetic/generator.py` `build_track()` to source geometry from a real circuit
        (default `monza`); steering/curvature from the centerline curvature profile
  - [ ] Fix the legacy generic track's open-loop seam (keep as `generic` fallback fixture)
  - [ ] Regenerate `data/synthetic_*.csv` + ground truth on the new track

## 2. Core Pipeline

- [x] `pitmind/corners.py` — track-agnostic corner detection
      (chord-based curvature + brake-zone confirmation; 4000-pt arc grid)
- [x] `pitmind/events.py` — brake point, entry/apex/exit speed, throttle-on point
      (`corner_features_table`)
- [ ] `tests/test_corners.py` — **waits on section 1 switches**: make checks circuit-agnostic
      (corner count + positions from GT, not hardcoded 6), drop the invalid "total turn = 360°" check,
      add a track-module test (loop closure, length ≈ official, both turn directions)
- [ ] `pitmind/reference.py` — pick reference lap (best) + per-corner reference feature values
- [ ] `pitmind/features.py` — decision: merge into `events.corner_features_table`
      or keep as thin wrapper

## 3. Intelligence (doc v0.3–v0.4)

- [ ] `pitmind/mistakes.py` — threshold rules → mistake classes + confidence
- [ ] `pitmind/timeloss.py` — kinematic time-loss estimate
- [ ] `pitmind/potential_lap.py` — best-sector composite lap

## 4. Coaching + Dashboard (doc v0.5)

- [ ] `pitmind/coaching.py` — template directives + priority filter
- [ ] `dashboard/app.py` — Streamlit: lap list, telemetry plots, corner table, mistake cards,
      overlay, potential lap

## 5. Tests & Validation

- [ ] pytest: corner detection (circuit-agnostic) + `synthetic/circuit.py`
- [ ] pytest: mistake detection — synthetic laps with known mistakes must be found
- [ ] pytest: time-loss sane ranges
- [ ] `recorder/record_acc.py` — ACC shared memory → lap-split CSVs
- [ ] Record real ACC F1-circuit laps; validate pipeline + tune thresholds in `config.yaml`

## 6. Later (roadmap v0.6+)

- [ ] Live telemetry streaming (reuse recorder reader)
- [ ] Optional LLM coaching phrasing (isolated hook)
- [ ] ML models replacing thresholds (doc §26 stages)
- [ ] Voice feedback loop