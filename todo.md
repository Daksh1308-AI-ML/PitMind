# PitMind — Progress Tracker

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · note blockers inline.

## 0. Project Setup

- [ ] Repo scaffolding: `pyproject.toml`, `config.yaml`, `data/`, `tests/`, venv
- [ ] Dependencies installed: numpy, pandas, scipy, streamlit, plotly, pyaccsharedmemory
- [ ] `git init` + first commit

## 1. Data Layer (doc v0.1)

- [ ] Define CSV schema (design.md "the contract")
- [ ] `synthetic/generate_synthetic.py` — generic F1-circuit laps with injected, known mistakes
- [ ] `recorder/record_acc.py` — ACC shared memory → lap-split CSVs
- [ ] Generate seed synthetic laps into `data/`

## 2. Core Pipeline

- [ ] `pitmind/preprocess.py` — resample to 60 Hz, smooth, fill gaps
- [ ] `pitmind/segmentation.py` — lap splitting via track_position wrap; drop invalid laps
- [ ] `pitmind/corners.py` — automatic corner detection (heading / curvature + brake zones)
- [ ] `pitmind/events.py` — brake point, entry/apex/exit speed, throttle-on point
- [ ] `pitmind/features.py` — per-lap-per-corner feature table
- [ ] `pitmind/reference.py` — reference lap + track-distance alignment

## 3. Intelligence (doc v0.3–v0.4)

- [ ] `pitmind/mistakes.py` — threshold rules → mistake classes + confidence
- [ ] `pitmind/timeloss.py` — kinematic time-loss estimate
- [ ] `pitmind/potential_lap.py` — best-sector composite lap

## 4. Coaching + Dashboard (doc v0.5)

- [ ] `pitmind/coaching.py` — template directives + priority filter
- [ ] `dashboard/app.py` — Streamlit: lap list, telemetry plots, corner table, mistake cards, overlay, potential lap

## 5. Tests & Validation

- [ ] pytest: segmentation
- [ ] pytest: corner detection
- [ ] pytest: mistake detection — synthetic laps with known mistakes must be found
- [ ] pytest: time-loss sane ranges
- [ ] Record real ACC F1-circuit laps; validate pipeline + tune thresholds in `config.yaml`

## 6. Later (roadmap v0.6+)

- [ ] Live telemetry streaming (reuse recorder reader)
- [ ] Optional LLM coaching phrasing (isolated hook)
- [ ] ML models replacing thresholds (doc §26 stages)
- [ ] Voice feedback loop