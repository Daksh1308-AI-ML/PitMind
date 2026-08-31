# PitMind — Architecture

> AI Driver Coach: real-time race engineer for ACC that analyzes telemetry, detects mistakes,
> estimates lost time, and coaches the driver. This file documents the *intended* system structure.

## Status

- **Scope:** Offline MVP (doc roadmap v0.1–v0.5). Live telemetry (v0.6) comes later.
- **Simulator:** Assetto Corsa Competizione (ACC).
- **Data:** Recorded CSV. ACC's built-in MoTeC `.ld` export is known-fragile, so we ship our own
  shared-memory recorder instead.
- **Synthetic tracks:** the generator drives *real F1 circuit geometry* (Monza, Spa, Silverstone, …)
  from [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) (MIT; OSM-derived). Geometry
  is a **synthetic fixture only** — runtime analysis never uses it.

## System Overview

```text
                 ACC (running)
                      │  shared memory
                      ▼
          recorder/record_acc.py ────────► data/*.csv   (real laps)
          data/circuits/*.geojson ─┐
                                  ▼
          synthetic/generator.py ─┼────► data/*.csv   (dev/test laps)
          (drives real circuits)  │
                      │
                      ▼
          pitmind/  (analysis pipeline, pure numpy/pandas)
              preprocess → segmentation → corners → events
              → features → reference → mistakes → timeloss
              → coaching → potential_lap
                      │
                      ▼
          dashboard/app.py  (Streamlit + Plotly)
```

Guiding principle (from concept doc):

> **Telemetry → Understanding → Diagnosis → Coaching.**
> Don't just tell the driver what happened. Tell them what to do next.

## Core Pipeline Modules

| Module | Responsibility | Kinds of logic |
|---|---|---|
| `pitmind/preprocess.py` | Clean raw telemetry: resample to fixed Hz, smooth, fill gaps | numpy/scipy filtering |
| `pitmind/segmentation.py` | Split a recording into laps; drop invalid / out-laps | `track_position` wrap + lap counter |
| `pitmind/corners.py` | Detect corners from telemetry only (track-agnostic) | **chord-based curvature** over a resampled arc grid (robust to position jitter), brake-zone confirmation |
| `pitmind/events.py` | Extract corner metrics: brake point, entry/apex/exit speed, throttle-on point | signal logic (brake/throttle/speed transitions); `corner_features_table` |
| `pitmind/features.py` | Per-lap-per-corner feature table | thin wrapper over `events` (or merged) |
| `pitmind/reference.py` | Pick reference lap (best lap / best sector / best corner) and align laps | track-distance alignment |
| `pitmind/mistakes.py` | Classify mistakes with confidence (EARLY_BRAKING, LOW_APEX_SPEED, ...) | threshold rules (doc §14), later ML |
| `pitmind/timeloss.py` | Estimate lost seconds per corner and per lap | kinematic heuristic (MVP), later ML |
| `pitmind/coaching.py` | Convert analysis into short, actionable directives | templates now, optional LLM hook later |
| `pitmind/potential_lap.py` | Compute composite best-sector/best-corner lap | doc §16 |

*Synthetic fixture:* `synthetic/circuit.py` loads a circuit GeoJSON, projects lat/lon → meters,
resamples to a ~1 m arc grid, and **auto-derives corner regions** (start/end/apex by the same
chord-curvature logic, radius from peak curvature). `synthetic/generator.py` drives those regions
with a discrete-corner kinematic planner so per-corner mistakes can be injected with known labels.
This keeps the model simple **and** realistic enough to develop/test against.

## Data Flow (one lap)

1. Raw CSV → `preprocess` → fixed-rate, clean time series.
2. `segmentation` → one DataFrame per valid lap.
3. `corners` → list of corners with index ranges (start → apex → end).
4. `events` → per-corner metrics for every lap.
5. `reference` → alignment against the selected reference lap.
6. `mistakes` → per-corner deviations + mistake classes + confidence.
7. `timeloss` → estimated seconds lost (per corner, per lap, potential lap).
8. `coaching` → directive list ("Brake 12m later into T3").
9. `dashboard` → lap list, telemetry plots, corner table, mistake cards, overlay, potential lap.

## Tech Stack

- **Language:** Python 3.11+ (single repo).
- **Data:** numpy, pandas (polars optional later).
- **Analysis:** scipy (filtering), our own rule logic.
- **Synthetic tracks:** circuit GeoJSON from `bacinger/f1-circuits` (MIT, OSM-derived), vendored into
  `data/circuits/` with attribution; lat/lon → meters via a local equirectangular projection.
- **Recorder:** `pyaccsharedmemory` (MIT, reads ACC Physics/Graphics/Static blocks).
- **UI:** Streamlit + Plotly.
- **Config:** `config.yaml` (sample rate, thresholds, track, car).
- **Tests:** pytest against synthetic data with known injected mistakes, driven on real circuit geometry.

## Hard Architectural Rules

1. **No per-track geometry at analysis time.** Corner detection is derived from telemetry alone
   (x/y path → chord-based curvature + brake zones). Any F1 circuit works with zero config.
   Circuit geometry files exist **only** as synthetic-generator fixtures for dev/test data.
2. **LLM is not in the decision path.** ML/rules decide; templates (or later an LLM) only phrase
   the coaching. Keeps the system reliable (doc §21, §34).
3. **Start rule-based, graduate to ML.** Mistake thresholds and time-loss are heuristics in the MVP;
   the interfaces must tolerate swapping in models later (doc §26 stages).
4. **Analysis quality gates coaching.** Garbage in = garbage out. Telemetry correctness is the
   hard part, not the LLM.
5. **Recorder format = the contract.** The recorder and synthetic generator must emit the exact
   same CSV schema so the pipeline is testable without the sim.

## Roadmap Mapping

| Doc version | Content | In this repo? |
|---|---|---|
| v0.1 Telemetry Analyzer | CSV → graphs, lap stats | Yes (segmentation + dashboard basics) |
| v0.2 Corner Intelligence | corner/brake/apex/throttle + reference | Yes |
| v0.3 Mistake Detection | early/late brake, low apex, late throttle, exit | Yes |
| v0.4 Time-Loss Model | corner time predict, time loss, potential lap | Yes (heuristic) |
| v0.5 AI Coach | coaching engine + priorities + NLG | Yes (templates) |
| v0.6 Real-Time | streaming processor over live UDP/shared memory | Later |
| v1.0 Real-Time Race Engineer | voice + live dashboard loop | Later |