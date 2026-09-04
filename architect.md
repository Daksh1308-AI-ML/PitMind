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
          F1 (public telemetry)   │
              FastF1 / OpenF1     │
                      │           │
                      ▼           │
          f1/fastf1_bridge.py ────┼────► session DataFrame  (real F1 laps)
                      │
                      ▼
          pitmind/  (analysis pipeline, pure numpy/pandas)
              preprocess → segmentation → corners → events
              → features → reference → mistakes → timeloss
              → coaching → potential_lap
                      │
                      ▼
          dashboard/app.py  (Streamlit + Plotly)
              └─ folders: telemetry · corners · mistakes ·
                 potential lap · coaching · 🗺️ track map · 🏎️ F1
                      │
          tools/tune.py  (CLI: full-pipeline validation report +
                          config threshold tuning, `--write`)
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
| `pitmind/coaching.py` | Convert analysis into short, actionable directives | templates (priority-ordered) |
| `pitmind/summarize.py` | **Race-engineer callout** (display-only): rephrase the diagnosis | `TemplateEngineer` (deterministic) ⇄ `OllamaEngineer` (local `qwen2.5:7b`) with automatic fallback — **never in the decision path** |
| `pitmind/potential_lap.py` | Compute composite best-sector/best-corner lap | doc §16 |

*Synthetic fixture:* `synthetic/circuit.py` loads a circuit GeoJSON, projects lat/lon → meters,
resamples to a ~1 m arc grid, and **auto-derives corner regions** (start/end/apex by the same
chord-curvature logic, radius from peak curvature). `synthetic/generator.py` drives those regions
with a discrete-corner kinematic planner so per-corner mistakes can be injected with known labels.
This keeps the model simple **and** realistic enough to develop/test against.

### Data capture & validation tooling

- `recorder/record_acc.py` — reads ACC's shared memory (`pyaccsharedmemory`) and writes telemetry
  in the exact same CSV contract as the synthetic generator; optionally splits into per-lap CSVs
  via `pitmind/segmentation`. Field mapping: `Graphics.normalized_car_position` → `track_position`,
  `Graphics.car_coordinates[player_car_id]` → `x, y, z` (world meters, feeds corner detection),
  `Graphics.completed_lap + 1` → `lap_number`, `Graphics.clock` → `timestamp`, Physics fields for
  speed/throttle/brake/steering/gear/rpm.
- `tools/tune.py` — runs the full pipeline on any recorded/synthetic CSV and prints a validation
  report (corners, mistakes, time loss, potential lap, coaching), then recommends `config.yaml`
  thresholds to hit a target flag-rate (`--write` applies them). Thresholds are only tuned on
  **real** laps before calling detection done.

### Dashboard track map

`dashboard/map_plot.py` renders the actual circuit layout from a lap's `x, y` path (re-centred to
meters from track centre, equal aspect) colored by `speed_kmh`, with detected-corner apex markers
and a start/finish line. Because corners/layout come from the driven path (never a GeoJSON), this
works identically for synthetic data and real recorded ACC laps.

### F1 data bridge (`f1/`)

Extends the same contract to **real Formula 1 telemetry**. `f1/fastf1_bridge.py` converts a FastF1
session (public F1 timing/telemetry) into the exact 13-column session DataFrame the pipeline
consumes, so the full analysis core runs on real F1 drivers with zero analysis changes.
`f1/openf1.py` is a dependency-light (stdlib `urllib`) *alternative* source that maps OpenF1's
`car_data` rows to the same contract — both feed the identical analysis, and
`f1/live.openf1_source()` exposes OpenF1 as a live feed under the same signature as FastF1:

- Field mapping: `Speed`→`speed_kmh`; `Throttle` (0–100 %)→`throttle` (0–1, ÷100);
  `Brake` (bool)→`brake` (0/1 float); `nGear`→`gear`; `RPM`→`rpm`;
  `X`/`Y`/`Z` (1/10 m)→`x`/`y`/`z` (meters, ÷10).
- `lap_number` is taken directly from FastF1's per-lap telemetry; `sector` 1..3 from lap sector
  times; `timestamp` from session time.
- `track_position` is **synthesized** as the normalized cumulative arc length of the x/y path —
  the one non-trivial derivation (F1 broadcasts no `track_position`), and the basis on which
  corner detection still works despite ~4 Hz sampling (the chord-curvature logic resamples onto
  its own 4000-pt arc grid).
- **Capability limits:** F1 broadcasts no `steering` channel and `Brake` is boolean. `mistakes.py`
  therefore needs a channels/capability flag to prune `EXCESSIVE_STEERING` (and any
  steering-dependent logic) on F1 inputs while keeping full behaviour for ACC. Brake-point,
  apex-speed, throttle-on and exit-speed mistakes all still work on F1 data.
- License note: FastF1/OpenF1 are **educational / non-commercial** (CC BY-NC-SA). "F1 official"
  here means professional-grade analysis of public F1 data — as opposed to an FOM-commercial
  license, which is a separate business thread (out of scope for the engineering roadmap).

## Roadmap: "Reach F1" mapping

| Milestone | Content | Status |
|---|---|---|
| M0 | Validate models on **real recorded ACC laps** (`tools/tune.py --write`) | open (todo §5 blocker — needs sim rig) |
| M1 | `f1/fastf1_bridge.py` + CLI + contract tests on a committed fixture | done |
| M2 | Capability-aware mistakes + F1 sanity validation (corner counts, time-loss ranges) | done |
| M3 | Multi-driver F1 delta + `🏎️ F1` dashboard tab (reuses track map) | done |
| M4 | Live race-engineer view + corner heat-map overlays + licensing README | done |
| M5 | LLM race-engineer callout (display-only, Ollama + template fallback) | done |
| C1–C3 | OpenF1 client/bridge, multi-track validation, sector comparisons | done |
| B | CI, ruff, packaging + console scripts, badges, demo docs | done |
| **suite** | 131 tests passing + `ruff check` green | done |

## Data Flow (one lap)

1. Raw CSV → `preprocess` → fixed-rate, clean time series.
2. `segmentation` → one DataFrame per valid lap.
3. `corners` → list of corners with index ranges (start → apex → end).
4. `events` → per-corner metrics for every lap.
5. `reference` → alignment against the selected reference lap.
6. `mistakes` → per-corner deviations + mistake classes + confidence.
7. `timeloss` → estimated seconds lost (per corner, per lap, potential lap).
8. `coaching` → directive list ("Brake 12m later into T3").
9. `dashboard` → lap list, telemetry plots, corner table, mistake cards, overlay, potential lap,
   and a **circuit track map** (lap `x,y` path colored by speed + corner apex markers +
   start/finish line).
10. `tools/tune.py` → same pipeline as a CLI that prints a validation report and recommends
    `config.yaml` threshold values (flag-rate based), optionally writing them back (`--write`).

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
2. **LLM is not in the decision path.** ML/rules decide; the LLM only phrases the
   coaching (implemented as `pitmind/summarize.py`, display-only, with a deterministic
   template fallback — so it's never a correctness risk) (doc §21, §34).
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