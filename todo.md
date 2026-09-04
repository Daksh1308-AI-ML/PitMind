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
- [x] **Synthetic track → real F1 circuits** (bacinger/f1-circuits, MIT/OSM):
  - [x] Vendor GeoJSON circuit data into `data/circuits/` — Monza (default), Spa,
        Silverstone (+ optional Imola) — plus `data/circuits/README.md` attribution
  - [x] `synthetic/circuit.py` — load GeoJSON → project lat/lon → resample ~1 m arc grid
        → auto-derive corner regions (reuse chord-curvature logic) → `Track`/`Corner` list
  - [x] Refactor `synthetic/generator.py` `build_track()` to source geometry from a real circuit
        (default `monza`); steering/curvature from the centerline curvature profile
  - [x] Fix the legacy generic track's open-loop seam (keep as `generic` fallback fixture)
  - [x] Regenerate `data/synthetic_*.csv` + ground truth on the new track

## 2. Core Pipeline

- [x] `pitmind/corners.py` — track-agnostic corner detection
      (chord-based curvature + brake-zone confirmation; 4000-pt arc grid)
- [x] `pitmind/events.py` — brake point, entry/apex/exit speed, throttle-on point
      (`corner_features_table`)
- [x] `tests/test_corners.py` — **circuit-agnostic**: corner count from GT, position matching,
      drop invalid "total turn = 360°" check, add track-module tests
      (loop closure, length ≈ official, both turn directions)
- [x] `pitmind/reference.py` — pick reference lap (best) + per-corner reference feature values
- [x] `pitmind/features.py` — thin wrapper around `events.corner_features_table` + reference deltas

## 3. Intelligence (doc v0.3–v0.4)

- [x] `pitmind/mistakes.py` — threshold rules → mistake classes + confidence
- [x] `pitmind/timeloss.py` — kinematic time-loss estimate
- [x] `pitmind/potential_lap.py` — best-sector composite lap

## 4. Coaching + Dashboard (doc v0.5)

- [x] `pitmind/coaching.py` — template directives + priority filter
- [x] `dashboard/app.py` — Streamlit: lap list, telemetry plots, corner table, mistake cards,
      overlay, potential lap

## 5. Tests & Validation

- [x] pytest: corner detection (circuit-agnostic) + `synthetic/circuit.py`
- [x] pytest: mistake detection — synthetic laps with known mistakes must be found
- [x] pytest: time-loss sane ranges
- [x] `recorder/record_acc.py` — ACC shared memory → CSV (13-col contract); optionally
      splits into per-lap CSVs reusing `pitmind/segmentation`; CLI + `pytest` (7 tests)
- [x] `tools/tune.py` — turnkey validation + threshold tuning CLI:
      runs full pipeline on any recorded/synthetic CSV, prints corner/mistake/time-loss/
      coaching report, recommends config thresholds to hit a target flag-rate,
      `--write` applies them to `config.yaml` (tests: 5)
- [ ] Record real ACC F1-circuit laps; `python -m tools.tune data/<session>.csv --write` to validate + tune

## 6. Later (roadmap v0.6+)

- [ ] Live telemetry streaming (reuse recorder reader)
- [ ] Optional LLM coaching phrasing (isolated hook)
- [ ] ML models replacing thresholds (doc §26 stages)
- [ ] Voice feedback loop

## 7. Dashboard — Circuit Track Map (current)

Purpose: render the actual circuit layout (a "track image") in the dashboard by plotting
each lap's `x, y` world coordinates colored by speed (heat-map), with detected-corner apex
markers and a start/finish line. Works for BOTH synthetic and real recorded CSVs (same
`x, y` source — the shape comes from the driven path, never from a GeoJSON, keeping
architect.md rule 1).

- [x] `dashboard/map_plot.py` — helper: re-center/re-scale `x,y` (axis in meters from track
      centre, equal aspect) → Plotly figure of the lap path colored by `speed_kmh`,
      corner apex markers (`T1..Tn`), start/finish marker
- [x] `dashboard/app.py` — add a `🗺️ Track Map` tab (lap selector + optional "color by speed")
- [x] `tests/test_map_plot.py` — assert figure structure (path + corner markers + centering)
- [x] Verdict: `python -m pytest` green (60 pass); Track Map renders in `streamlit run
      dashboard/app.py` (confirmed working)

## 8. F1-Grade Engine (roadmap — reach real Formula 1)

Goal: make PitMind's track-agnostic analysis core (corners → mistakes → timeloss → coaching)
run on **real Formula 1 telemetry** via FastF1 (and optionally OpenF1), so it analyses actual
F1 sessions with the same race-engineering depth it gives ACC laps. "F1 official" here means
**professional-grade on real F1 data** — NOT an FOM-commercial license (that is a separate
business path, not engineering). The whole approach relies on architect.md rule 1
(track-agnostic: corners from the x/y path), which was built for exactly this portability.

- [ ] **M0 — validate on real ACC first** (gate before trusting F1):
      record realistic ACC laps → `python -m tools.tune data/<session>.csv --write`
      (todo §5's open blocker)
- [x] **M1 — F1 data bridge** (`f1/` package):
      `f1/fastf1_bridge.py` → convert a FastF1 session to the 13-col CSV contract:
      - Map `Speed`→`speed_kmh`, `Throttle`(0–100)→`throttle`(0–1), `Brake`(bool)→`brake`(0/1),
        `nGear`→`gear`, `RPM`→`rpm`, `X/Y/Z`(1/10 m)→`x/y/z`(meters)
      - `lap_number` per FastF1 lap; `sector` 1..3 from lap sector times
      - `track_position` ← normalized cumulative arc length of the x/y path
        (the one non-trivial derivation — document it)
  - [x] `f1/cli.py` — `python -m f1.cli --year 2024 --event Monaco --session Q --driver VER`
        (with `--analyze` to run the full pipeline; `-o` to write the bridged CSV)
  - [x] `tests/test_fastf1_bridge.py` — offline tests on a committed telemetry fixture
        (synthetic FastF1-shaped frames; unit conversions, track_position, lap numbers,
        capability flag prunes EXCESS_STEERING) — 10 passing
  - [x] add `fastf1` as an optional extra `[f1]` in `pyproject.toml`
  - [x] `pitmind/mistakes.py` — `capabilities={"steering": False}` prunes EXCESS_STEERING
        (M2 groundwork landed early)
- [x] **M2 — make the engine trust F1 data**:
      `pitmind/mistakes.py` capability flag (F1 has NO `steering`, Brake is boolean)
      → prune `EXCESSIVE_STEERING`/steering-based classes on F1; verify
      `EARLY/LATE_BRAKING`, `LOW_APEX_SPEED`, `LATE_THROTTLE`, `POOR_CORNER_EXIT` still fire;
      extend `tools/tune.py` to run the pipeline on a real F1 lap set and sanity-check
      corner counts (Monza ~7) + time-loss ranges
  - [x] `pitmind/mistakes.py` `detect_mistakes(..., capabilities={"steering": False})`
        prunes `EXCESS_STEERING` (landed in M1; verified in M2 tests)
  - [x] `tools/tune.py --f1` — F1 mode: threads `capabilities` through `run_pipeline`,
        runs `sanity_check_f1` (corner count near expected, steering capability off,
        time-loss within sane range); `--f1-corners <n>` sets the expected count
        (Monza ~7 default; spa ~19, silverstone ~17)
  - [x] `tools/fixture_f1.py` — synthetic F1 Monza lap set (`data/f1_monza_laps.csv`):
        generator output reshaped to raw FastF1 form (bool brake, NO steering,
        Throttle 0-100 %, X/Y/Z in 1/10 m, ~4 Hz) then re-bridged via `f1/` —
        exercises the real F1 ingest path fully offline; steering-NaN-capable
  - [x] `tests/test_tune_f1.py` — 7 tests: fixture is F1-shaped, pipeline prunes
        steering + keeps F1 mistake classes, Monza corner count ~7, sanity checks pass
        on Monza + fail on wrong corner count/capability, CLI `--f1` runs end-to-end
  - [x] fixed `tools/tune.py` threshold diagnostics to handle NaN steering channel
        (all-NaN metric rows now print `n/a` instead of crashing: missing p50/p85/flag keys)
- [x] **M3 — F1 comparisons + dashboard**:
      multi-driver delta ("VER loses 0.22s to LEC in T7 by braking 14 m early") using
      existing reference/timeloss; add a `🏎️ F1` dashboard tab reusing the track map on
      real F1 x/y
  - [x] `f1/comparison.py` — multi-driver delta module:
        `driver_corner_table` (per-corner time loss + dominant mistake) →
        `compare_drivers({code: session}, cfg)` aligns corners by index across drivers
        (track-agnostic, all same circuit) and produces per-corner `CornerDelta`
        (delta vs reference driver + `root_cause` phrase like "braking 31.3m late");
        reference driver auto = lowest total corner loss, or pass explicitly;
        `phrase_delta` renders the race-engineer summary "VER loses X.XXs to SAI ..."
  - [x] `tools/fixture_f1.py --drivers` — multi-driver F1 fixture: one bridged CSV per
        driver (`data/f1_monza_driver_{VER,LEC,SAI}.csv`), each with its own RNG seed so
        mistake profiles differ; exercises the real F1 ingest path offline
  - [x] `tests/test_comparison.py` — 7 tests: corner-table shape (Monza ≈7), compares all
        non-reference drivers, auto-reference selection, delta sign correctness, worst-corners
        ordering, table flatten, phrase readability, <2-driver guard
  - [x] `dashboard/app.py` — new `🏎️ F1` tab: loads the multi-driver fixture, shows per-driver
        delta phrases, renders the circuit track map on real F1 x/y via `map_plot`
        (track-agnostic, no GeoJSON), and a corner-by-corner delta table
  - [x] `f1/cli.py --compare --other-driver CODE` — fetch + bridge a small field and print the
        multi-driver comparison report
- [x] **M4 — live + polish**:
      live race-engineer view (FastF1 livetiming / OpenF1, ~3 s delay);
      multi-metric corner heat-map overlays on the track map;
      licensing README (FastF1/OpenF1 are educational/non-commercial — CC BY-NC-SA)
  - [x] `f1/live.py` — live race-engineer loop: `engineer_loop(source, emit, cfg)` polls an
        injectable source (callable -> bridged slice or None), runs the analysis with F1
        capabilities (steering pruned) and emits a `LiveLap` per fresh slice; graceful
        `_error_lap` on a corrupt slice; `fastf1_source()` wraps the real FastF1/OpenF1
        feed via the same signature for a live (~3 s) view. Fully offline-testable.
  - [x] `dashboard/map_plot.py` — `corner_overlay_figure()` multi-metric corner heat-map
        (bubbles coloured+sized by a chosen metric, RdYlGn, colorbar) + `corner_metric_table()`
        (per-corner time-loss sum or per-corner feature mean); track-agnostic, no GeoJSON.
  - [x] `dashboard/app.py` — F1 tab now has: a metric selector for the corner heat-map overlay
        (`load_f1_overlay_metric`), and a "Live race-engineer view" replay that streams the
        selected driver's laps through `f1.live.engineer_loop` and prints the feed.
  - [x] `LICENSING.md` — FastF1 (MIT lib / © F1 non-commercial data), OpenF1 (© F1
        non-commercial), track maps derive from own x/y (no proprietary GeoJSON),
        synthetic fixtures are generated; F1-non-affiliation clarification.
  - [x] `tests/test_live.py` — 7 tests (analyze-session F1 caps, zero-delta lone lap,
        loop emits each new slice, skips no-new-data, error-handling, severity map, defaults)
  - [x] `tests/test_map_plot.py` — +6 overlay tests (time-loss agg, apex metric, figure
        structure + colorbar + equal aspect, mismatched-corner reject, unknown-metric reject)

### Open decisions (not yet resolved)
- Data source: FastF1 only, or also OpenF1 as a live secondary source?
- Commit a small real F1 telemetry JSON fixture into `tests/fixtures/` for offline tests?
  → Resolved (M2): a **synthetic F1-shaped** fixture (`data/f1_monza_laps.csv`, via
  `tools/fixture_f1.py`) stands in offline; a real F1 fixture would need a license-legal,
  small committed slice and can be added when M4's licensing README lands.
- Implementation cadence: M0+M1 together (first real F1 analysis), or A–D in one pass?
  → M1 and M2 done as separate passes; M0 still blocked on real-recorded ACC laps.