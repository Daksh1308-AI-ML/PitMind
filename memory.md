# PitMind - Project Memory

> **Purpose:** This file is the single source of truth for any agent/model working on this codebase. Read this before making any changes.

---

## What is PitMind?

AI Driver Coach for sim racing. Analyzes telemetry, detects driving mistakes, estimates lost lap time, provides coaching feedback. Primary sim: Assetto Corsa Competizione (ACC). Real F1 data via **both FastF1 and OpenF1**, converted to the same 13-column contract. Optional local-Ollama LLM callout (display-only).

**Core philosophy:** Rules/ML decide, templates/LLM phrase. The LLM is never in the decision path (`pitmind/summarize.py`).

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` (NOT pip/npm) |
| Data | NumPy, Pandas, SciPy |
| UI | Streamlit + Plotly |
| Config | PyYAML (`config.yaml`) |
| Testing | pytest (131 tests), ruff check in CI |
| Database | **None** |

---

## Directory Structure

```
PitMind/
├── pitmind/          # Core analysis package (13 modules)
│   ├── config.py         # Config loader (dataclasses from config.yaml; incl. LLMConfig)
│   ├── preprocess.py     # Resample, smooth, validate telemetry
│   ├── segmentation.py   # Lap splitting & validation
│   ├── corners.py        # Track-agnostic corner detection (chord-based)
│   ├── events.py         # Per-corner event extraction
│   ├── features.py       # Feature table builder
│   ├── reference.py      # Reference lap selection + delta computation
│   ├── mistakes.py       # Threshold-based mistake classification (6 types; StrEnum)
│   ├── timeloss.py       # Kinematic time-loss estimation
│   ├── coaching.py       # Template-based coaching directives
│   ├── potential_lap.py  # Best-sector composite theoretical best
│   └── summarize.py      # Race-engineer callout (display-only LLM + template fallback)
├── f1/               # F1 bridge: fastf1_bridge, openf1 (stdlib urllib), cli, comparison (corner + sector), live
├── synthetic/        # Synthetic data generator (GeoJSON circuits)
├── dashboard/        # Streamlit UI (7 tabs + F1 tab)
├── recorder/         # ACC shared memory -> CSV
├── tools/            # tune.py (CLI + F1 sanity), fixture_f1.py, render_readme_images.py
├── tests/            # 13 test files, 131 tests (+ live Ollama test via -m live)
├── data/             # Synthetic CSVs + vendored circuit GeoJSONs (CSVs gitignored)
├── .github/workflows/ci.yml   # ruff lint + pytest matrix
├── config.yaml       # ALL thresholds + llm block (no magic numbers in code)
├── design.md         # Design decisions
├── architect.md      # Architecture & hard rules
├── skills.md         # Agent working practices
├── docs/demo.md      # Interactive walkthrough
└── todo.md           # Progress tracker
```

---

## Data Contract (13 columns)

Both recorder and synthetic generator MUST emit this CSV schema:

| Column | Type | Description |
|---|---|---|
| `timestamp` | float | Seconds since start |
| `lap_number` | int | Completed lap counter |
| `sector` | int | 1..3 |
| `track_position` | float | 0..1 normalized |
| `speed_kmh` | float | Speed in km/h |
| `throttle` | float | 0..1 |
| `brake` | float | 0..1 |
| `steering` | float | -1..1 |
| `gear` | int | Current gear |
| `rpm` | float | Engine RPM |
| `x`, `y`, `z` | float | World coordinates (meters) |

---

## Analysis Pipeline (9 steps)

```
CSV -> preprocess -> segmentation -> corners -> events -> features
    -> reference -> mistakes -> timeloss -> coaching -> potential_lap
```

---

## Key Enums

- **Mistake types:** `EARLY_BRAKE, LATE_BRAKE, LOW_APEX_SPEED, LATE_THROTTLE, SLOW_EXIT, EXCESS_STEERING`
- **Confidence levels:** `WEAK, SIGNIFICANT, STRONG`

---

## Hard Rules (NEVER break these)

1. **No per-track geometry at analysis time.** Corner detection is derived from telemetry alone.
2. **LLM is never in the decision path.** Rules/ML decide; templates/LLM phrase.
3. **Start rule-based, graduate to ML.** Interfaces must tolerate swapping in models.
4. **Analysis quality gates coaching.** Garbage in = garbage out.
5. **Recorder format = the contract.** Same 13-column CSV from recorder and synthetic generator.
6. **No magic numbers in code.** All thresholds live in `config.yaml`.

---

## Commands

```bash
# Run all tests
uv run python -m pytest
uv run pytest tests/test_summarize.py -m live -q   # live Ollama callout test

# Lint
uv run ruff check .

# Generate synthetic data
uv run python synthetic/generator.py --track monza --laps 12

# F1 fixtures (multi-track + multi-driver)
uv run python tools/fixture_f1.py --track monza --laps 12
uv run python tools/fixture_f1.py --track monza --drivers '{"VER":1,"LEC":2,"SAI":3}'

# Run dashboard
uv run streamlit run dashboard/app.py

# F1 CLI (analyze + compare + race-engineer summary)
uv run python -m f1.cli --year 2024 --event Monza --session R --driver VER --analyze --compare --summary

# Threshold tuning / pipeline validation
uv run python -m tools.tune data/<session>.csv
uv run python -m tools.tune data/<session>.csv --f1 --summary --write

# Console scripts (after uv sync)
uv run pitmind-tune --help
uv run pitmind-f1 --help

# Regenerate README result images
uv run python tools/render_readme_images.py

# Circuit info
uv run python synthetic/circuit.py --info
uv run python synthetic/circuit.py --list
```

---

## Available Circuits

| ID | Circuit |
|---|---|
| `monza` | Autodromo Nazionale Monza (default) |
| `spa` | Circuit de Spa-Francorchamps |
| `silverstone` | Silverstone Circuit |
| `imola` | Autodromo Enzo e Dino Ferrari |
| `generic` | Legacy hand-built track |

---

## Dashboard Tabs

1. **Telemetry** -- Multi-lap overlay (speed, throttle, brake, steering, gear, RPM)
2. **Corners** -- Corner analysis table + time vs reference chart
3. **Mistakes** -- Filterable table + time loss by type/lap
4. **Potential Lap** -- Best-sector composite vs best actual
5. **Coaching** -- Prioritized directives + session report + **📣 Race Engineer** panel (LLM/template callout)
6. **Track Map** -- Circuit layout, speed heat-map, corner markers, corner overlay
7. **F1** -- Multi-driver deltas, sector bar chart, real-F1 x/y track map, corner heat-map, live race-engineer replay

---

## Roadmap (Current Status)

| Milestone | Status |
|---|---|
| M0: Validate on real ACC laps | **Open (blocker)** — needs sim rig |
| M1: FastF1 bridge (`f1/` package) | Done |
| M2: Capability-aware mistakes for F1 | Done |
| M3: Multi-driver F1 delta + dashboard tab | Done |
| M4: Live race-engineer view + heat-map overlays | Done |
| M5: LLM race-engineer callout (display-only) | Done |
| C1–C3: OpenF1, multi-track, sector comparisons | Done |
| B: CI + ruff + packaging + console scripts | Done |
| v0.6: Live telemetry streaming | Later |
| v1.0: Real-time AI race engineer with voice | Later |

---

## When Working on This Project

- Always run `uv run python -m pytest` after changes
- Always run `uv run ruff check .` after changes (CI gate)
- Never add magic numbers; use `config.yaml`
- Follow existing code style (dataclasses, type hints, docstrings)
- Tests use synthetic data with known injected mistakes -- no live ACC needed
- If changing the CSV schema, update BOTH `recorder/` and `synthetic/`
- LLM features must stay display-only; never put the LLM in the decision path
