# PitMind - Project Memory

> **Purpose:** This file is the single source of truth for any agent/model working on this codebase. Read this before making any changes.

---

## What is PitMind?

AI Driver Coach for sim racing. Analyzes telemetry, detects driving mistakes, estimates lost lap time, provides coaching feedback. Primary sim: Assetto Corsa Competizione (ACC). Real F1 data via FastF1 (planned).

**Core philosophy:** Rules/ML decide, templates/LLM phrase. The LLM is never in the decision path.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` (NOT pip/npm) |
| Data | NumPy, Pandas, SciPy |
| UI | Streamlit + Plotly |
| Config | PyYAML (`config.yaml`) |
| Testing | pytest (60 tests) |
| Database | **None** |

---

## Directory Structure

```
PitMind/
├── pitmind/          # Core analysis package (12 modules)
│   ├── config.py         # Config loader (dataclasses from config.yaml)
│   ├── preprocess.py     # Resample, smooth, validate telemetry
│   ├── segmentation.py   # Lap splitting & validation
│   ├── corners.py        # Track-agnostic corner detection (chord-based)
│   ├── events.py         # Per-corner event extraction
│   ├── features.py       # Feature table builder
│   ├── reference.py      # Reference lap selection + delta computation
│   ├── mistakes.py       # Threshold-based mistake classification (6 types)
│   ├── timeloss.py       # Kinematic time-loss estimation
│   ├── coaching.py       # Template-based coaching directives
│   └── potential_lap.py  # Best-sector composite theoretical best
├── synthetic/        # Synthetic data generator (GeoJSON circuits)
├── dashboard/        # Streamlit UI (6 tabs)
├── recorder/         # ACC shared memory -> CSV
├── tools/            # Pipeline validator + threshold tuner CLI
├── tests/            # 6 test files, 60 tests
├── data/             # Synthetic CSVs + vendored circuit GeoJSONs
├── config.yaml       # ALL thresholds (no magic numbers in code)
├── design.md         # Design decisions
├── architect.md      # Architecture & hard rules
├── skills.md         # Agent working practices
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

# Generate synthetic data
uv run python synthetic/generator.py --track monza --laps 12

# Run dashboard
uv run streamlit run dashboard/app.py

# Threshold tuning
uv run python -m tools.tune data/<session>.csv
uv run python -m tools.tune data/<session>.csv --write

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
5. **Coaching** -- Prioritized directives + session report
6. **Track Map** -- Circuit layout, speed heat-map, corner markers

---

## Roadmap (Current Status)

| Milestone | Status |
|---|---|
| M0: Validate on real ACC laps | **Open (blocker)** |
| M1: FastF1 bridge (`f/` package) | Open |
| M2: Capability-aware mistakes for F1 | Open |
| M3: Multi-driver F1 delta + dashboard tab | Open |
| M4: Live race-engineer view | Open |
| v0.6: Live telemetry streaming | Later |
| v1.0: Real-time AI race engineer with voice | Later |

---

## When Working on This Project

- Always run `uv run python -m pytest` after changes
- Never add magic numbers; use `config.yaml`
- Follow existing code style (dataclasses, type hints, docstrings)
- Tests use synthetic data with known injected mistakes -- no live ACC needed
- If changing the CSV schema, update BOTH `recorder/` and `synthetic/`
