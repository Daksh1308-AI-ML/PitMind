# PitMind — Skills & Working Practices

How to work in this repo (project-level agent instructions).

## Project Identity

PitMind = **AI Driver Coach**: an AI race engineer for sim racing telemetry.
This folder is greenfield — concept doc first, then the offline MVP.

## Relevant Environment Skills

| Skill | When to use |
|---|---|
| `design-system` / `ui-styling` | Building or reviewing the Streamlit dashboard UI, tokens, layout |
| `slides` | Presenting the project (portfolio, milestone demos) |
| `brand` | Any branded copy / positioning material for PitMind |

Data/ML work uses plain numpy/pandas — no heavy skill required.

## Non-Negotiables (from architect.md / design.md)

1. **No per-track geometry at analysis time.** Corner detection is derived from telemetry alone;
   circuit GeoJSON is a synthetic test fixture only.
2. **LLM is never in the decision path** — rules/ML decide, templates/LLM phrase.
3. **Same CSV schema from recorder and synthetic generator** (the contract).
4. **Track-agnostic everywhere**; all F1 circuits supported without config.
5. `config.yaml` holds all thresholds — no magic numbers in code (design.md).
6. Synthetic dev/test data drives **real circuit geometry** (`data/circuits/`, bacinger/f1-circuits);
   circuit name is a generator input, never hardcoded into analysis.
7. **Real F1 data goes through the bridge, never a fork.** FastF1/OpenF1 telemetry is converted to
   the CSV contract in `f1/`; the pipeline is unchanged. F1 has no `steering` and bool-only
   `Brake`, so mistakes uses a **capability flag** to prune steering-based classes on F1 inputs.
8. **Licensing is explicit.** FastF1/OpenF1 are educational/non-commercial (CC BY-NC-SA); keep that
   documented. "F1 official" = pro-grade analysis of public data, not an FOM-commercial license.

## Workflow

1. Read `todo.md`, then `architect.md` → `design.md` before touching code.
2. Implement with the best model available; use sub-agents for parallel feature work.
3. Every feature gets a pytest in `tests/` that runs against synthetic data on real circuits.
4. After any change: run `uv run python -m pytest` and, for dashboard code,
   `uv run streamlit run dashboard/app.py --server.headless true`.
5. Never `drizzle push`-style unsafe DB ops — we have no DB; if one is added, follow
   generate/migrate discipline.
6. Update `todo.md` when milestones move. Keep `architect.md` current if modules change.

## Verification Commands

```text
uv run python -m pytest                         # unit tests against synthetic data
uv run python synthetic/generator.py --track monza --laps 12  # regenerate dev laps
uv run python synthetic/circuit.py --info       # inspect a vendored circuit (derived corners)
uv run python recorder/record_acc.py --help     # ACC recorder usage
uv run python tools/tune.py data/<session>.csv  # validation report + threshold suggestions
uv run python f1/cli.py --year 2024 --event Monaco --session Q --driver VER  # real F1 -> pipeline
uv run streamlit run dashboard/app.py           # dashboard (local dev, incl. 🗺️ Track Map)
```

## Guarding Against the Failure Mode

The doc's core warning (doc §34): the hard part is *understanding telemetry*, not the LLM.
If an analysis step is wrong, coaching is garbage. Therefore:

- Synthetic laps are built with **known, labeled mistakes** — the pipeline must find exactly them.
- Tune thresholds on **real recorded laps** before calling detection done.
- Dashboard must always show the evidence behind each coaching line.