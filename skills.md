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

1. **No per-track geometry.** Corner detection must be derived from telemetry alone.
2. **LLM is never in the decision path** — rules/ML decide, templates/LLM phrase.
3. **Same CSV schema from recorder and synthetic generator** (the contract).
4. **Track-agnostic everywhere**; all F1 circuits supported without config.
5. `config.yaml` holds all thresholds — no magic numbers in code (design.md).

## Workflow

1. Read `todo.md`, then `architect.md` → `design.md` before touching code.
2. Implement with the best model available; use sub-agents for parallel feature work.
3. Every feature gets a pytest in `tests/` that runs against synthetic data.
4. After any change: run `pytest` and, for dashboard code, `streamlit run dashboard/app.py --server.headless true`.
5. Never `drizzle push`-style unsafe DB ops — we have no DB; if one is added, follow
   generate/migrate discipline.
6. Update `todo.md` when milestones move. Keep `architect.md` current if modules change.

## Verification Commands

```text
python -m pytest                      # unit tests against synthetic data
python synthetic/generate_synthetic.py # regenerate dev laps
python recorder/record_acc.py --help  # ACC recorder usage
streamlit run dashboard/app.py        # dashboard (local dev)
```

## Guarding Against the Failure Mode

The doc's core warning (doc §34): the hard part is *understanding telemetry*, not the LLM.
If an analysis step is wrong, coaching is garbage. Therefore:

- Synthetic laps are built with **known, labeled mistakes** — the pipeline must find exactly them.
- Tune thresholds on **real recorded laps** before calling detection done.
- Dashboard must always show the evidence behind each coaching line.