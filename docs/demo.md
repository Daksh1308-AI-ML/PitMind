# 🏎️ PitMind — interactive demo & walkthrough

This guide shows how to exercise PitMind end to end: from raw telemetry to a
race-engineer callout, all offline. CI runs the identical pipeline in
`tests/` (131 tests).

---

## 1. Generate offline F1 fixtures

Real FastF1/OpenF1 data can't be fetched offline, so we synthesize a faithful
stand-in: the real circuit geometry driven through the synthetic generator, then
reshaped into **raw FastF1 form** and bridged back through
`f1/fastf1_bridge.py` — exercising the exact F1 ingest path with no network.

```bash
# single track, single driver, contract CSV
uv run python tools/fixture_f1.py --track monza --laps 12        # -> data/f1_monza_laps.csv

# multi-driver field for comparisons (each driver gets its own RNG seed)
uv run python tools/fixture_f1.py --track monza \
  --drivers '{"VER":1,"LEC":2,"SAI":3}'
```

## 2. Run the engine from the CLI

```bash
uv run pitmind-tune data/f1_monza_laps.csv --f1

# add a race-engineer callout (deterministic template unless LLM enabled)
uv run pitmind-tune data/f1_monza_laps.csv --f1 --summary

# validate the F1 sanity contract (corner count, steering pruned, time-loss range)
uv run pitmind-tune data/f1_monza_laps.csv --f1 --f1-corners 7
```

## 3. LLM race-engineer callout (optional, display-only)

The engine is 100% deterministic — the LLM only rephrases its diagnosis.

```bash
# opt-in in config.yaml:
#   llm:  { provider: ollama, model: qwen2.5:7b, enabled: true }

uv run pitmind-tune data/f1_monza_laps.csv --f1 --summary
```

Verify the local Ollama model end-to-end (skipped unless `-m live`):

```bash
uv run pytest tests/test_summarize.py -m live -q
```

Offline / no server? It falls back to a deterministic template — same numbers,
different phrasing. The LLM is **never** in the decision path
(`pitmind/summarize.py`).

## 4. Multi-driver F1 comparison (+ sector roll-ups)

```bash
uv run python -m f1.cli --year 2024 --event Monza --session R --driver VER \
  --analyze --compare --other-driver LEC --other-driver SAI --summary
```

- Per-corner deltas vs an auto-selected reference driver.
- **S1/S2/S3 sector roll-ups** (`f1/comparison.compare_sectors`) — the
  "VER is 0.4s down in sector 2" summary.

## 5. Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Focus tabs:
- **🏁 Potential Lap** — theoretical best lap from your best corner executions.
- **📋 Coaching** → 📣 **Race Engineer** panel — LLM or template callout.
- **🏎️ F1** — driver deltas, **sector bar chart**, real x/y track map, corner
  time-loss heat-map overlay, live race-engineer replay.

## 6. Real, live F1 telemetry

Two sources, both convertible to the same 13-column contract:

| Source | Module | Live feed |
|--------|--------|-----------|
| FastF1 | `f1/fastf1_bridge.py` | `f1.live.fastf1_source(...)` |
| OpenF1 | `f1/openf1.py` (stdlib urllib) | `f1.live.openf1_source(session_key, driver_number)` |

```python
from f1.openf1 import fetch_car_data, to_contract
raw = fetch_car_data(session_key=9437, driver_number=1)   # OpenF1 REST
session = to_contract(raw)                                # -> 13-col contract
```

## 7. Regenerate the README result images

```bash
uv run python tools/render_readme_images.py   # overwrite docs/images/*.png
```

---

**Non-commercial caveat:** F1 telemetry is © Formula One World Championship Ltd
and for personal/educational use only — see `LICENSING.md`.
