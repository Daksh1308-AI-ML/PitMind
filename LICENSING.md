# Licensing & Attribution

PitMind is an **independent, non-commercial educational project**. It is an AI
driver coach for sim racing and F1 telemetry analysis, built for learning and
personal use. It is **not affiliated with, endorsed by, or sponsored by**
Formula 1, FIA, any F1 team, driver, or KUNOS/Assetto Corsa Competizione.

This file documents the third-party data sources PitMind consumes and what that
means for you when using, copying, or redistributing this project.

---

## 1. FastF1 (real F1 telemetry)

- Source: [`theOehrly/FastF1`](https://github.com/theOehrly/FastF1) — packaged under
  the **MIT License** (the library code itself).
- **Data**: FastF1 fetches official F1 timing/tracking data from
  [F1's public API](https://www.formula1.com/en/latest/tags.api). That data is
  © 2018–present Formula One World Championship Ltd and is made available
  **for personal and non-commercial use only**.
- **Implication**: The live/offline F1 telemetry path in `f1/` (bridge,
  comparison, live race-engineer view) is **educational / non-commercial only**.
  Do not use real F1 telemetry output in a commercial product.

## 2. OpenF1 (live feed)

- Source: [`openf1.org`](https://openf1.org) — an open live F1 data API.
- OpenF1 distributes F1 data for free, but the underlying data remains
  **© Formula One World Championship Ltd** and is restricted to **personal,
  non-commercial** use under the project's own educational terms.
- The `f1/live.py` live race-engineer loop is designed to accept an OpenF1-style
  source via the same source signature; the same non-commercial constraint
  applies.

## 3. Circuit / track geometry

- Track maps in the dashboard are rendered from **data-derived coordinates**
  (the `x`/`y` columns of a telemetry lap), not from proprietary GeoJSON
  circuit files. This keeps track visualisation free of third-party map data.
  See `data/circuits/README.md` for any bundled circuit assets.

## 4. Synthetic fixtures

- All `data/*.csv` fixtures used for offline tests are **generated
  programmatically** by `tools/fixture_f1.py` and other fixture generators.
  They contain **no real driver telemetry** and are free to use/modify.

---

## Summary

| Component | License / terms |
|-----------|-----------------|
| FastF1 library | MIT |
| F1 timing data (via FastF1/OpenF1) | © F1, personal & non-commercial (CC BY-NC-SA spirit) |
| OpenF1 live data | © F1, personal & non-commercial |
| Track maps | Derived from your own telemetry x/y; no proprietary GeoJSON |
| Synthetic fixtures | Generated; free |
| PitMind own code | See `LICENSE` (project's own license) |

**Bottom line**: PitMind's real-F1 features are for personal education. If you
ship PitMind commercially using real F1 data, you must replace/remove the
FastF1/OpenF1 data paths and any F1 trademarked names.
