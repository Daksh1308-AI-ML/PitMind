# Circuit Geometry (synthetic fixtures)

GeoJSON track centerlines used by `synthetic/generator.py` to produce realistic dev/test
telemetry. **Fixtures only** — runtime analysis is track-agnostic and never reads these files
(architect.md rule 1).

## Files & IDs

| File | Circuit | GeoJSON source id |
|---|---|---|
| `monza.geojson` | Autodromo Nazionale Monza | `it-1922` |
| `spa.geojson` | Circuit de Spa-Francorchamps | `be-1925` |
| `silverstone.geojson` | Silverstone Circuit | `gb-1948` |
| `imola.geojson` | Autodromo Enzo e Dino Ferrari | `it-1953` |

Rename via `synthetic/circuit.py --info <id>` to inspect a track's derived corners.

## Source & License

- **Source:** [bacinger/f1-circuits](https://github.com/bacinger/f1-circuits) (Copyright (c)
  2019-2025 Tomislav Bacinger), **MIT License**.
- Circuit coordinates are **derived from OpenStreetMap** data and are therefore also subject to the
  **Open Database License (ODbL)** — attribution to © OpenStreetMap contributors.
- Not associated with, approved, or endorsed by Formula One Licensing B.V. or any Grand Prix rights
  holder. "Formula 1", "FIA Formula One World Championship" and related marks are trademarks of
  Formula One Licensing B.V.

Files are vendored unmodified (only renamed) from the source repository's `circuits/` directory.