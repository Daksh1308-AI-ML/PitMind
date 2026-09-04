#!/usr/bin/env python3
"""Regenerate the README's 📸 Result images (Monza speed map + time-loss heat-map).

The two images under `docs/images/` referenced by README.md are rendered from real
F1-format telemetry through the exact dashboard pipeline (chord corner detection ->
feature table -> mistakes -> timeloss), using kaleido to write Plotly figures to
PNG. Re-run this whenever the map/overlay styling or the fixture changes:

    uv run python tools/render_readme_images.py

Requires kaleido (already a base dependency). No network. Offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "images"

# Monza driver fixture committed as part of the repo (gitignored? no - tracked).
DATA = ROOT / "data" / "f1_monza_driver_VER.csv"


def _build(driver_csv: Path, cfg) -> tuple[object, object]:
    import pandas as pd

    from dashboard import map_plot
    from pitmind import segmentation
    from pitmind.corners import detect_corners, track_length_m
    from pitmind.features import build_feature_table
    from pitmind.mistakes import detect_mistakes
    from pitmind.preprocess import preprocess
    from pitmind.timeloss import estimate_time_loss

    session = pd.read_csv(driver_csv)
    clean = preprocess(session, cfg)
    laps = segmentation.valid_laps(clean)
    first_lap_no = int(laps[0]["lap_number"].iloc[0])
    first_lap = clean[clean["lap_number"] == first_lap_no].reset_index(drop=True)

    corners = detect_corners(first_lap, cfg)
    L = track_length_m(laps[0])

    # pick a representative lap that shows time loss (use the last valid lap)
    lap_no = int(laps[-1]["lap_number"].iloc[0])
    lap = clean[clean["lap_number"] == lap_no].sort_values("track_position").reset_index(drop=True)

    table = build_feature_table(clean, cfg, detected_corners=corners, L=L)
    mlist = detect_mistakes(table, cfg, capabilities={"steering": False})
    tlist = estimate_time_loss(mlist, table, cfg)
    over = map_plot.corner_metric_table(table, tlist, metric="time_loss_s")

    speed_fig = map_plot.track_map_figure(
        lap, corners,
        color_by_speed=True,
        title=f"Monza (F1) - speed-coloured track map (L{lap_no})",
        ref=first_lap,
    )
    heat_fig = map_plot.corner_overlay_figure(
        lap, corners, over,
        metric="time_loss_s",
        title=f"Monza (F1) - corner time-loss heat-map (L{lap_no})",
        ref=first_lap,
    )
    return speed_fig, heat_fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render README result images.")
    parser.add_argument("--driver-csv", default=str(DATA), help="Driver bridge CSV (default Monza VER).")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory.")
    args = parser.parse_args(argv)

    from pitmind.config import Config

    cfg = Config.from_file()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_fig, heat_fig = _build(Path(args.driver_csv), cfg)
    speed_png = out_dir / "f1_monza_speed_map.png"
    heat_png = out_dir / "f1_monza_time_loss_heatmap.png"

    speed_fig.write_image(speed_png, scale=2)
    heat_fig.write_image(heat_png, scale=2)

    print(f"wrote {speed_png}")
    print(f"wrote {heat_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
