"""PitMind F1 CLI.

Downloads a real Formula 1 session via FastF1, converts it to the 13-column
CSV contract through `fastf1_bridge.py`, and (optionally) runs the full PitMind
analysis pipeline over it (corners -> mistakes -> timeloss -> coaching).

Requires the optional "f1" extra:  pip install -e ".[f1]"

Examples:
    python -m f1.cli --year 2024 --event Monaco --session Q --driver VER
    python -m f1.cli --year 2024 --event Monaco --session Q --driver VER --analyze
    python -m f1.cli --year 2024 --event Monaco --session Q --driver VER --out data/monaco_ver.csv
"""

from __future__ import annotations

import argparse

import pandas as pd


def fetch_session(
    year: int,
    event: str,
    session: str,
    driver: str,
) -> list[pd.DataFrame]:
    """Download a FastF1 session and return one telemetry frame per lap for a driver."""
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover - env guard
        raise ImportError(
            "fastf1 not installed. Run: pip install -e '.[f1]'"
        ) from exc

    fastf1.Cache.enable_cache("f1/_cache")
    sess = fastf1.get_session(year, event, session)
    sess.load(laps=True, telemetry=True)

    laps = sess.laps.pick_driver(driver)
    frames: list[pd.DataFrame] = []
    for _, lap in laps.iterrows():
        data = lap.get_car_data().add_distance()
        if data is not None and len(data) > 0:
            frames.append(data)
    return frames


def run_pipeline(session_df: pd.DataFrame) -> None:
    """Run the PitMind analysis pipeline over a bridged session (print report)."""
    from pitmind import coaching, features, mistakes, timeloss
    from pitmind.config import Config

    cfg = Config.from_file()
    try:
        table = features.build_feature_table(session_df, cfg)
    except ValueError as exc:
        print(f"[pipeline] could not build feature table: {exc}")
        return
    found = mistakes.detect_mistakes(table, cfg, capabilities={"steering": False})
    losses = timeloss.estimate_time_loss(found, table, cfg)
    directives = coaching.generate_directives(found, losses)

    print(f"\n=== F1 analysis ({len(session_df['lap_number'].unique())} lap(s)) ===")
    print(f"corner rows: {len(table)}")
    print(f"mistakes:    {len(found)}")
    for m in found[:20]:
        print(f"  {m.message}")
    print(f"coaching directives: {len(directives)}")
    for d in directives[:10]:
        print(f"  - {d.message}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="PitMind F1 bridge -- real F1 telemetry -> pipeline",
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2024")
    parser.add_argument("--event", required=True, help="Event/GP name, e.g. Monaco")
    parser.add_argument("--session", required=True, help="Session id, e.g. Q, R, FP1")
    parser.add_argument("--driver", required=True, help="Driver code, e.g. VER")
    parser.add_argument(
        "--analyze", action="store_true",
        help="Also run the full PitMind analysis pipeline over the bridged session",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare the main driver against one or more --other-driver codes "
             "(multi-driver delta, todo.md M3)",
    )
    parser.add_argument(
        "--other-driver", action="append", default=[],
        help="Another driver code to compare against (repeatable). Requires --compare.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a race-engineer callout for the analysed session (local Ollama "
             "qwen2.5:7b; template fallback — todo.md M5)",
    )
    parser.add_argument(
        "-o", "--out", default=None,
        help="Write the bridged session to this CSV path (13-column contract)",
    )
    args = parser.parse_args(argv)

    from f1.fastf1_bridge import FastF1Bridge, bridge_to_csv

    print(f"Fetching {args.event} {args.session} {args.driver} ({args.year}) ...")
    frames = fetch_session(args.year, args.event, args.session, args.driver)
    if not frames:
        print("No telemetry retrieved. Is fastf1 cached / the session available?")
        return 1

    bridge = FastF1Bridge()
    session = bridge.convert_session(frames)
    print(f"Converted {len(frames)} lap(s) -> {len(session)} samples "
          f"({list(session.columns)})")

    if args.out:
        bridge_to_csv(session, args.out)
        print(f"Wrote {args.out}")

    if args.analyze:
        run_pipeline(session)

    if args.summary:
        from pitmind import summarize
        from pitmind.config import Config as _Config
        from tools import tune as _tune
        _cfg = _Config.from_file()
        _bundle = _tune.run_pipeline(session, _cfg, capabilities={"steering": False})
        print("\n=== Race engineer ===")
        print(summarize.engineer_callout(
            _bundle, _cfg, driver=args.driver, capabilities={"steering": False},
            force=True))

    if args.compare:
        if not args.other_driver:
            print("--compare requires at least one --other-driver CODE")
            return 1
        _run_comparison(args, bridge)

    return 0


def _run_comparison(args, bridge) -> None:
    """Fetch + bridge the other drivers and print the M3 comparison report."""
    from f1 import comparison
    from pitmind.config import Config

    sessions: dict[str, pd.DataFrame] = {}
    for code in [args.driver, *args.other_driver]:
        print(f"Fetching {code} ({args.event} {args.session}, {args.year}) ...")
        frames = fetch_session(args.year, args.event, args.session, code)
        if not frames:
            print(f"  no telemetry for {code}; skipping")
            continue
        sessions[code] = bridge.convert_session(frames)

    if len(sessions) < 2:
        print("Not enough drivers with telemetry to compare.")
        return

    comps = comparison.compare_drivers(sessions, Config.from_file())
    print("\n=== F1 multi-driver comparison ===")
    for c in comps:
        print(comparison.phrase_delta(c))
    table = comparison.comparison_to_dataframe(comps)
    print("\nCorner-by-corner deltas:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
