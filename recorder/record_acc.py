"""PitMind ACC telemetry recorder.

Reads Assetto Corsa Competizione's shared memory via `pyaccsharedmemory` and
writes telemetry to disk in the same CSV schema as the synthetic generator
(design.md "CSV Telemetry Schema (the contract)").

The recorder maps ACC shared-memory fields onto our fixed column names:
  timestamp   <- Graphics.clock        (seconds, monotonic across session)
  lap_number  <- Graphics.completed_lap + 1  (1-based, first full lap = 1)
  sector      <- Graphics.current_sector_index + 1  (1..3)
  track_position <- Graphics.normalized_car_position (0..1)
  speed_kmh   <- Physics.speed_kmh
  throttle    <- Physics.gas   (0..1)
  brake       <- Physics.brake (0..1)
  steering    <- Physics.steer_angle (normalized to -1..1)
  gear        <- Physics.gear
  rpm         <- Physics.rpm
  x, y, z     <- Graphics.car_coordinates[player_car_id]  (world meters)

Requires the optional "recorder" extra:  pip install -e ".[recorder]"
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pitmind.config import Config

RECORDER_COLUMNS = [
    "timestamp",
    "lap_number",
    "sector",
    "track_position",
    "speed_kmh",
    "throttle",
    "brake",
    "steering",
    "gear",
    "rpm",
    "x",
    "y",
    "z",
]


class ACCRecorder:
    """Poll ACC shared memory and buffer telemetry rows.

    Usage (typically run while ACC is in a live session):
        rec = ACCRecorder()
        rec.start()          # begins buffering
        input("press enter to stop...")
        rec.stop()
        rec.save("data/recorded_session.csv")
    """

    def __init__(self, sample_rate_hz: float = 60.0):
        try:
            from pyaccsharedmemory import accSharedMemory
        except ImportError as exc:  # pragma: no cover - import guard for non-recorder envs
            raise ImportError(
                "pyaccsharedmemory not installed. Run: pip install -e '.[recorder]'"
            ) from exc
        self._asm = accSharedMemory()
        self._rate_hz = float(sample_rate_hz)
        self._running = False
        self._rows: list[dict] = []
        try:
            cfg = Config.from_file()
            self._rate_hz = float(cfg.detection.sample_rate_hz)
        except Exception:
            pass

    # ---- public API ----
    def read(self) -> dict | None:
        """Read one telemetry sample mapped to the CSV contract, or None."""
        sm = self._asm.read_shared_memory()
        if sm is None:
            return None
        return self._map(sm)

    def start(self, duration_s: float | None = None) -> list[dict]:
        """Poll in a loop, buffering samples, until stopped or duration elapses.

        Poll interval respects the configured sample rate. Returns the buffered rows.
        """
        self._running = True
        self._rows = []
        period = 1.0 / self._rate_hz
        deadline = None if duration_s is None else time.monotonic() + duration_s

        while self._running:
            row = self.read()
            if row is not None:
                self._rows.append(row)
            # pacing: sleep in small slices so Ctrl-C / stop() responds promptly
            slept = 0.0
            while slept < period and self._running:
                time.sleep(min(0.01, period - slept))
                slept += 0.01
            if deadline is not None and time.monotonic() >= deadline:
                break
        return self._rows

    def stop(self) -> None:
        """Stop the polling loop (safe to call from another thread/KeyboardInterrupt)."""
        self._running = False
        self._asm.close()

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the buffered rows as a DataFrame in the fixed column order."""
        return pd.DataFrame(self._rows, columns=RECORDER_COLUMNS)

    def save(self, path: str | Path, split_laps: bool = False) -> Path:
        """Write the session CSV (and optionally per-lap CSVs) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        if split_laps:
            self._save_lap_splits(df, path)
        return path

    # ---- internals ----
    def _map(self, sm) -> dict:
        phys = sm.Physics
        gfx = sm.Graphics

        # player world position: pick this car's entry out of the full grid
        x = y = z = float("nan")
        car_ids = gfx.car_id
        coords = gfx.car_coordinates
        player_id = gfx.player_car_id
        if car_ids and coords:
            pid = int(player_id)
            try:
                idx = car_ids.index(pid)
                v = coords[idx]
                x, y, z = float(v.x), float(v.y), float(v.z)
            except (ValueError, IndexError, AttributeError):
                pass

        return {
            "timestamp": float(getattr(gfx, "clock", 0.0)),
            "lap_number": int(getattr(gfx, "completed_lap", 0)) + 1,
            "sector": int(getattr(gfx, "current_sector_index", 0) or 0) + 1,
            "track_position": float(getattr(gfx, "normalized_car_position", 0.0)),
            "speed_kmh": float(getattr(phys, "speed_kmh", 0.0)),
            "throttle": _bounded(float(getattr(phys, "gas", 0.0))),
            "brake": _bounded(float(getattr(phys, "brake", 0.0))),
            "steering": _bounded(float(getattr(phys, "steer_angle", 0.0))),
            "gear": int(getattr(phys, "gear", 0)),
            "rpm": float(getattr(phys, "rpm", 0.0)),
            "x": x,
            "y": y,
            "z": z,
        }

    def _save_lap_splits(self, df: pd.DataFrame, session_path: Path) -> None:
        # reuse the pipeline's lap segmentation to write one CSV per valid lap
        from pitmind import segmentation

        laps = segmentation.valid_laps(df)
        out_dir = session_path.with_name(session_path.stem + "_laps")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, lap in enumerate(laps, start=1):
            lap.to_csv(out_dir / f"lap_{i:02d}.csv", index=False)


def _bounded(v: float) -> float:
    return max(-1.0, min(1.0, v))


def main(argv=None) -> int:
    """CLI entry: record a session (Ctrl-C to stop)."""
    import argparse

    parser = argparse.ArgumentParser(description="PitMind ACC telemetry recorder")
    parser.add_argument(
        "-o", "--output", default="data/recorded_session.csv",
        help="Output CSV path (default: data/recorded_session.csv)",
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=None,
        help="Record for this many seconds, then stop. Default: until Ctrl-C.",
    )
    parser.add_argument(
        "-r", "--rate", type=float, default=None,
        help="Sample rate Hz (default: config.yaml detection.sample_rate_hz)",
    )
    parser.add_argument(
        "--split-laps", action="store_true",
        help="Also write one CSV per valid lap into <output stem>_laps/",
    )
    args = parser.parse_args(argv)

    rate = args.rate
    rec = ACCRecorder(rate) if rate else ACCRecorder()

    print(f"Recording to {args.output} @ ~{rec._rate_hz:.0f} Hz. "
          f"Press Ctrl-C to stop.")

    try:
        rec.start(duration_s=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()

    df = rec.to_dataframe()
    if df.empty:
        print("No telemetry captured (is ACC running a live session?).")
        return 1

    rec.save(args.output, split_laps=args.split_laps)
    print(f"Saved {len(df)} samples to {args.output} "
          f"({df['lap_number'].nunique()} lap number(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
