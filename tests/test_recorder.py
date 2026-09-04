"""Tests for the ACC shared-memory recorder (field mapping + CSV output).

These run without a live ACC session by mocking the shared-memory reader, so
the 13-column CSV contract and per-lap splitting are verified on CI.
"""

from __future__ import annotations

import pandas as pd

from recorder.record_acc import RECORDER_COLUMNS, ACCRecorder, _bounded


# ---------------------------------------------------------------- fakes ----
class _Vec3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeGraphics:
    car_id = [0, 1, 2]
    car_coordinates = [_Vec3(1, 2, 3), _Vec3(100, 200, 300), _Vec3(4, 5, 6)]
    player_car_id = 1
    clock = 123.4
    completed_lap = 5
    current_sector_index = 1
    normalized_car_position = 0.567


class _FakePhysics:
    speed_kmh = 250.0
    gas = 0.9
    brake = 0.0
    steer_angle = -0.4
    gear = 6
    rpm = 9500.0


class _FakeMap:
    Physics = _FakePhysics()
    Graphics = _FakeGraphics()
    Static = None


class _FakeReader:
    """Stand-in for pyaccsharedmemory.accSharedMemory."""
    def __init__(self, samples):
        self._samples = list(samples)
        self._i = 0
        self.closed = False

    def read_shared_memory(self):
        if self._i < len(self._samples):
            s = self._samples[self._i]
            self._i += 1
            return s
        return None

    def close(self):
        self.closed = True


class _FakeAVec:
    x, y, z = 0.0, 0.0, 0.0


def _recorder_with(samples, rate=60.0):
    rec = ACCRecorder.__new__(ACCRecorder)
    rec._rate_hz = float(rate)
    rec._rows = []
    rec._running = False
    rec._asm = _FakeReader(samples)
    return rec


# ------------------------------------------------------------------ tests ----
def test_contract_columns_constant():
    assert RECORDER_COLUMNS == [
        "timestamp", "lap_number", "sector", "track_position",
        "speed_kmh", "throttle", "brake", "steering", "gear", "rpm", "x", "y", "z",
    ]


def test_map_production_columns():
    rec = _recorder_with([_FakeMap()])
    row = rec.read()
    assert row is not None
    assert set(RECORDER_COLUMNS) == set(row.keys())

    # ACC -> schema field mappings
    assert row["timestamp"] == 123.4
    assert row["lap_number"] == 6          # completed_lap(5) + 1
    assert row["sector"] == 2              # current_sector_index(1) + 1
    assert row["track_position"] == 0.567  # normalized_car_position
    assert row["speed_kmh"] == 250.0
    assert row["throttle"] == 0.9          # gas
    assert row["brake"] == 0.0
    assert row["steering"] == -0.4
    assert row["gear"] == 6
    assert row["rpm"] == 9500.0
    assert (row["x"], row["y"], row["z"]) == (100.0, 200.0, 300.0)


def test_player_position_uses_player_car_id():
    rec = _recorder_with([_FakeMap()])
    row = rec.read()
    # player_car_id = 1 -> car_coordinates[1]
    assert row["x"] == 100.0
    assert row["y"] == 200.0
    assert row["z"] == 300.0


def test_to_dataframe_column_order():
    rec = _recorder_with([_FakeMap() for _ in range(3)])
    df = rec.to_dataframe()
    assert list(df.columns) == RECORDER_COLUMNS
    assert len(df) == 0  # read() buffers into read loop; to_dataframe uses _rows
    # now simulate buffered rows via read loop path
    rec._rows = [rec.read(), rec.read(), rec.read()]
    df2 = rec.to_dataframe()
    assert len(df2) == 3
    assert list(df2.columns) == RECORDER_COLUMNS


def test_bounded_clamps_to_unit_interval():
    assert _bounded(-1.5) == -1.0
    assert _bounded(2.0) == 1.0
    assert _bounded(0.5) == 0.5


def test_start_captures_samples_until_duration(tmp_path):
    # Many samples; use a short wall-clock duration so the loop terminates.
    samples = [_FakeMap() for _ in range(2000)]
    rec = _recorder_with(samples)
    rec._rate_hz = 1000.0
    rows = rec.start(duration_s=0.05)
    assert len(rows) > 0
    assert len(rows) <= len(samples)
    rec.stop()


def test_save_splits_into_per_lap_files(tmp_path):
    # build a tiny session that segmentation recognises as 2 valid laps
    rows = []
    for lap in (1, 2):
        step = 1.0 / 60.0
        t0 = lap * 40.0
        n = 1300  # > 20s so segmentation treats the lap as valid
        for i in range(n):
            tp = (i / n)
            rows.append({
                "timestamp": t0 + i * step,
                "lap_number": lap,
                "sector": 1,
                "track_position": tp,
                "speed_kmh": 150.0,
                "throttle": 0.5,
                "brake": 0.0,
                "steering": 0.0,
                "gear": 4,
                "rpm": 7000.0,
                "x": float(lap) * 100.0 + tp * 100.0,
                "y": float(lap) * 200.0,
                "z": 0.0,
            })
    rec = _recorder_with([])
    rec._rows = rows
    out = tmp_path / "session.csv"
    rec.save(out, split_laps=True)

    assert out.exists()
    lap_dir = tmp_path / "session_laps"
    assert lap_dir.is_dir()
    files = sorted(lap_dir.glob("lap_*.csv"))
    assert len(files) >= 1
    # spot-check lap-split schema
    first = pd.read_csv(files[0])
    assert list(first.columns) == RECORDER_COLUMNS
