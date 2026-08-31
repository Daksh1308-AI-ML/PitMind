"""Configuration loader.

All thresholds / tunables live in config.yaml (design.md: no magic numbers in code).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

REQUIRED_TELEMETRY_COLUMNS = [
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


@dataclass
class DetectionConfig:
    min_brake_pressure: float = 0.2
    throttle_resume: float = 0.3
    sample_rate_hz: int = 60


@dataclass
class RangesConfig:
    brake_point_delta_m: dict = field(default_factory=lambda: {"significant": 3.0, "potential": 10.0, "strong": 15.0})
    apex_speed_delta_kmh: float = 5.0
    throttle_delay_s: float = 0.1
    exit_speed_delta_kmh: float = 3.0
    steering_excess: float = 0.15
    corner_angle_deg: float = 10.0


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ranges: RangesConfig = field(default_factory=RangesConfig)
    timeloss_mode: str = "kinematic"
    synthetic: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        detection = raw.get("detection", {}) or {}
        ranges = raw.get("ranges", {}) or {}
        return cls(
            detection=DetectionConfig(**detection),
            ranges=RangesConfig(**ranges),
            timeloss_mode=(raw.get("timeloss", {}) or {}).get("mode", "kinematic"),
            synthetic=raw.get("synthetic", {}) or {},
        )