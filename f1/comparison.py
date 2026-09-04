"""Multi-driver F1 comparison (todo.md M3).

Compares several drivers' bridged F1 sessions corner-by-corner against a chosen
reference driver, using the existing analysis core (features -> mistakes ->
timeloss). Produces the classic race-engineer delta, e.g.:

    "VER loses 0.22s to LEC in T7 by braking 14m early"

Because every driver's session is generated on the same circuit and corner
detection is track-agnostic (architect.md rule 1), corners align 1:1 by index
across drivers, so corner *n* refers to the same apex for every driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitmind.config import Config
from pitmind import features, mistakes, timeloss

# F1 inputs have no steering channel -> prune steering-based mistakes.
F1_CAPABILITIES = {"steering": False}


@dataclass
class CornerDelta:
    """Per-driver, per-corner delta vs a reference driver."""
    driver: str
    corner: int
    corner_name: str
    driver_corner_loss_s: float      # this driver's total time loss in the corner (all laps)
    ref_corner_loss_s: float         # reference driver's total time loss in the corner
    delta_s: float                   # driver_corner_loss_s - ref_corner_loss_s (+ = slower)
    root_cause: str | None           # e.g. "braking 14.0m early in T7"
    root_loss_s: float               # time loss attributable to that root cause
    mistake_type: str | None


@dataclass
class DriverComparison:
    """Comparison of one driver against a reference driver."""
    driver: str
    reference_driver: str
    corner_deltas: list[CornerDelta] = field(default_factory=list)

    @property
    def total_delta_s(self) -> float:
        return sum(d.delta_s for d in self.corner_deltas)

    def worst_corners(self, k: int = 3) -> list[CornerDelta]:
        """The k corners where this driver loses the most vs the reference."""
        return sorted(self.corner_deltas, key=lambda d: d.delta_s, reverse=True)[:k]


def driver_corner_table(session: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-corner summary for ONE driver's bridged session.

    Returns a DataFrame with one row per corner:
        corner, name,
        n_laps, mean_corner_time_s,
        total_loss_s (sum of timeloss across the driver's laps in this corner),
        top_mistake (mistake type with the largest summed loss),
        top_delta_value (the worst raw delta of that top mistake)
    """
    table = features.build_feature_table(session, cfg)
    mlist = mistakes.detect_mistakes(table, cfg, capabilities=F1_CAPABILITIES)
    tlist = timeloss.estimate_time_loss(mlist, table, cfg)

    loss_df = timeloss.time_loss_to_dataframe(tlist)

    # aggregate time loss by (corner, mistake_type)
    by_corner: dict[int, dict] = {}
    for corner in sorted(table["corner"].unique()):
        ct = table[table["corner"] == corner]
        name = str(ct["name"].iloc[0]) if "name" in ct.columns else f"T{corner + 1}"
        mean_time = float(ct["corner_time_s"].mean()) if pd.notna(ct["corner_time_s"]).any() else np.nan
        by_corner[int(corner)] = {
            "corner": int(corner),
            "name": name,
            "n_laps": int(table["lap"].nunique()),
            "mean_corner_time_s": mean_time,
            "total_loss_s": 0.0,
            "top_mistake": None,
            "top_delta_value": 0.0,
        }

    # sum losses per (corner, type), keep the worst per corner
    if not loss_df.empty:
        per_corner_groups: dict[int, tuple[str, float, float]] = {}
        for (corner, mtype), grp in loss_df.groupby(["corner", "mistake_type"]):
            k = int(corner)
            if k not in by_corner:
                continue
            loss = float(grp["time_loss_s"].sum())
            worst = float(grp["delta_value"].abs().max()) if "delta_value" in grp.columns else 0.0
            by_corner[k]["total_loss_s"] += loss
            # root cause = the mistake type with the LARGEST summed loss in this corner
            if k not in per_corner_groups or loss > per_corner_groups[k][1]:
                per_corner_groups[k] = (str(mtype), loss, worst)
        for k, (mtype, _loss, worst) in per_corner_groups.items():
            by_corner[k]["top_mistake"] = mtype
            by_corner[k]["top_delta_value"] = worst

    rows = list(by_corner.values())
    return pd.DataFrame(rows)


def _root_cause_phrase(mtype: str | None, delta_value: float) -> str | None:
    """Turn a mistake type + magnitude into a coaching phrase, e.g. 'braking 14m early'."""
    if mtype is None:
        return None
    mapping = {
        "early_brake": ("braking {:.1f}m early", abs(delta_value)),
        "late_brake": ("braking {:.1f}m late", abs(delta_value)),
        "low_apex_speed": ("carrying {:.1f}km/h less apex speed", abs(delta_value)),
        "late_throttle": ("getting on the throttle {:.2f}s late", abs(delta_value)),
        "slow_exit": ("exiting {:.1f}km/h slower", abs(delta_value)),
        "excess_steering": ("over-steering {:.2f}", abs(delta_value)),
    }
    tmpl, val = mapping.get(mtype, ("{:.2f} ({})", 0.0))
    try:
        return tmpl.format(val, mtype) if mtype not in mapping else tmpl.format(val)
    except (ValueError, TypeError):
        return mtype


def compare_drivers(
    sessions: dict[str, pd.DataFrame],
    cfg: Config,
    reference_driver: str | None = None,
) -> list[DriverComparison]:
    """Compare every driver against a reference driver, corner by corner.

    Args:
        sessions: {driver_code: bridged session DataFrame}.
        cfg: Configuration.
        reference_driver: Driver to compare against. If None, the driver with the
            lowest total time loss is used (the fastest/most consistent).
        sessions must all be from the same circuit so corners align by index.

    Returns:
        One DriverComparison per non-reference driver.
    """
    if len(sessions) < 2:
        raise ValueError("compare_drivers needs at least two drivers")

    per_driver = {
        code: driver_corner_table(df, cfg)
        for code, df in sessions.items()
    }

    if reference_driver is None:
        # reference = driver whose summed corner loss is smallest
        totals = {
            code: float(tbl["total_loss_s"].sum())
            for code, tbl in per_driver.items()
        }
        reference_driver = min(totals, key=totals.get)

    ref_table = per_driver[reference_driver]
    ref_loss_by_corner = {
        int(r["corner"]): float(r["total_loss_s"])
        for _, r in ref_table.iterrows()
    }

    results: list[DriverComparison] = []
    for code, tbl in per_driver.items():
        if code == reference_driver:
            continue
        deltas: list[CornerDelta] = []
        for _, r in tbl.sort_values("corner").iterrows():
            corner = int(r["corner"])
            d_loss = float(r["total_loss_s"])
            r_loss = ref_loss_by_corner.get(corner, 0.0)
            top_m = r["top_mistake"]
            root = _root_cause_phrase(top_m, float(r["top_delta_value"]) if r["top_delta_value"] else 0.0)
            deltas.append(CornerDelta(
                driver=code,
                corner=corner,
                corner_name=str(r["name"]),
                driver_corner_loss_s=round(d_loss, 4),
                ref_corner_loss_s=round(r_loss, 4),
                delta_s=round(d_loss - r_loss, 4),
                root_cause=root,
                root_loss_s=round(float(r["total_loss_s"]) - r_loss, 4) if top_m else 0.0,
                mistake_type=top_m,
            ))
        results.append(DriverComparison(driver=code, reference_driver=reference_driver,
                                        corner_deltas=deltas))
    return results


def comparison_to_dataframe(comparisons: list[DriverComparison]) -> pd.DataFrame:
    """Flatten all driver comparisons into a single table for the dashboard."""
    rows = []
    for comp in comparisons:
        for d in comp.corner_deltas:
            rows.append({
                "driver": d.driver,
                "vs": comp.reference_driver,
                "corner": d.corner_name,
                "delta_s": d.delta_s,
                "driver_loss_s": d.driver_corner_loss_s,
                "ref_loss_s": d.ref_corner_loss_s,
                "root_cause": d.root_cause or "",
            })
    return pd.DataFrame(rows)


def phrase_delta(comp: DriverComparison) -> str:
    """Human summary: 'VER loses 0.22s to LEC overall' + worst corners."""
    if not comp.corner_deltas:
        return f"{comp.driver} vs {comp.reference_driver}: no corners compared"
    lines = []
    sign = "loses" if comp.total_delta_s >= 0 else "gains"
    lines.append(f"{comp.driver} {sign} {abs(comp.total_delta_s):.2f}s to "
                 f"{comp.reference_driver} across the lap")
    for d in comp.worst_corners(3):
        if d.delta_s <= 0:
            continue
        cause = f" by {d.root_cause}" if d.root_cause else ""
        lines.append(f"  {d.corner_name} {d.delta_s:+.2f}s{cause}")
    return "\n".join(lines)
