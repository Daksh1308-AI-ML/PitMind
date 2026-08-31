"""Potential lap: best-sector composite from all laps.

Constructs a theoretical "perfect lap" by stitching together the best
sector times from all valid laps in the session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pitmind.config import Config
from pitmind import segmentation, features


@dataclass(frozen=True)
class SectorTime:
    """Single sector time from a lap."""
    lap: int
    sector: int
    time_s: float
    start_tp: float
    end_tp: float


@dataclass(frozen=True)
class PotentialLap:
    """Composite best-sector lap."""
    sector_times: tuple[SectorTime, ...]  # best sector from each lap
    total_time_s: float
    source_laps: tuple[int, ...]          # which lap each sector came from
    improvement_vs_best_s: float          # vs fastest actual lap
    improvement_vs_ref_s: float           # vs reference lap


def extract_sector_times(laps: list[pd.DataFrame], n_sectors: int = 3) -> list[SectorTime]:
    """Extract sector times from valid laps.
    
    Sector boundaries at 0/3, 1/3, 2/3 of track_position.
    """
    sectors: list[SectorTime] = []
    for lap_df in laps:
        lap_no = int(lap_df["lap_number"].iloc[0])
        tp = lap_df["track_position"].to_numpy()
        t = lap_df["timestamp"].to_numpy()
        
        for s in range(1, n_sectors + 1):
            start_tp = (s - 1) / n_sectors
            end_tp = s / n_sectors
            
            # Find indices within this sector
            mask = (tp >= start_tp) & (tp <= end_tp)
            if mask.sum() < 2:
                continue
            
            sector_start_idx = np.where(mask)[0][0]
            sector_end_idx = np.where(mask)[0][-1]
            
            sector_time = t[sector_end_idx] - t[sector_start_idx]
            
            sectors.append(SectorTime(
                lap=lap_no,
                sector=s,
                time_s=float(sector_time),
                start_tp=start_tp,
                end_tp=end_tp,
            ))
    return sectors


def build_potential_lap(
    session: pd.DataFrame,
    cfg: Config,
    n_sectors: int = 3,
) -> PotentialLap:
    """Build the potential (theoretical best) lap from sector times.
    
    Args:
        session: Full session DataFrame
        cfg: Configuration
        n_sectors: Number of sectors (default 3)
    
    Returns:
        PotentialLap with best sector times and total
    """
    from pitmind import preprocess
    session = preprocess.preprocess(session, cfg)
    laps = segmentation.valid_laps(session)
    
    if not laps:
        raise ValueError("No valid laps")
    
    sector_times = extract_sector_times(laps, n_sectors)
    
    # Find best time for each sector
    best_sectors: list[SectorTime] = []
    source_laps = []
    
    for s in range(1, n_sectors + 1):
        sector_candidates = [st for st in sector_times if st.sector == s]
        if not sector_candidates:
            continue
        best = min(sector_candidates, key=lambda x: x.time_s)
        best_sectors.append(best)
        source_laps.append(best.lap)
    
    total_time = sum(s.time_s for s in best_sectors)
    
    # Compare to fastest actual lap
    lap_times = [(lap["timestamp"].max() - lap["timestamp"].min(), int(lap["lap_number"].iloc[0])) 
                 for lap in laps]
    fastest_actual = min(lap_times)[0]
    
    # Compare to reference lap (lap 1)
    ref_lap = next((lap for lap in laps if int(lap["lap_number"].iloc[0]) == 1), laps[0])
    ref_time = ref_lap["timestamp"].max() - ref_lap["timestamp"].min()
    
    return PotentialLap(
        sector_times=tuple(best_sectors),
        total_time_s=total_time,
        source_laps=tuple(source_laps),
        improvement_vs_best_s=fastest_actual - total_time,
        improvement_vs_ref_s=ref_time - total_time,
    )


def potential_lap_to_dataframe(potential: PotentialLap) -> pd.DataFrame:
    """Convert to DataFrame for display."""
    rows = []
    for st in potential.sector_times:
        rows.append({
            "sector": f"S{st.sector}",
            "time_s": st.time_s,
            "source_lap": st.lap,
            "start_tp": st.start_tp,
            "end_tp": st.end_tp,
        })
    df = pd.DataFrame(rows)
    # Add summary row
    summary = pd.DataFrame([{
        "sector": "TOTAL",
        "time_s": potential.total_time_s,
        "source_lap": "-",
        "start_tp": 0.0,
        "end_tp": 1.0,
    }])
    return pd.concat([df, summary], ignore_index=True)


def interpolate_potential_telemetry(
    potential: PotentialLap,
    session: pd.DataFrame,
    cfg: Config,
    n_points: int = 2000,
) -> pd.DataFrame:
    """Generate a synthetic 'potential lap' telemetry by interpolating between 
    the best sector source laps.
    
    This creates a full telemetry trace for the potential lap that can be
    overlaid on plots.
    """
    from pitmind import preprocess, corners
    session = preprocess.preprocess(session, cfg)
    laps = segmentation.valid_laps(session)
    
    # Build a map of lap_number -> lap DataFrame
    lap_map = {int(lap["lap_number"].iloc[0]): lap for lap in laps}
    
    # Resample each source lap sector onto uniform grid
    tp_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    merged_data = {"track_position": tp_grid}
    
    for col in ["speed_kmh", "throttle", "brake", "steering", "gear", "rpm", "x", "y", "z"]:
        merged_data[col] = np.full(n_points, np.nan)
    
    for st in potential.sector_times:
        source_lap = lap_map.get(st.lap)
        if source_lap is None:
            continue
        
        source_lap = source_lap.sort_values("track_position").reset_index(drop=True)
        tp_src = source_lap["track_position"].to_numpy()
        
        # Sector mask on target grid
        if st.start_tp <= st.end_tp:
            mask = (tp_grid >= st.start_tp) & (tp_grid <= st.end_tp)
        else:
            mask = (tp_grid >= st.start_tp) | (tp_grid <= st.end_tp)
        
        if mask.sum() < 2:
            continue
        
        for col in ["speed_kmh", "throttle", "brake", "steering", "gear", "rpm", "x", "y", "z"]:
            if col in source_lap.columns:
                vals = np.interp(tp_grid[mask], tp_src, source_lap[col].to_numpy())
                merged_data[col][mask] = vals
    
    # Fill any remaining NaN with forward/backward fill
    for col in merged_data:
        if col != "track_position":
            arr = merged_data[col]
            # Forward fill
            mask = ~np.isnan(arr)
            if mask.any():
                np.maximum.accumulate(mask.astype(float) * np.arange(len(arr)), out=arr)
                arr[~mask] = np.nan
                # Actually simpler: pandas ffill
                s = pd.Series(arr)
                s = s.ffill().bfill()
                merged_data[col] = s.to_numpy()
    
    # Add timestamps
    t_start = 0.0
    dt = potential.total_time_s / n_points
    merged_data["timestamp"] = np.arange(n_points) * dt
    merged_data["lap_number"] = 0  # special marker for potential lap
    merged_data["sector"] = np.digitize(tp_grid, np.linspace(0, 1, 4))  # 1,2,3
    
    return pd.DataFrame(merged_data)