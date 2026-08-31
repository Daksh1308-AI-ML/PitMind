"""Coaching directives: template messages + priority filter.

Takes mistakes + time losses and generates prioritized, human-readable
coaching messages for the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pitmind import mistakes, timeloss, potential_lap
from pitmind.config import Config


@dataclass(frozen=True)
class CoachingDirective:
    """A single coaching instruction for the driver."""
    priority: int                    # 1 = highest, 3 = lowest
    category: str                    # "brake", "apex", "throttle", "exit", "steering"
    corner_name: str
    lap: int
    message: str                     # e.g., "Brake 5m later at T1"
    time_loss_s: float               # estimated loss from this mistake
    confidence: str                  # "weak" | "significant" | "strong"


# Message templates per mistake type
TEMPLATES = {
    mistakes.MistakeType.EARLY_BRAKE: {
        "weak": "Try braking {abs_delta:.0f}m later at {corner}",
        "significant": "Braking {abs_delta:.0f}m too early at {corner} — costs ~{loss:.2f}s",
        "strong": "Major early brake at {corner} ({abs_delta:.0f}m) — lose ~{loss:.2f}s",
    },
    mistakes.MistakeType.LATE_BRAKE: {
        "weak": "Brake {abs_delta:.0f}m earlier at {corner}",
        "significant": "Late brake at {corner} by {abs_delta:.0f}m — costs ~{loss:.2f}s",
        "strong": "Overshooting {corner} by {abs_delta:.0f}m — lose ~{loss:.2f}s",
    },
    mistakes.MistakeType.LOW_APEX_SPEED: {
        "weak": "Carry {abs_delta:.0f}km/h more speed at {corner} apex",
        "significant": "Apex speed {abs_delta:.0f}km/h low at {corner} — ~{loss:.2f}s lost",
        "strong": "Slow apex at {corner} by {abs_delta:.0f}km/h — costs ~{loss:.2f}s",
    },
    mistakes.MistakeType.LATE_THROTTLE: {
        "weak": "Get on throttle {delta:.2f}s earlier at {corner}",
        "significant": "Throttle delay {delta:.2f}s at {corner} — ~{loss:.2f}s lost",
        "strong": "Late throttle at {corner} ({delta:.2f}s) — costs ~{loss:.2f}s",
    },
    mistakes.MistakeType.SLOW_EXIT: {
        "weak": "Exit {corner} {abs_delta:.0f}km/h faster",
        "significant": "Slow exit at {corner} by {abs_delta:.0f}km/h — ~{loss:.2f}s lost",
        "strong": "Poor exit at {corner} (-{abs_delta:.0f}km/h) — costs ~{loss:.2f}s",
    },
    mistakes.MistakeType.EXCESS_STEERING: {
        "weak": "Smooth steering input at {corner}",
        "significant": "Overdriving {corner} (steer {delta:.2f}) — ~{loss:.2f}s scrubbed",
        "strong": "Excessive steering at {corner} — lose ~{loss:.2f}s",
    },
}


def format_message(m: mistakes.Mistake, time_loss: float) -> str:
    """Format a coaching message from a mistake and its time loss."""
    template_dict = TEMPLATES.get(m.mistake_type, {})
    template = template_dict.get(m.confidence.value, "{mistake_type} at {corner}")
    
    abs_delta = abs(m.delta_value)
    return template.format(
        corner=m.corner_name,
        delta=m.delta_value,
        abs_delta=abs_delta,
        loss=time_loss,
        mistake_type=m.mistake_type.value.replace("_", " "),
    )


def priority_from_confidence(confidence: str, time_loss: float) -> int:
    """Map confidence + time loss to priority (1=high, 2=med, 3=low)."""
    if confidence == "strong" or time_loss > 0.8:
        return 1
    if confidence == "significant" or time_loss > 0.3:
        return 2
    return 3


def category_from_type(mtype: mistakes.MistakeType) -> str:
    """Map mistake type to coaching category."""
    mapping = {
        mistakes.MistakeType.EARLY_BRAKE: "brake",
        mistakes.MistakeType.LATE_BRAKE: "brake",
        mistakes.MistakeType.LOW_APEX_SPEED: "apex",
        mistakes.MistakeType.LATE_THROTTLE: "throttle",
        mistakes.MistakeType.SLOW_EXIT: "exit",
        mistakes.MistakeType.EXCESS_STEERING: "steering",
    }
    return mapping.get(mtype, "other")


def generate_directives(
    mistake_list: list[mistakes.Mistake],
    time_loss_list: list[timeloss.TimeLoss],
    max_per_lap: int = 3,
    max_total: int = 10,
) -> list[CoachingDirective]:
    """Generate prioritized coaching directives.
    
    Args:
        mistake_list: Detected mistakes
        time_loss_list: Corresponding time loss estimates
        max_per_lap: Max directives per lap
        max_total: Max total directives
    
    Returns:
        Sorted list of CoachingDirective (highest priority first)
    """
    # Build time loss lookup
    loss_lookup = {
        (t.lap, t.corner, t.mistake_type): t.time_loss_s
        for t in time_loss_list
    }
    
    directives: list[CoachingDirective] = []
    
    for m in mistake_list:
        key = (m.lap, m.corner, m.mistake_type.value)
        t_loss = loss_lookup.get(key, 0.0)
        
        directive = CoachingDirective(
            priority=priority_from_confidence(m.confidence.value, t_loss),
            category=category_from_type(m.mistake_type),
            corner_name=m.corner_name,
            lap=m.lap,
            message=format_message(m, t_loss),
            time_loss_s=t_loss,
            confidence=m.confidence.value,
        )
        directives.append(directive)
    
    # Sort by priority, then by time loss descending
    directives.sort(key=lambda d: (d.priority, -d.time_loss_s))
    
    # Filter: max per lap
    per_lap_count: dict[int, int] = {}
    filtered: list[CoachingDirective] = []
    for d in directives:
        count = per_lap_count.get(d.lap, 0)
        if count < max_per_lap:
            filtered.append(d)
            per_lap_count[d.lap] = count + 1
    
    # Limit total
    return filtered[:max_total]


def directives_to_dataframe(directives: list[CoachingDirective]) -> pd.DataFrame:
    """Convert to DataFrame for display."""
    if not directives:
        return pd.DataFrame(columns=[
            "priority", "category", "corner", "lap", "message", 
            "time_loss_s", "confidence",
        ])
    return pd.DataFrame([
        {
            "priority": d.priority,
            "category": d.category,
            "corner": d.corner_name,
            "lap": d.lap,
            "message": d.message,
            "time_loss_s": d.time_loss_s,
            "confidence": d.confidence,
        }
        for d in directives
    ])


def generate_session_report(
    session_directives: list[CoachingDirective],
    potential: potential_lap.PotentialLap,
    total_time_loss: dict[int, float],
) -> str:
    """Generate a text summary report for the session."""
    lines = []
    lines.append("=" * 50)
    lines.append("PITMIND COACHING REPORT")
    lines.append("=" * 50)
    lines.append("")
    
    # Potential lap summary
    lines.append(f"Potential Lap Time: {potential.total_time_s:.3f}s")
    lines.append(f"  vs Best Actual: {potential.improvement_vs_best_s:+.3f}s")
    lines.append(f"  vs Reference:   {potential.improvement_vs_ref_s:+.3f}s")
    lines.append(f"  Sectors from:   L{potential.source_laps}")
    lines.append("")
    
    # Time loss summary
    lines.append("TIME LOSS BY LAP:")
    for lap, loss in sorted(total_time_loss.items()):
        lines.append(f"  Lap {lap}: {loss:.3f}s")
    lines.append("")
    
    # Directives grouped by priority
    for prio in [1, 2, 3]:
        prio_directives = [d for d in session_directives if d.priority == prio]
        if not prio_directives:
            continue
        label = {1: "HIGH PRIORITY", 2: "MEDIUM PRIORITY", 3: "LOW PRIORITY"}[prio]
        lines.append(f"{label}:")
        for d in prio_directives:
            lines.append(f"  L{d.lap} {d.corner_name}: {d.message}")
        lines.append("")
    
    return "\n".join(lines)