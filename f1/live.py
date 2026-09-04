"""Live race-engineer view (todo.md M4).

Polls a live data source for the freshest bridged telemetry slice and pushes it
through the PitMind pipeline, emitting human race-engineer summaries.

Two concerns are kept separate so the whole thing is testable offline:

  * a *source* is any callable ``source() -> pd.DataFrame | None`` that returns
    the latest session slice (a 13-column contract DataFrame) or ``None`` when
    there is nothing new since the last poll;
  * the *engineer loop* polls the source, runs the analysis and hands the
    result to an *emit* callback.

Out of the box we provide ``fastf1_source`` (fragments fetched + bridged via the
M1 bridge) which requires the ``f1`` extra and a network/livetiming feed; tests
inject a fake in-memory source instead (no fastf1, no network).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from pitmind.config import Config

log = logging.getLogger(__name__)

# F1 lacks a steering channel and its Brake is boolean -> prune steering mistakes.
F1_CAPABILITIES: dict[str, bool] = {"steering": False}


@dataclass
class LiveLap:
    """One analysed lap as emitted by the race engineer."""

    lap_number: int
    total_time_loss_s: float
    n_mistakes: int
    n_directives: int
    summary: str
    directives: list[str] = field(default_factory=list)


def analyze_session(
    df: pd.DataFrame,
    cfg: Config,
    capabilities: Optional[dict[str, bool]] = None,
) -> LiveLap:
    """Analyse a single-lap slice and produce a race-engineer LiveLap."""
    from tools import tune  # heavy, defer import

    caps = capabilities if capabilities is not None else F1_CAPABILITIES
    bundle = tune.run_pipeline(df, cfg, capabilities=caps)

    lap_no = int(df["lap_number"].iloc[0]) if "lap_number" in df and len(df) else 1
    top_directives = [
        f"[{_severity(d.priority)}] {d.message}"
        for d in bundle["directives"][:5]
    ]
    return LiveLap(
        lap_number=lap_no,
        total_time_loss_s=bundle["total_time_loss_s"],
        n_mistakes=len(bundle["mistakes"]),
        n_directives=len(bundle["directives"]),
        summary=_summary_text(bundle["summary"]),
        directives=top_directives,
    )


def _severity(priority: int) -> str:
    """Map a directive priority (1 = most critical) to a severity tag."""
    if priority == 1:
        return "HIGH"
    if priority == 2:
        return "MED"
    return "LOW"


def _summary_text(summary: dict) -> str:
    """Render the mistakes summary dict as a short race-engineer line."""
    total = summary.get("total", 0)
    worst = sorted(summary.get("by_corner", {}).items(),
                   key=lambda kv: kv[1], reverse=True)[:3]
    worst_txt = ", ".join(f"{c} ({n})" for c, n in worst) or "none"
    return (f"{total} issues; worst corners: {worst_txt}; "
            f"~{summary.get('by_confidence', {}).get('significant', 0)} significant")


def engineer_loop(
    source: Callable[[], Optional[pd.DataFrame]],
    emit: Callable[[LiveLap], None],
    cfg: Config,
    *,
    interval_s: float = 3.0,
    max_ticks: Optional[int] = None,
    capabilities: Optional[dict[str, bool]] = None,
) -> None:
    """Poll ``source`` and emit a LiveLap for every new slice.

    Args:
        source: Returns the latest bridged telemetry slice or ``None`` when
            there is nothing new since the last poll.
        emit: Receives each newly-analysed LiveLap (the race-engineer hook).
        cfg: PitMind config.
        interval_s: Seconds to wait between polls (real livetiming ~3 s).
        max_ticks: Stop after this many polls (testing / dry-run). None = forever.
        capabilities: Channel capability flags (default F1, steering=False).
    """
    caps = capabilities if capabilities is not None else F1_CAPABILITIES
    seen = 0
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        fresh = source()
        if fresh is not None and not fresh.empty:
            seen += 1
            try:
                lap = analyze_session(fresh, cfg, capabilities=caps)
            except Exception:
                log.exception("failed to analyse live slice")
                emit(_error_lap(ticks))
                continue
            log.info("analysed live slice %d (lap %d)", seen, lap.lap_number)
            emit(lap)
        time.sleep(interval_s)


def _error_lap(ticks: int) -> LiveLap:
    return LiveLap(
        lap_number=ticks,
        total_time_loss_s=0.0,
        n_mistakes=0,
        n_directives=0,
        summary="Analysis failed on this slice.",
        directives=[],
    )


# --------------------------------------------------------------------------- #
# Real livetiming source (optional, needs the `f1` extra + network)
# --------------------------------------------------------------------------- #
def fastf1_source(
    year: int,
    event: str,
    session: str,
    driver: str,
    *,
    poll_laps: int = 1,
):
    """Build a live source that fetches the latest ``poll_laps`` lap(s) via FastF1.

    Returns a callable returning a fresh bridged slice each time new telemetry
    is available (or ``None`` otherwise). Emits nothing until a race/livetiming
    feed has telemetry. Requires the ``f1`` extra and network access.

    Note: for a truly live (~3 s delay) view use OpenF1's live endpoints through
    this same source signature; FastF1 livetiming is generally session/full-race
    telemetry rather than sub-second. The bridged slice is a 13-column contract
    DataFrame exactly like an offline file.
    """
    from f1.cli import fetch_session  # deferred, needs fastf1
    from f1.fastf1_bridge import FastF1Bridge

    bridge = FastF1Bridge()
    seen_keys: set = set()

    def _poll() -> Optional[pd.DataFrame]:
        nonlocal seen_keys
        frames = fetch_session(year, event, session, driver)
        if not frames:
            return None
        session_df = bridge.convert_session(frames)
        key = tuple(session_df["lap_number"].unique())
        if key in seen_keys:
            return None
        seen_keys.add(key)
        return session_df

    return _poll
