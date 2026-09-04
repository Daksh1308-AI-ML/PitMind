"""Race-engineer summariser (todo.md M5 / plan A).

Turns the deterministic diagnosis bundle into a short, natural-language race
engineer callout.

Design rule (architect.md): the LLM is a *display-only* layer. `build_engineer_payload`
is pure code that assembles the structured, unambiguous diagnosis; a provider only
*phrases* that diagnosis. If no provider is configured or the server is down, the
`TemplateEngineer` falls back to the exact same content deterministically — so the
project always works and never fails because of the language model.

Providers:
  * `OllamaEngineer`  — local Ollama (qwen2.5:7b) via its /api/generate endpoint.
  * `TemplateEngineer` — deterministic sentence builder (no network, always works).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from pitmind.config import Config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Deterministic payload — the single source of truth the LLM only phrases.
# --------------------------------------------------------------------------- #
@dataclass
class EngineerPayload:
    driver: str | None = None
    lap: int | None = None
    total_time_loss_s: float = 0.0
    n_mistakes: int = 0
    top_corner: dict | None = None          # {"name","loss","cause"}
    directives: list[dict] = field(default_factory=list)  # top 4, priority order
    potential_delta_s: float | None = None
    summary_raw: str = ""

    def to_dict(self) -> dict:
        return {
            "driver": self.driver,
            "lap": self.lap,
            "total_time_loss_s": round(self.total_time_loss_s, 3),
            "n_mistakes": self.n_mistakes,
            "top_corner": self.top_corner,
            "directives": self.directives[:4],
            "potential_delta_s": self.potential_delta_s,
            "summary_raw": self.summary_raw,
        }


def build_engineer_payload(
    bundle: dict,
    cfg: Config,
    *,
    driver: str | None = None,
    lap: int | None = None,
    capabilities: dict[str, bool] | None = None,
) -> EngineerPayload:
    """Assemble the structured diagnosis from a pipeline bundle (pure code).

    `bundle` is what `tools.tune.run_pipeline` returns. `capabilities` only
    affects how the payload is *labelled* (e.g. skip steering root causes when
    F1 has no steering channel) — never the numbers, which come from the bundle.
    """
    directives = bundle.get("directives", [])

    # top corner by summed time loss across directives
    by_corner: dict[str, float] = {}
    for d in directives:
        by_corner[d.corner_name] = by_corner.get(d.corner_name, 0.0) + float(d.time_loss_s or 0.0)

    top_corner = None
    if by_corner:
        t_name, t_loss = max(by_corner.items(), key=lambda kv: kv[1])
        causes = [
            f"{d.category}" for d in directives
            if d.corner_name == t_name and d.time_loss_s and d.time_loss_s > 0
        ]
        top_corner = {
            "name": t_name,
            "loss": round(t_loss, 2),
            "cause": causes[0] if causes else "misc",
        }

    dir_rows = [
        {
            "corner": d.corner_name,
            "priority": int(getattr(d, "priority", 0)),
            "message": d.message,
            "time_loss_s": round(float(d.time_loss_s or 0.0), 3),
        }
        for d in directives[:4]
    ]

    pot = bundle.get("potential_lap")
    pot_delta = None
    if pot is not None:
        imp = getattr(pot, "improvement_vs_best_s", None)
        pot_delta = float(imp) if imp is not None else None

    return EngineerPayload(
        driver=driver,
        lap=lap if lap is not None else _first_lap(bundle),
        total_time_loss_s=float(bundle.get("total_time_loss_s", 0.0)),
        n_mistakes=len(bundle.get("mistakes", [])),
        top_corner=top_corner,
        directives=dir_rows,
        potential_delta_s=pot_delta,
        summary_raw=str(bundle.get("summary", "")),
    )


def _first_lap(bundle: dict) -> int | None:
    lap_times = bundle.get("lap_times") or []
    return len(lap_times) if lap_times else None


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class EngineerProvider(Protocol):
    def callout(self, payload: EngineerPayload, cfg: Config) -> str: ...


class TemplateEngineer:
    """Deterministic fallback: builds the callout with no network/server."""

    def callout(self, payload: EngineerPayload, cfg: Config) -> str:
        p = payload
        parts = []
        who = f"{p.driver} " if p.driver else ""
        if p.lap:
            parts.append(f"Lap {p.lap}:")
        parts.append(f"~{p.total_time_loss_s:.2f}s to find ({p.n_mistakes} issues).")
        if p.top_corner and p.top_corner["loss"] > 0:
            parts.append(
                f"Biggest win: {p.top_corner['name']} "
                f"(loss -{p.top_corner['loss']:.2f}s, root cause \"{p.top_corner['cause']}\")."
            )
        if p.directives:
            top = p.directives[0]
            parts.append(f"Try: {top['message']}")
        if p.potential_delta_s is not None:
            parts.append(
                f"Potential lap is {p.potential_delta_s:+.2f}s faster than best."
            )
        return f"{who}{' '.join(parts)}"


class OllamaEngineer:
    """Phrase the payload through local Ollama. Falls back to Template on any error."""

    def __init__(self, fallback: EngineerProvider | None = None):
        self._fallback = fallback or TemplateEngineer()

    def callout(self, payload: EngineerPayload, cfg: Config) -> str:
        prompt = self._build_prompt(payload, cfg)
        raw = self._generate(prompt, cfg)
        if raw is None:
            return self._fallback.callout(payload, cfg)
        text = raw.strip()
        if not text:
            return self._fallback.callout(payload, cfg)
        return text

    def _generate(self, prompt: str, cfg: Config) -> str | None:
        """POST to Ollama /api/generate (stream off). Returns None on failure."""
        url = cfg.llm.base_url.rstrip("/") + "/api/generate"
        body = json.dumps({"model": cfg.llm.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.llm.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            log.warning("OllamaEngineer unavailable (%s); using template fallback.", e)
            return None

    def _build_prompt(self, payload: EngineerPayload, cfg: Config) -> str:
        system = (
            "You are a concise F1 race engineer. Reply with 1-3 short, actionable "
            "sentences addressed to the driver. Use exact numbers from the data. "
            "Do not invent causes. Do not mention that you are an AI."
        )
        return (
            f"{system}\n\nSTRUCTURED DIAGNOSIS (JSON):\n"
            f"{json.dumps(payload.to_dict(), indent=2)}\n\n"
            "Now write the coaching callout:"
        )


def get_provider(cfg: Config) -> EngineerProvider:
    """Return the provider for cfg.llm (Template when off/unknown/missing deps)."""
    if cfg.llm.enabled and cfg.llm.provider == "ollama":
        return OllamaEngineer()
    return TemplateEngineer()


def engineer_callout(
    bundle: dict,
    cfg: Config,
    *,
    driver: str | None = None,
    lap: int | None = None,
    capabilities: dict[str, bool] | None = None,
    force: bool = False,
) -> str:
    """One-stop helper: build payload + run the configured provider.

    ``force=True`` runs the provider even when ``cfg.llm.enabled`` is False
    (used by CLI --summary so the user can opt in per-call); disabled by default
    returns the deterministic template callout.
    """
    provider = get_provider(cfg)
    payload = build_engineer_payload(bundle, cfg, driver=driver, lap=lap,
                                     capabilities=capabilities)
    if cfg.llm.enabled or force:
        return provider.callout(payload, cfg)
    return TemplateEngineer().callout(payload, cfg)
