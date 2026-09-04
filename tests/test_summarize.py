"""Tests for pitmind.summarize — the race-engineer callout (todo.md M5).

The LLM is display-only: it rephrases the deterministic diagnosis. These tests
verify the deterministic payload + TemplateEngineer (no network), and that the
Ollama path falls back gracefully when the server is unavailable. A live Ollama
test exists but is skipped unless run with ``-m live`` -- CI stays offline.
"""

from __future__ import annotations

import os
import urllib.error

import pandas as pd
import pytest

from pitmind import summarize
from pitmind.config import Config
from tools import tune

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "f1_monza_laps.csv")


def _ollama_reachable(base_url: str) -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(base_url.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def bundle():
    df = pd.read_csv(DATA)
    cfg = Config.from_file()
    return tune.run_pipeline(df, cfg, capabilities={"steering": False})


@pytest.fixture()
def cfg():
    cfg = Config.from_file()
    cfg.llm.enabled = False  # hermetically off unless a test opts in
    return cfg


def test_build_engineer_payload_shape(bundle, cfg):
    p = summarize.build_engineer_payload(bundle, cfg, driver="VER")
    payload = p.to_dict()
    assert payload["driver"] == "VER"
    assert payload["total_time_loss_s"] >= 0
    assert payload["n_mistakes"] >= 0
    assert isinstance(payload["directives"], list) and len(payload["directives"]) <= 4
    for d in payload["directives"]:
        assert set(d) >= {"corner", "priority", "message", "time_loss_s"}
    # top corner present when there is loss to find
    assert payload["top_corner"] is None or {"name", "loss", "cause"} <= payload["top_corner"].keys()


def test_template_engineer_is_deterministic(bundle, cfg):
    a = summarize.TemplateEngineer().callout(
        summarize.build_engineer_payload(bundle, cfg), cfg)
    b = summarize.TemplateEngineer().callout(
        summarize.build_engineer_payload(bundle, cfg), cfg)
    assert a == b
    # reads like a race engineer
    assert "to find" in a or "Potential lap" in a


def test_engineer_callout_returns_template_when_disabled(bundle, cfg):
    cfg.llm.enabled = False
    out = summarize.engineer_callout(bundle, cfg, driver="VER")
    assert isinstance(out, str) and out


def test_engineer_callout_force_runs_provider_even_when_disabled(bundle, cfg, monkeypatch):
    cfg.llm.enabled = False
    cfg.llm.provider = "ollama"
    # provider should be invoked because force=True
    called = {"n": 0}

    class _Fake:
        def callout(self, payload, c):
            called["n"] += 1
            return "FAKE_ORACLE"

    monkeypatch.setattr(summarize, "get_provider", lambda c: _Fake())
    out = summarize.engineer_callout(bundle, cfg, driver="VER", force=True)
    assert out == "FAKE_ORACLE"
    assert called["n"] == 1


def test_ollama_falls_back_to_template_on_failure(bundle, cfg, monkeypatch):
    cfg.llm.enabled = True
    cfg.llm.provider = "ollama"

    # _generate returning None is exactly what a failed/offline call produces
    # (its internal try/except eats the network error and returns None).
    monkeypatch.setattr(summarize.OllamaEngineer, "_generate", lambda self, p, c: None)
    out = summarize.engineer_callout(bundle, cfg, driver="VER")
    assert isinstance(out, str) and out  # deterministic template, not a crash


def test_ollama_generate_catches_network_error(bundle, cfg, monkeypatch):
    cfg.llm.enabled = True
    cfg.llm.provider = "ollama"

    eng = summarize.OllamaEngineer()

    def _raising_urlopen(*a, **k):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", _raising_urlopen)
    assert eng._generate("prompt", cfg) is None


def test_get_provider_returns_template_when_off(cfg):
    cfg.llm.enabled = False
    assert isinstance(summarize.get_provider(cfg), summarize.TemplateEngineer)


def test_get_provider_returns_ollama_when_enabled(cfg):
    cfg.llm.enabled = True
    cfg.llm.provider = "ollama"
    assert isinstance(summarize.get_provider(cfg), summarize.OllamaEngineer)


def test_payload_capabilities_never_changes_numbers(bundle, cfg):
    # capabilities only label root causes; the time-loss numbers must not move
    p1 = summarize.build_engineer_payload(bundle, cfg, capabilities={})
    p2 = summarize.build_engineer_payload(bundle, cfg, capabilities={"steering": False})
    assert p1.total_time_loss_s == p2.total_time_loss_s


@pytest.mark.live
@pytest.mark.skipif(not _ollama_reachable("http://localhost:11434"),
                    reason="Ollama not reachable; run with a local server + pytest -m live")
def test_live_ollama_callout(bundle, cfg):
    """Requires a running local Ollama with qwen2.5:7b (run with pytest -m live)."""
    cfg.llm.enabled = True
    cfg.llm.provider = "ollama"
    cfg.llm.model = "qwen2.5:7b"
    out = summarize.engineer_callout(bundle, cfg, driver="VER")
    assert isinstance(out, str) and len(out) > 0
