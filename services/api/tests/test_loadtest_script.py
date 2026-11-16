"""The load-test script, driven against an in-process mock transport (no network)."""

import asyncio
import importlib.util
import pathlib
import sys

import httpx
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "loadtest" / "load_test.py"
spec = importlib.util.spec_from_file_location("load_test", _PATH)
load_test = importlib.util.module_from_spec(spec)
sys.modules["load_test"] = load_test
spec.loader.exec_module(load_test)


def test_governance_scenario_reports_latency_and_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202 if request.method == "POST" else 200, json={})

    stats, elapsed = asyncio.run(
        load_test.run("http://t", "governance", 4, 0.2, transport=httpx.MockTransport(handler))
    )
    s = stats.summary(elapsed)
    assert s["requests"] > 0 and s["errors"] == 0 and s["error_rate"] == 0
    assert s["latency_ms"]["p95"] >= s["latency_ms"]["p50"]


def test_errors_are_counted_and_fail_the_gate(monkeypatch, capsys):
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    stats, elapsed = asyncio.run(load_test.run("http://t", "health", 2, 0.1, transport=transport))
    assert stats.summary(elapsed)["error_rate"] == 1.0

    async def fake_run(*a, **k):
        return stats, elapsed

    monkeypatch.setattr(load_test, "run", fake_run)
    assert load_test.main(["--scenario", "health", "--duration", "1"]) == 1


def test_agent_scenario_detects_cross_thread_leakage():
    threads: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            import json

            body = json.loads(request.content)
            threads.setdefault(body["thread_id"], []).append(body["message"])
            return httpx.Response(200, json={})
        thread = request.url.path.rsplit("/", 1)[1]
        # a leaky server: every thread also contains someone else's message
        msgs = threads.get(thread, []) + ["intruder"]
        return httpx.Response(
            200, json={"messages": [{"type": "human", "content": m} for m in msgs]}
        )

    stats, _ = asyncio.run(
        load_test.run("http://t", "agent", 2, 0.1, transport=httpx.MockTransport(handler))
    )
    assert stats.isolation_violations > 0


def test_agent_scenario_passes_on_isolated_threads():
    threads: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            import json

            body = json.loads(request.content)
            threads.setdefault(body["thread_id"], []).append(body["message"])
            return httpx.Response(200, json={})
        thread = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(
            200, json={"messages": [{"type": "human", "content": m} for m in threads[thread]]}
        )

    stats, _ = asyncio.run(
        load_test.run("http://t", "agent", 3, 0.1, transport=httpx.MockTransport(handler))
    )
    assert stats.isolation_violations == 0 and stats.errors == 0


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        load_test.main(["--scenario", "nope"])
