#!/usr/bin/env python3
"""
Load test for the localAIStack API (async httpx; no extra dependencies).

    python3 scripts/loadtest/load_test.py --scenario governance --concurrency 50 --duration 30

Scenarios
  health       GET /health                                     (infra smoke)
  governance   GET /governance/state + POST /governance/dispatch   (no LLM needed)
  agent        POST /lg/agent/run on a private thread per worker, then GET
               /lg/agent/state/{thread} and check the thread holds exactly its own
               turns — a cross-thread isolation (race) check; needs a live LLM
  threads      same as ``agent`` with --concurrency defaulting to 500 (plan T8.1)

Exit status is non-zero when the error rate or p95 latency exceed the thresholds,
so it can gate a deploy. Point --base-url at the nginx balancer to exercise HA.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field

import httpx


@dataclass
class Stats:
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    isolation_violations: int = 0
    error_samples: list[str] = field(default_factory=list)

    def record(self, started: float, ok: bool, why: str = "") -> None:
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        if not ok:
            self.errors += 1
            if len(self.error_samples) < 5:
                self.error_samples.append(why)

    def summary(self, elapsed_s: float) -> dict:
        n = len(self.latencies_ms)
        lat = sorted(self.latencies_ms)

        def pct(p: float) -> float:
            return round(lat[min(n - 1, int(p * n))], 1) if n else 0.0

        return {
            "requests": n,
            "errors": self.errors,
            "error_rate": round(self.errors / n, 4) if n else 1.0,
            "isolation_violations": self.isolation_violations,
            "rps": round(n / elapsed_s, 1) if elapsed_s else 0.0,
            "latency_ms": {
                "mean": round(statistics.fmean(lat), 1) if n else 0.0,
                "p50": pct(0.50),
                "p95": pct(0.95),
                "p99": pct(0.99),
                "max": round(lat[-1], 1) if n else 0.0,
            },
            "error_samples": self.error_samples,
        }


async def _call(client: httpx.AsyncClient, stats: Stats, method: str, url: str, **kw):
    started = time.perf_counter()
    try:
        r = await client.request(method, url, **kw)
        stats.record(started, r.status_code < 400, f"{method} {url} -> {r.status_code}")
        return r
    except httpx.HTTPError as exc:
        stats.record(started, False, f"{method} {url}: {type(exc).__name__}")
        return None


async def scenario_health(client, stats, worker):
    await _call(client, stats, "GET", "/health")


async def scenario_governance(client, stats, worker):
    await _call(client, stats, "GET", "/governance/state")
    await _call(
        client,
        stats,
        "POST",
        "/governance/dispatch",
        json={"thread_id": f"load-{worker}", "target": "load-test", "payload": {"n": worker}},
    )


async def scenario_agent(client, stats, worker):
    thread = f"load-{uuid.uuid4().hex[:12]}"
    marker = f"marker-{thread}"
    for _ in range(2):
        await _call(
            client, stats, "POST", "/lg/agent/run", json={"thread_id": thread, "message": marker}
        )
    r = await _call(client, stats, "GET", f"/lg/agent/state/{thread}")
    if r is not None and r.status_code == 200:
        humans = [m for m in r.json()["messages"] if m["type"] == "human"]
        if len(humans) != 2 or any(m["content"] != marker for m in humans):
            stats.isolation_violations += 1


SCENARIOS = {
    "health": scenario_health,
    "governance": scenario_governance,
    "agent": scenario_agent,
    "threads": scenario_agent,
}


async def run(
    base_url: str,
    scenario: str,
    concurrency: int,
    duration_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = 60.0,
) -> tuple[Stats, float]:
    stats = Stats()
    step = SCENARIOS[scenario]
    deadline = time.perf_counter() + duration_s
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_s, limits=limits, transport=transport
    ) as client:

        async def worker(i: int):
            while time.perf_counter() < deadline:
                await step(client, stats, i)

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        return stats, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:4000")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="governance")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    ap.add_argument("--max-error-rate", type=float, default=0.01)
    ap.add_argument("--max-p95-ms", type=float, default=2000.0)
    args = ap.parse_args(argv)
    concurrency = args.concurrency or (500 if args.scenario == "threads" else 20)

    stats, elapsed = asyncio.run(run(args.base_url, args.scenario, concurrency, args.duration))
    summary = {"scenario": args.scenario, "concurrency": concurrency, **stats.summary(elapsed)}
    print(json.dumps(summary, indent=2))
    ok = (
        summary["error_rate"] <= args.max_error_rate
        and summary["latency_ms"]["p95"] <= args.max_p95_ms
        and summary["isolation_violations"] == 0
    )
    print("PASS" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
