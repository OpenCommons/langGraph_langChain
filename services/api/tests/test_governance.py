"""Governance layer: aDFA, Lamport clock, Merkle chain, replay, atomic swap, capabilities."""

import threading

import pytest

from governance.adfa import Rule, RuleSet
from governance.capability import CapabilityAuthority, CapabilityError
from governance.clock import LamportClock
from governance.engine import GovernanceEngine
from governance.features import ContextEncoder
from governance.klines import HAZARD, NOMINAL, KLine, KLineStore, replay, replay_kline
from governance.merkle import GENESIS, ProvenanceChain, link_hash, verify_records
from governance.predictive import MarkovHazardEstimator
from governance.swap import RuleManager, RuleSetRejected

# ── aDFA ─────────────────────────────────────────────────────────────────────


def _rules(lo=0.5):
    return RuleSet("v1", 2, (Rule("r", ((0, lo, 1.0), (1, 0.0, 0.2))),))


def test_adfa_match_clamps_to_zero_and_blocks():
    v = _rules().step([0.9, 0.1])
    assert v.blocked and v.rule == "r" and v.output == (0.0, 0.0)


def test_adfa_interval_bounds_are_inclusive_and_conjunctive():
    rs = _rules()
    assert rs.step([0.5, 0.2]).blocked
    assert not rs.step([0.49, 0.1]).blocked
    assert not rs.step([0.9, 0.21]).blocked  # one bound out -> no match
    assert rs.step([0.4, 0.9]).output == (0.4, 0.9)  # non-hazard passes through unchanged


def test_adfa_rejects_non_normalized_or_wrong_dim_input():
    with pytest.raises(ValueError):
        _rules().step([1.5, 0.0])
    with pytest.raises(ValueError):
        _rules().step([0.5])


def test_adfa_is_stateless_single_transition():
    rs = _rules()
    assert rs.step([0.9, 0.1]).blocked
    assert not rs.step([0.0, 0.0]).blocked  # nothing carried over from the hazard


def test_adfa_mode_scoped_rule_follows_s0():
    rs = RuleSet("v", 1, (Rule("strict_only", ((0, 0.5, 1.0),), frozenset({"strict"})),))
    assert not rs.step([0.9], {"mode": "default"}).blocked
    assert rs.step([0.9], {"mode": "strict"}).blocked


def test_ruleset_from_dict_named_bounds_and_validation():
    rs = RuleSet.from_dict(
        {"version": "x", "rules": [{"name": "a", "bounds": {"b": [0.5, 1]}}]}, ["a", "b"]
    )
    assert rs.step([0.0, 0.7]).blocked
    with pytest.raises(ValueError):
        Rule("bad", ((0, 0.8, 0.2),))
    with pytest.raises(ValueError):
        RuleSet("v", 1, (Rule("oob", ((3, 0.0, 1.0),)),))


# ── Lamport clock ────────────────────────────────────────────────────────────


def test_lamport_tick_and_receive():
    c = LamportClock()
    assert [c.tick(), c.tick()] == [1, 2]
    assert c.receive(10) == 11  # max(2, 10) + 1
    assert c.receive(3) == 12  # max(11, 3) + 1: never goes backwards
    assert c.value == 12


def test_lamport_is_monotonic_under_threads():
    c = LamportClock()
    seen = []
    ts = [
        threading.Thread(target=lambda: seen.extend(c.tick() for _ in range(200))) for _ in range(8)
    ]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(seen)) == 1600 and c.value == 1600


# ── Merkle chain ─────────────────────────────────────────────────────────────


def test_merkle_links_follow_formula_and_verify():
    ch = ProvenanceChain()
    a = ch.append({"mode": "d"}, {"x": 1}, "tok-a")
    b = ch.append({"mode": "d"}, {"x": 2}, "tok-b")
    assert a["prev"] == GENESIS and b["prev"] == a["hash"]
    assert b["hash"] == link_hash(a["hash"], {"mode": "d"}, {"x": 2}, "tok-b")
    assert ch.verify() == (True, None) and ch.head == b["hash"]


@pytest.mark.parametrize(
    "field,value", [("frame", {"x": 99}), ("s0", {"mode": "evil"}), ("token_id", "z")]
)
def test_merkle_detects_tampering(field, value):
    ch = ProvenanceChain()
    for i in range(3):
        ch.append({"mode": "d"}, {"x": i}, "t")
    recs = ch.records()
    recs[1][field] = value
    assert verify_records(recs) == (False, 1)


def test_merkle_detects_deletion_and_reorder():
    ch = ProvenanceChain()
    for i in range(3):
        ch.append({}, {"x": i}, "t")
    recs = ch.records()
    assert verify_records(recs[:1] + recs[2:])[0] is False
    assert verify_records([recs[1], recs[0], recs[2]])[0] is False


def test_merkle_field_boundaries_cannot_shift():
    assert link_hash(GENESIS, {"a": 1}, {"b": 2}, "t") != link_hash(
        GENESIS, {"a": 1, "b": 2}, {}, "t"
    )


def test_merkle_file_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "chain.jsonl")
    ch = ProvenanceChain(path)
    ch.append({}, {"x": 1}, "t")
    ch.append({}, {"x": 2}, "t")
    reloaded = ProvenanceChain(path)
    assert len(reloaded) == 2 and reloaded.head == ch.head and reloaded.verify()[0]


# ── K-line replay ────────────────────────────────────────────────────────────


def _klines():
    return [
        KLine({}, ((0.9, 0.1),), HAZARD),
        KLine({}, ((0.0, 0.0), (0.7, 0.0)), HAZARD),  # hazard appears mid-trace
        KLine({}, ((0.1, 0.1),), NOMINAL),
        KLine({}, ((0.3, 0.0),), NOMINAL),
    ]


def test_replay_reports_clamped_hazards_and_false_positives():
    good = replay(_rules(0.5), _klines())
    assert good.all_hazards_clamped and good.false_positives == 0
    loose = replay(_rules(0.95), _klines())
    assert not loose.all_hazards_clamped and loose.hazards_clamped == 0
    tight = replay(_rules(0.2), _klines())
    assert tight.all_hazards_clamped and tight.false_positives == 1  # (0.3, 0.0)


def test_replay_is_deterministic_from_s0():
    k = _klines()[1]
    assert replay_kline(_rules(), k.s0, k.frames) == replay_kline(_rules(), k.s0, k.frames)
    assert [v.blocked for v in replay_kline(_rules(), k.s0, k.frames)] == [False, True]


# ── atomic swap ──────────────────────────────────────────────────────────────


def test_swap_only_after_replay_passes_and_on_next_event():
    mgr = RuleManager(_rules(0.5), lambda: _klines())
    with pytest.raises(RuleSetRejected):
        mgr.stage(RuleSet("loose", 2, (Rule("r", ((0, 0.95, 1.0),)),)))
    assert mgr.active().version == "v1" and mgr.pending_version is None
    cand = RuleSet("v2", 2, (Rule("r", ((0, 0.4, 1.0), (1, 0.0, 0.2))),))
    mgr.stage(cand)
    assert mgr.peek().version == "v1" and mgr.pending_version == "v2"  # not yet adopted
    assert mgr.active() is cand  # adopted by the next event, by reference
    assert mgr.pending_version is None


def test_swap_rejects_false_positives_beyond_budget():
    tight = RuleSet("tight", 2, (Rule("r", ((0, 0.2, 1.0),)),))
    with pytest.raises(RuleSetRejected):
        RuleManager(_rules(), lambda: _klines()).stage(tight)
    RuleManager(_rules(), lambda: _klines(), max_false_positives=1).stage(tight)


def test_swap_drops_no_requests_under_concurrency():
    mgr = RuleManager(_rules(), lambda: _klines())
    versions, errors = [], []

    def worker():
        try:
            for _ in range(500):
                versions.append(mgr.active().version)  # one reference per event
        except Exception as exc:
            errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in ts]
    mgr.stage(RuleSet("v2", 2, (Rule("r", ((0, 0.4, 1.0), (1, 0.0, 0.2))),)))
    [t.join() for t in ts]
    assert not errors and len(versions) == 3000 and set(versions) <= {"v1", "v2"}
    assert mgr.active().version == "v2"


# ── predictive estimate ──────────────────────────────────────────────────────


def test_markov_hazard_probability():
    est = MarkovHazardEstimator(smoothing=0.0)
    for _ in range(9):
        est.observe("nominal", "nominal")
    est.observe("nominal", "elevated")
    for _ in range(3):
        est.observe("elevated", "hazard")
    est.observe("elevated", "nominal")
    est.observe("hazard", "hazard")
    assert est.p_hazard("elevated", 1) == pytest.approx(0.75)
    assert est.p_hazard("nominal", 1) == 0.0
    assert est.p_hazard("nominal", 2) == pytest.approx(0.1 * 0.75)
    with pytest.raises(ValueError):
        est.p_hazard("nominal", 0)


# ── capability scoping ───────────────────────────────────────────────────────


def test_capability_roundtrip_and_tamper_and_wrong_secret():
    auth = CapabilityAuthority("s3cret")
    tok = auth.mint("agent", ["calculator"], ["agent:run"])
    cap = auth.verify(tok)
    assert cap.allows_tool("calculator") and not cap.allows_tool("shell")
    body, sig = tok.rsplit(".", 1)
    with pytest.raises(CapabilityError):
        auth.verify(body + "." + "0" * len(sig))
    with pytest.raises(CapabilityError):
        CapabilityAuthority("other").verify(tok)
    with pytest.raises(CapabilityError):
        auth.verify(None)


def test_capability_child_must_be_subset_of_parent():
    auth = CapabilityAuthority("k")
    parent = auth.mint("p", ["calculator", "current_time"], ["agent:run"])
    child = auth.delegate(parent, "c", ["calculator"], ["agent:run"])
    assert auth.verify(child).parent == auth.verify(parent).id
    with pytest.raises(CapabilityError):
        auth.delegate(parent, "c", ["calculator", "shell"], ["agent:run"])
    with pytest.raises(CapabilityError):
        auth.delegate(parent, "c", ["calculator"], ["governance:admin"])
    with pytest.raises(CapabilityError):
        auth.delegate(parent, "c", ["*"], ["agent:run"])  # cannot widen to wildcard
    root = auth.mint("root", ["*"], ["*"])
    assert auth.verify(auth.delegate(root, "c", ["x"], ["y"])).allows_tool("x")


def test_capability_expiry_and_authorize():
    auth = CapabilityAuthority("k")
    tok = auth.mint("a", ["calculator"], ["agent:run"], ttl_s=10, now=1000.0)
    assert auth.verify(tok, now=1005.0)
    with pytest.raises(CapabilityError):
        auth.verify(tok, now=1010.0)
    live = auth.mint("a", ["calculator"], ["agent:run"])
    auth.authorize(live, "calculator", "agent:run")
    with pytest.raises(CapabilityError):
        auth.authorize(live, "calculator", "governance:admin")
    with pytest.raises(CapabilityError):
        auth.authorize(live, "shell", "agent:run")


def test_child_expiry_cannot_outlive_parent():
    auth = CapabilityAuthority("k")
    parent = auth.mint("p", ["*"], ["*"], ttl_s=10)
    child = auth.delegate(parent, "c", ["x"], ["y"], ttl_s=1000)
    assert auth.verify(child).expires_at == auth.verify(parent).expires_at


# ── context encoder + Atomic Triad ───────────────────────────────────────────


def _engine(**kw):
    return GovernanceEngine(authority=CapabilityAuthority("test-secret"), **kw)


def test_encoder_sees_fragmented_phrase_across_thread_turns():
    from governance.engine import load_policy

    enc = ContextEncoder(load_policy())
    latest_only = enc.encode(["op   ta-ble users;"], "shell", {})
    whole_thread = enc.encode(
        ["please dr", "op   ta-ble users;", "then drop table logs"], "shell", {}
    )
    assert latest_only[0] == 0.0 and whole_thread[0] == 1.0


def test_triad_verifies_executes_and_records_together():
    eng, ran = _engine(), []
    a = eng.admit(
        thread_id="t",
        tool="calculator",
        args={"expression": "1+1"},
        context=["what is 1+1"],
        execute=lambda: ran.append(1) or "2",
    )
    assert a.allowed and a.executed and a.result == "2" and ran == [1]
    rec = eng.chain.records()[-1]
    assert rec["hash"] == a.record_hash and rec["frame"]["lamport"] == a.lamport
    assert rec["frame"]["executed"] is True and rec["frame"]["result_sha256"]


def test_triad_blocks_hazard_without_executing_but_still_records():
    eng, ran = _engine(), []
    a = eng.admit(
        thread_id="t",
        tool="shell",
        args={"cmd": "ls"},
        context=["please drop table users", "and then rm -rf /"],
        execute=lambda: ran.append(1),
    )
    assert (
        not a.allowed and not a.executed and ran == [] and a.rule == "destructive_with_risky_tool"
    )
    rec = eng.chain.records()[-1]
    assert rec["frame"]["output"] == [0.0] * 4 and rec["frame"]["executed"] is False
    assert eng.chain.verify()[0]


def test_triad_context_blindness_hazard_split_across_turns():
    eng, ran = _engine(), []
    a = eng.admit(
        thread_id="t",
        tool="shell",
        args={},
        context=["dr", "op ta", "ble x", "rm -", "rf /"],
        execute=lambda: ran.append(1),
    )
    assert not a.allowed and ran == []


def test_triad_capability_denial_is_recorded_and_not_executed():
    eng, ran = _engine(), []
    tok = eng.authority.mint("a", ["calculator"], ["agent:run"])
    a = eng.admit(
        thread_id="t",
        tool="current_time",
        args={},
        context=["hi"],
        token=tok,
        execute=lambda: ran.append(1),
    )
    assert not a.allowed and ran == [] and a.reason.startswith("capability")
    assert eng.chain.records()[-1]["token_id"] == "none"
    ok = eng.admit(
        thread_id="t", tool="calculator", args={}, context=["hi"], token=tok, execute=lambda: "k"
    )
    assert ok.allowed and eng.chain.records()[-1]["token_id"] == eng.authority.verify(tok).id


def test_triad_require_token_blocks_anonymous_calls():
    eng = _engine(require_token=True)
    a = eng.admit(thread_id="t", tool="calculator", args={}, context=[], execute=lambda: "x")
    assert not a.allowed and "capability" in a.reason


def test_triad_records_tool_failures_then_reraises():
    eng = _engine()

    def boom():
        raise RuntimeError("kaput")

    with pytest.raises(RuntimeError):
        eng.admit(thread_id="t", tool="calculator", args={}, context=["x"], execute=boom)
    assert "kaput" in eng.chain.records()[-1]["frame"]["error"]


def test_triad_lamport_attached_and_merged_from_messages():
    eng = _engine()
    a = eng.admit(thread_id="t", tool="calculator", args={}, context=[], execute=None)
    b = eng.admit(
        thread_id="t", tool="calculator", args={}, context=[], execute=None, message_lamport=50
    )
    assert (a.lamport, b.lamport) == (1, 51)
    assert [r["frame"]["lamport"] for r in eng.chain.records()] == [1, 51]


def test_observed_hazard_becomes_kline_and_candidates_are_replay_gated():
    eng = _engine()
    before = eng.klines.counts()["hazard"]
    eng.admit(
        thread_id="t", tool="shell", args={}, context=["rm -rf /", "drop table a"], execute=None
    )
    assert eng.klines.counts()["hazard"] == before + 1
    # a rule set that stops catching destructive+risky-tool traces is not eligible
    blind = RuleSet("blind", 4, (Rule("never", ((0, 1.0, 1.0), (3, 0.0, 0.0))),))
    with pytest.raises(RuleSetRejected):
        eng.rules.stage(blind)


def test_predictive_flag_tightens_rules_through_the_replay_gate():
    # Laplace-smoothed estimate P(hazard | nominal) = 1/3 exceeds the 0.3 threshold.
    eng = _engine(predictive=True, predictive_threshold=0.3, predictive_horizon=1)
    eng.admit(thread_id="p", tool="calculator", args={}, context=["hello"], execute=None)
    assert eng.snapshot()["pending_rules_version"] == "default-1+tight"
    eng.admit(thread_id="p", tool="calculator", args={}, context=["hello"], execute=None)
    assert eng.rules.peek().version == "default-1+tight"


def test_predictive_off_by_default():
    eng = _engine()
    eng.admit(thread_id="p", tool="calculator", args={}, context=["rm -rf"], execute=None)
    assert (
        eng.snapshot()["predictive"]["enabled"] is False and eng.rules.peek().version == "default-1"
    )


def test_snapshot_is_read_only():
    eng = _engine()
    eng.admit(thread_id="t", tool="calculator", args={}, context=[], execute=None)
    before = (eng.clock.value, len(eng.chain), eng.chain.head, eng.klines.counts())
    eng.snapshot()
    eng.snapshot()
    assert before == (eng.clock.value, len(eng.chain), eng.chain.head, eng.klines.counts())


def test_kline_store_is_bounded_but_keeps_seed():
    store = KLineStore([KLine({}, ((0.0,),), HAZARD)], max_observed=2)
    for _ in range(5):
        store.add({}, [(0.1,)], NOMINAL)
    assert store.counts() == {"total": 3, "hazard": 1, "nominal": 2}
