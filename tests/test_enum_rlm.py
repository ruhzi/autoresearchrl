"""Stage-5 RLM environment: coverage ledger, spawn digests, submit gate, merge.

The enumerator subagent is mocked throughout -- these tests are about the
environment's bookkeeping, not the LLM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from arl import candidates
from arl.candidates import EnumerationEnv, run_enumerate_rlm
from arl.openrouter import ToolError
from arl.paperstore import PaperStore

PARSED = {
    "ok": True,
    "abstract": "We study widgets.",
    "sections": [
        {"level": 1, "title": "Intro", "kind": "intro", "chars": 0,
         "body": "widgets " * 50},
        {"level": 1, "title": "Method", "kind": "method", "chars": 0,
         "body": "gating " * 50},
        {"level": 1, "title": "Appendix", "kind": "appendix", "chars": 0,
         "body": "extra ablations " * 50},
    ],
    "n_tables": 1,
    "tables": [
        {"env": "table", "caption": "Results.", "label": "tab:r",
         "body": "a & 1 \\\\ b & 2 \\\\"},
    ],
}

PAPER = {"base_id": "0000.00000", "title": "Widgets", "abstract": "We study."}


def fake_samples(n, tag):
    return [{"technique": f"tech-{tag}-{i}", "description": "d",
             "setting": "set", "metrics": ["acc", "f1"],
             "effect": "improves", "effect_detail": ""} for i in range(n)]


@pytest.fixture()
def env():
    store = PaperStore(PARSED)
    calls = []

    def enum_fn(target_ids):
        calls.append(list(target_ids))
        return {"samples": fake_samples(2, len(calls)), "notes": ""}

    e = EnumerationEnv(store, enum_fn, max_batch_chars=10_000)
    e._test_calls = calls
    return e


def test_spawn_covers_targets_and_appends_findings(env):
    digest = env.dispatch("spawn_enumerator", {"targets": ["s0", "t0"]})
    assert "2" in digest                       # sample count visible to root
    assert env._test_calls == [["s0", "t0"]]
    assert len(env.findings) == 2
    assert all(f["_targets"] == ["s0", "t0"] for f in env.findings)
    cov = env.dispatch("coverage", {})
    assert "s1" in cov and "s2" in cov         # still uncovered, listed


def test_spawn_unknown_or_oversized_batch_refused(env):
    with pytest.raises(ToolError):
        env.dispatch("spawn_enumerator", {"targets": ["s99"]})
    big = PaperStore({"ok": True, "sections": [
        {"level": 1, "title": "S", "kind": "other", "chars": 0,
         "body": "x" * 20_000}], "tables": []})
    e = EnumerationEnv(big, lambda t: {"samples": [], "notes": ""},
                       max_batch_chars=10_000)
    with pytest.raises(ToolError, match="chars"):
        e.dispatch("spawn_enumerator", {"targets": ["s0"]})
    assert e.dispatch("coverage", {})          # s0 still uncovered


def test_spawn_empty_targets_refused(env):
    with pytest.raises(ToolError):
        env.dispatch("spawn_enumerator", {"targets": []})


def test_submit_gate(env):
    with pytest.raises(ToolError, match="s0"):
        env.dispatch("submit", {})
    env.dispatch("spawn_enumerator", {"targets": ["s0", "s1"]})
    with pytest.raises(ToolError, match="s2"):   # appendix uncovered
        env.dispatch("submit", {})
    env.dispatch("spawn_enumerator", {"targets": ["s2"]})
    env.dispatch("skip", {"targets": ["t0"], "reason": "raw data dump"})
    assert env.dispatch("submit", {})            # now passes


def test_failed_enumerator_becomes_toolerror_and_stays_uncovered():
    store = PaperStore(PARSED)

    def broken(_ids):
        raise ValueError("completion truncated at max_tokens")

    e = EnumerationEnv(store, broken, max_batch_chars=10_000)
    with pytest.raises(ToolError, match="split the batch"):
        e.dispatch("spawn_enumerator", {"targets": ["s0"]})
    assert "s0" in e._uncovered()          # not silently marked covered
    assert e.findings == []


def test_skip_requires_reason(env):
    with pytest.raises(ToolError):
        env.dispatch("skip", {"targets": ["t0"], "reason": ""})


def test_findings_brief_lists_all(env):
    env.dispatch("spawn_enumerator", {"targets": ["s0"]})
    env.dispatch("spawn_enumerator", {"targets": ["s1"]})
    brief = env.dispatch("findings_brief", {})
    assert "tech-1-0" in brief and "tech-2-1" in brief


def test_peek_and_grep_pass_through(env):
    assert "widgets" in env.dispatch("peek", {"target": "s0"})
    hits = env.dispatch("grep", {"query": "gating"})
    assert "s1" in hits
    with pytest.raises(ToolError):
        env.dispatch("peek", {"target": "nope"})


def test_unknown_tool_is_toolerror(env):
    with pytest.raises(ToolError):
        env.dispatch("frobnicate", {})


def test_merge_dedupes_across_batches():
    dup = {"technique": "Gating", "description": "d", "setting": "WB",
           "metrics": ["acc"], "effect": "improves", "effect_detail": ""}
    a = dict(dup, _targets=["s0"])
    b = dict(dup, _targets=["s1"])          # same (technique, setting)
    c = dict(dup, setting="ImageNet", _targets=["s1"])   # different setting
    d2 = dict(dup, metrics=["latency"], _targets=["s1"])  # only metric differs
    merged = candidates.merge_findings([a, b, c, d2])
    assert len(merged) == 2                 # b AND d2 collapse into a
    assert merged[0]["_targets"] == ["s0"]  # first occurrence wins


# ----------------------------------------------------------- checkpointing

def test_spawn_and_skip_write_checkpoint(tmp_path):
    from arl.util import read_json
    store = PaperStore(PARSED)
    ck = tmp_path / "ck.json"
    e = EnumerationEnv(store, lambda t: {"samples": fake_samples(1, "c"),
                                         "notes": "n1"},
                       max_batch_chars=10_000, checkpoint_path=ck)
    e.dispatch("spawn_enumerator", {"targets": ["s0"]})
    d = read_json(ck)
    assert d["ledger"] == {"s0": "covered"}
    assert len(d["findings"]) == 1 and d["notes"] == ["n1"]
    e.dispatch("skip", {"targets": ["t0"], "reason": "raw dump"})
    d = read_json(ck)
    assert d["ledger"]["t0"] == "skipped"


def test_restore_seeds_ledger_and_findings_and_gate():
    store = PaperStore(PARSED)
    e = EnumerationEnv(store, lambda t: {"samples": [], "notes": ""},
                       max_batch_chars=10_000)
    e.restore({"ledger": {"s0": "covered", "s1": "covered", "s2": "covered",
                          "t0": "skipped", "zombie99": "covered"},
               "findings": fake_samples(3, "prev"),
               "batches": [["s0", "s1"]],
               "skipped": [{"targets": ["t0"], "reason": "r"}],
               "notes": ["old note"]})
    assert "zombie99" not in e.ledger        # stale ids dropped
    assert len(e.findings) == 3
    assert e.dispatch("submit", {})          # gate satisfied from checkpoint


def test_rlm_run_resumes_from_checkpoint_and_clears_it(monkeypatch, tmp_path):
    from arl.util import write_json
    monkeypatch.setattr(candidates.config, "CANDIDATES_PARTIAL_DIR", tmp_path)
    monkeypatch.setattr(candidates.config, "RLM_FAST_PATH", False)

    bid = "9999.11111"
    write_json(tmp_path / f"{bid}.json",
               {"ledger": {"s0": "covered", "s1": "covered"},
                "findings": fake_samples(2, "prev"),
                "batches": [["s0", "s1"]], "skipped": [], "notes": []})

    def fake_subset(paper, store, target_ids):
        return {"samples": fake_samples(1, "new"), "notes": ""}, {}

    monkeypatch.setattr(candidates, "run_enumerate_subset", fake_subset)

    def fake_chat_tools(model, system, user, tools, dispatch, **kw):
        assert "RESUMING" in user and "2 findings" in user
        dispatch("spawn_enumerator", {"targets": ["s2"]})
        dispatch("skip", {"targets": ["t0"], "reason": "raw dump"})
        return {"name": "submit", "args": {},
                "result": dispatch("submit", {})}

    monkeypatch.setattr(candidates, "chat_tools", fake_chat_tools)
    paper = dict(PAPER, base_id=bid)
    res, _ = run_enumerate_rlm(paper, PARSED)
    assert len(res["samples"]) == 3           # 2 restored + 1 new
    assert not (tmp_path / f"{bid}.json").exists()   # checkpoint cleared


# --------------------------------------------------------------- fast path

def test_fast_path_skips_root_loop(monkeypatch):
    called = {}

    def fake_subset(paper, store, target_ids):
        called["targets"] = list(target_ids)
        return {"samples": fake_samples(1, "fp"), "notes": ""}, {}

    monkeypatch.setattr(candidates, "run_enumerate_subset", fake_subset)
    monkeypatch.setattr(candidates.config, "RLM_FAST_PATH", True)
    monkeypatch.setattr(candidates.config, "RLM_SUBAGENT_MAX_CHARS", 10 ** 9)

    def boom(*a, **k):
        raise AssertionError("root loop must not run on the fast path")

    monkeypatch.setattr(candidates, "chat_tools", boom)
    res, _ = run_enumerate_rlm(PAPER, PARSED)
    assert called["targets"] == ["s0", "s1", "s2", "t0"]
    assert res["_rlm"]["fast_path"] is True
    assert len(res["samples"]) == 1
