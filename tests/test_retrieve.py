"""Stage-6 retrieval env: pointer validation, verbatim slicing, number checks.

No LLM anywhere -- these tests drive the environment's dispatch directly and
assert that nothing model-typed can become payload.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from arl import retrieve
from arl.openrouter import ToolError
from arl.paperstore import PaperStore
from arl.retrieve import RetrieveEnv, run_retrieve_sample

PARSED = {
    "ok": True,
    "abstract": "We study widgets.",
    "sections": [
        {"level": 1, "title": "Method", "kind": "method", "chars": 0,
         "body": "We propose GADGET, a widget with a gating mechanism. "
                 "Setting sigma to zero recovers the standard widget."},
        {"level": 1, "title": "Setup", "kind": "results", "chars": 0,
         "body": "All models train for 10 epochs on WidgetBench at "
                 "resolution 224."},
    ],
    "n_tables": 1,
    "tables": [
        {"env": "table", "caption": "Main results.", "label": "tab:main",
         "body": "Model & Acc \\\\ baseline & 71.2 \\\\ GADGET & 74.9 \\\\"},
    ],
}

GOOD = {
    "spans": [
        {"target": "s0", "start_anchor": "We propose GADGET",
         "end_anchor": "standard widget."},
        {"target": "s1", "start_anchor": "All models",
         "end_anchor": "resolution 224."},
    ],
    "evidence": [{"target": "t0", "start_anchor": "baseline & 71.2",
                  "end_anchor": "74.9 \\\\"}],
    "measurements": [
        {"benchmark": "WidgetBench", "metric": "Acc",
         "baseline_cell": "71.2", "method_cell": "74.9",
         "metric_unit": "%", "direction_per_paper": "higher_is_better"},
    ],
    "baseline_label": "baseline",
    "method_label": "GADGET",
    "comparison_kind": "baseline_vs_method",
    "isolates_single_technique": True,
    "table_ref": "tab:main",
    "confidence": "high",
}


@pytest.fixture()
def env():
    return RetrieveEnv(PaperStore(PARSED))


def test_happy_path_slices_everything_from_source(env):
    out = env.dispatch("submit_found", dict(GOOD))
    assert "accepted" in out
    ext = env.result["extraction"]
    # payload is exact source bytes
    assert ext["source_spans"][0] == PARSED["sections"][0]["body"]
    assert ext["source_spans"][1] == PARSED["sections"][1]["body"]
    assert ext["evidence"] == ["baseline & 71.2 \\\\ GADGET & 74.9 \\\\"]
    # numbers parsed from verified cells
    assert ext["measurements"][0]["baseline_value"] == 71.2
    assert ext["measurements"][0]["method_value"] == 74.9
    # pointers kept for audit
    assert ext["span_pointers"] == GOOD["spans"]
    assert env.result["found"] is True


def test_bad_span_anchor_is_toolerror(env):
    bad = dict(GOOD, spans=[{"target": "s0", "start_anchor": "never there",
                             "end_anchor": "standard widget."}])
    with pytest.raises(ToolError, match="not found"):
        env.dispatch("submit_found", bad)
    assert env.result is None            # nothing accepted


def _with_cells(baseline_cell="71.2", method_cell="74.9"):
    mm = dict(GOOD["measurements"][0], baseline_cell=baseline_cell,
              method_cell=method_cell)
    return dict(GOOD, measurements=[mm])


def test_cell_must_occur_in_evidence(env):
    with pytest.raises(ToolError, match="71.9"):
        env.dispatch("submit_found", _with_cells(baseline_cell="71.9"))


def test_cell_must_hold_exactly_one_number(env):
    with pytest.raises(ToolError):
        env.dispatch("submit_found", _with_cells(baseline_cell="baseline"))
    with pytest.raises(ToolError):
        env.dispatch("submit_found",
                     _with_cells(method_cell="71.2 \\\\ GADGET & 74.9"))


def test_substring_of_longer_number_rejected(env):
    # "71.2" must not be accepted just because it is a substring of "71.25";
    # here "4.9" is a substring of "74.9" but is not a number in the row.
    with pytest.raises(ToolError, match="not one of the numbers"):
        env.dispatch("submit_found", _with_cells(method_cell="4.9"))


def test_empty_measurements_refused(env):
    with pytest.raises(ToolError, match="measurements"):
        env.dispatch("submit_found", dict(GOOD, measurements=[]))


def test_spans_required_nonempty(env):
    with pytest.raises(ToolError):
        env.dispatch("submit_found", dict(GOOD, spans=[]))


def test_not_found_requires_reason(env):
    with pytest.raises(ToolError):
        env.dispatch("submit_not_found", {"reason": " "})
    out = env.dispatch("submit_not_found",
                       {"reason": "comparison only reported as a delta"})
    assert out
    assert env.result["found"] is False
    assert env.result["extraction"] is None
    assert "delta" in env.result["not_found_reason"]


def test_grep_and_read_passthrough(env):
    assert "t0" in env.dispatch("grep", {"query": "71.2"})
    assert "GADGET" in env.dispatch("read", {"target": "s0"})
    with pytest.raises(ToolError):
        env.dispatch("read", {"target": "zzz"})


# ------------------------------------------------------------ run wrapper

def test_run_retrieve_sample_returns_stage7_envelope(monkeypatch):
    def fake_chat_tools(model, system, user, tools, dispatch, **kw):
        # the sample handle must be in the user message, the paper must not
        assert "gating mechanism" in user            # technique handle
        assert "We propose GADGET" not in user        # no paper text
        return {"name": "submit_found", "args": dict(GOOD),
                "result": dispatch("submit_found", dict(GOOD))}

    monkeypatch.setattr(retrieve, "chat_tools", fake_chat_tools)
    s5 = {"sample_id": "x-s000", "technique": "gating mechanism",
          "description": "gating vs none", "setting": "WidgetBench",
          "metrics": ["Acc"], "_n_in_paper": 3}
    paper = {"base_id": "x", "title": "Widgets", "abstract": "We study."}
    res, _ = run_retrieve_sample(paper, PARSED, s5)
    assert res["found"] is True
    assert res["extraction"]["measurements"][0]["baseline_value"] == 71.2
    assert res["not_found_reason"] == ""
