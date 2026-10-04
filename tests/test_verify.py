"""Stage-8 verifier: LLM judge over the finished training sample. Mocked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl import verify
from arl.verify import run_verify_sample

COMPOSED = {
    "sample_id": "x-s000",
    "ok": True,
    "prompt": ("I've been brainstorming a gating mechanism g(x) = "
               "\\sigma(Wx). My baseline gets 71.2% Acc.\n\n"
               "Two questions about this setup:\n(a) ...\n(b) ..."),
    "body": "I've been brainstorming a gating mechanism...",
    "target": {"direction": "helps", "direction_detail": "",
               "measurements": [{"benchmark": "WidgetBench", "metric": "Acc",
                                 "baseline_value": 71.2,
                                 "method_value": 74.9, "unit": "%"}]},
    "meta": {"base_id": "2608.00001", "component": "gating mechanism"},
}
ROW_SPANS = ["We propose GADGET, a widget with a gating mechanism "
             "g(x) = \\sigma(Wx).",
             "All models train for 10 epochs on WidgetBench."]


def _mock(monkeypatch, verdict):
    seen = {}

    def fake_chat_json(model, system, user, schema, **kw):
        seen["system"], seen["user"] = system, user
        return verdict, {}

    monkeypatch.setattr(verify, "chat_json", fake_chat_json)
    return seen


CLEAN = {"outcome_reference": False, "outcome_evidence": "",
         "faithful": True, "fidelity_issues": []}


def test_clean_sample_passes(monkeypatch):
    seen = _mock(monkeypatch, dict(CLEAN))
    res, _ = run_verify_sample(COMPOSED, ROW_SPANS)
    assert res["passed"] is True
    # judge sees the prompt and the source spans...
    assert "brainstorming" in seen["user"] and "GADGET" in seen["user"]
    # ...but never the answer
    assert "74.9" not in seen["system"] + seen["user"]
    assert "helps" not in seen["system"] + seen["user"]


def test_outcome_reference_fails(monkeypatch):
    _mock(monkeypatch, dict(CLEAN, outcome_reference=True,
                            outcome_evidence="says it works better"))
    res, _ = run_verify_sample(COMPOSED, ROW_SPANS)
    assert res["passed"] is False
    assert "outcome" in " ".join(res["reasons"])


def test_unfaithful_fails(monkeypatch):
    _mock(monkeypatch, dict(CLEAN, faithful=False,
                            fidelity_issues=["epochs changed from 10 to 12"]))
    res, _ = run_verify_sample(COMPOSED, ROW_SPANS)
    assert res["passed"] is False
    assert any("epochs" in r for r in res["reasons"])


def test_method_value_in_prompt_fails_mechanically(monkeypatch):
    _mock(monkeypatch, dict(CLEAN))     # judge says clean; Python still catches
    leaked = dict(COMPOSED, prompt=COMPOSED["prompt"] + " Maybe 74.9?")
    res, _ = run_verify_sample(leaked, ROW_SPANS)
    assert res["passed"] is False
    assert any("method value" in r for r in res["reasons"])
