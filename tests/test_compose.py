"""Stage-8 composer: the fidelity gate that lets an LLM rewrite verbatim
material into a researcher-voice prompt without inventing, dropping numbers,
or leaking the answer. All LLM calls mocked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from arl import compose
from arl.compose import (assemble_prompt, compose_target, fidelity_check,
                         run_compose_sample)

ROW = {
    "sample_id": "x-s000",
    "component": "gating mechanism",
    "setting": "WidgetBench, ResNet-50",
    "metrics": ["Acc"],
    "measurements": [
        {"benchmark": "WidgetBench", "metric_raw": "Acc",
         "metric": "accuracy", "metric_unit": "%",
         "baseline_value": 71.2, "method_value": 74.9},
    ],
    "source_spans": [
        "We propose GADGET, a widget with a gating mechanism "
        "g(x) = \\sigma(Wx). Setting sigma to zero recovers the standard "
        "widget.",
        "All models train for 10 epochs on WidgetBench at resolution 224.",
    ],
    "baseline_label": "baseline",
    "method_label": "GADGET",
    "needs_review": False,
    "base_id": "2608.00001",
    "title": "Widgets",
}

GOOD_BODY = (
    "I've been brainstorming a widget with a gating mechanism "
    "g(x) = \\sigma(Wx) -- setting sigma to zero just recovers the standard "
    "widget. I'm training for 10 epochs on WidgetBench at resolution 224 "
    "with a ResNet-50. My baseline currently gets 71.2% Acc."
)


# ---------------------------------------------------------------- gate

def test_clean_body_passes():
    assert fidelity_check(GOOD_BODY, ROW) == []


def test_invented_number_rejected():
    bad = GOOD_BODY.replace("10 epochs", "12 epochs")
    fails = fidelity_check(bad, ROW)
    assert any("number" in f for f in fails)


def test_missing_baseline_rejected():
    bad = GOOD_BODY.replace(" My baseline currently gets 71.2% Acc.", "")
    fails = fidelity_check(bad, ROW)
    assert any("baseline" in f for f in fails)


def test_method_value_leak_rejected():
    bad = GOOD_BODY + " I expect around 74.9%."
    fails = fidelity_check(bad, ROW)
    assert any("method value" in f for f in fails)


def test_method_equal_to_baseline_is_exempt():
    mm = dict(ROW["measurements"][0], method_value=71.2)
    row = dict(ROW, measurements=[mm], direction="neutral")
    assert fidelity_check(GOOD_BODY, row) == []


def test_invented_latex_rejected():
    bad = GOOD_BODY.replace("\\sigma(Wx)", "\\tanh(Wx)")
    fails = fidelity_check(bad, ROW)
    assert any("latex" in f for f in fails)


def test_outcome_language_is_not_pattern_matched():
    # Outcome screening is semantic (stage 8's LLM judge), not a regex here:
    # goal phrasing and even result-y wording pass the MECHANICAL gate.
    ok = GOOD_BODY + " The design is meant to improve gradient flow."
    assert fidelity_check(ok, ROW) == []


# ------------------------------------------------------------ assembly

def test_assemble_prompt_appends_uniform_footer():
    p = assemble_prompt(GOOD_BODY, ROW)
    assert p.startswith(GOOD_BODY)
    assert "predict the resulting value for each" in p
    assert "gating mechanism" in p and "Acc" in p
    assert "74.9" not in p                      # answer never in prompt


def test_target_is_measurements_only():
    t = compose_target(ROW)
    assert set(t) == {"measurements"}, "the graded target carries no verdict"
    assert t["measurements"] == [{"benchmark": "WidgetBench",
                                  "metric": "Acc", "baseline_value": 71.2,
                                  "method_value": 74.9, "unit": "%"}]


def test_prompt_asks_only_for_values():
    """The prompt and the graded target must ask for the same thing: every
    slot named in the question is scored, and no unscored question is put to
    the model alongside it."""
    q = assemble_prompt("body.", ROW)
    assert "switching from baseline (my baseline above) to GADGET" in q
    assert "predict the resulting value for each: WidgetBench Acc (in %)." in q
    assert q.count("?") == 0                    # a single imperative ask
    for verdict_word in ("(a)", "(b)", "improve", "degrade",
                         "essentially unchanged", "If mixed"):
        assert verdict_word not in q, f"direction ask leaked back in: {verdict_word}"


# --------------------------------------------------------------- retry

def test_run_compose_retries_with_feedback_then_succeeds(monkeypatch):
    import json as _json
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append(messages[1]["content"])
        if len(calls) == 1:
            return _json.dumps({"body": GOOD_BODY + " I expect around 74.9%."}), {}
        return _json.dumps({"body": GOOD_BODY}), {}

    monkeypatch.setattr(compose, "chat", fake_chat)
    res, _ = run_compose_sample(ROW)
    assert res["ok"] is True
    assert res["prompt"].startswith(GOOD_BODY)
    assert res["attempts"] == 2
    assert "method value" in calls[1]           # failure fed back


def test_run_compose_gives_up_after_retries(monkeypatch):
    import json as _json

    def fake_chat(model, messages, **kw):
        return _json.dumps({"body": "It uses 99.9 epochs."}), {}  # invents

    monkeypatch.setattr(compose, "chat", fake_chat)
    monkeypatch.setattr(compose.config, "COMPOSE_RETRIES", 1)
    res, _ = run_compose_sample(ROW)
    assert res["ok"] is False
    assert res["failures"]


def test_composer_never_sees_answer(monkeypatch):
    import json as _json
    seen = {}

    def fake_chat(model, messages, **kw):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[1]["content"]
        return _json.dumps({"body": GOOD_BODY}), {}

    monkeypatch.setattr(compose, "chat", fake_chat)
    run_compose_sample(ROW)
    blob = seen["system"] + seen["user"]
    assert "74.9" not in blob
    assert "helps" not in blob


def test_footer_names_every_scored_slot():
    # Values are the whole target now, so the question must enumerate exactly
    # the slots compose_target grades -- an unnamed slot is an ungradable one.
    row = dict(ROW, measurements=[
        ROW["measurements"][0],
        {"benchmark": "OtherBench", "metric_raw": "F1", "metric_unit": "",
         "baseline_value": 0.5, "method_value": 0.6},
    ])
    p = assemble_prompt(GOOD_BODY, row)
    asked = p.rsplit("predict the resulting value for each: ", 1)[1]
    assert asked == "WidgetBench Acc (in %); OtherBench F1."
    assert len(asked.split("; ")) == len(compose_target(row)["measurements"])


def test_footer_orients_by_arm_labels():
    # "A vs B" components are ambiguous about which arm is the baseline;
    # the footer must say so using the row's own labels
    p = assemble_prompt(GOOD_BODY, ROW)
    assert "switching from baseline (my baseline above) to GADGET" in p
    assert "adding the" not in p


def test_footer_falls_back_to_component_without_labels():
    bare = dict(ROW, baseline_label="", method_label="")
    p = assemble_prompt(GOOD_BODY, bare)
    assert "adding the gating mechanism" in p


def test_footer_falls_back_when_labels_degenerate():
    same = dict(ROW, baseline_label="ours", method_label="Ours")
    p = assemble_prompt(GOOD_BODY, same)
    assert "adding the gating mechanism" in p


def test_opener_rotation_is_deterministic_and_varies():
    from arl.compose import _OPENERS, opener_for
    assert opener_for("x-s000") == opener_for("x-s000")
    assert all(opener_for(f"p-s{i:03d}") in _OPENERS for i in range(20))
    assert len({opener_for(f"p-s{i:03d}") for i in range(40)}) > 1


def test_opener_reaches_the_composer(monkeypatch):
    import json as _json
    seen = {}

    def fake_chat(model, messages, **kw):
        seen["user"] = messages[1]["content"]
        return _json.dumps({"body": GOOD_BODY}), {}

    monkeypatch.setattr(compose, "chat", fake_chat)
    run_compose_sample(ROW)
    from arl.compose import opener_for
    assert opener_for(ROW["sample_id"]) in seen["user"]


def test_prose_response_accepted_as_body(monkeypatch):
    # Providers sometimes return the body as plain prose despite the JSON
    # schema; the fallback accepts it and the gate validates the content.
    def fake_chat(model, messages, **kw):
        return GOOD_BODY, {}          # no JSON envelope at all

    monkeypatch.setattr(compose, "chat", fake_chat)
    res, _ = run_compose_sample(ROW)
    assert res["ok"] is True
    assert res["body"] == GOOD_BODY
    assert res["prompt"].startswith(GOOD_BODY)
