"""Stage 8: verify each finished training sample with a cheap LLM judge.

Two questions about the composed prompt, judged semantically (a regex cannot
tell a stated goal from a stated result -- this is why the check is a model,
not a pattern):

  1. outcome_reference -- does the prompt state, imply, or hint at how the
     experiment turned out? Any such reference makes the sample worthless
     (the answer would be in the question).
  2. faithful -- are the numbers, the experimental setup, and the technical
     details in the prompt exactly what the source passages say? Changed
     epochs, a dropped detail, or a reworded equation is a fidelity failure.

One mechanical check rides along in Python because it is exact and free:
the target's method_value must not appear in the prompt text.

The judge is never shown the method value, so its verdict
cannot leak them either. A sample passes only if the judge answers clean on
both questions AND the mechanical check passes.
"""
from __future__ import annotations

import re

import config
from arl.openrouter import chat_json
from arl.util import validate_against_schema

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["outcome_reference", "outcome_evidence", "faithful",
                 "fidelity_issues"],
    "properties": {
        "outcome_reference": {
            "type": "boolean",
            "description": "True if the message states, implies, or hints "
                           "at how the experiment turned out.",
        },
        "outcome_evidence": {
            "type": "string",
            "description": "The phrase(s) that reveal an outcome. Empty if "
                           "none.",
        },
        "faithful": {
            "type": "boolean",
            "description": "True if every number, setup detail, and "
                           "equation in the message matches the source "
                           "passages exactly.",
        },
        "fidelity_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each mismatch: what the message says vs what "
                           "the source says. Empty if faithful.",
        },
    },
}

VERIFY_SYSTEM = """\
You audit a training sample for a benchmark. The sample is a researcher
describing an idea and an experimental setup, ending with two questions
about what will happen. You are given the researcher's message and the
original source passages it was written from. Answer two things.

1. outcome_reference

   Does the message anywhere state, imply, or hint at the RESULT of the
   experiment -- that the idea works, fails, improves things, or how the
   numbers came out? The baseline result is allowed (it is the given
   starting point). The questions at the end are allowed. Statements of
   motivation or intent ("the goal is to reduce memory") are allowed.
   What is NOT allowed is any signal about what actually happened when the
   idea was tried. Be strict: a hedge like "which should help convergence"
   or "encouragingly" counts as a reference.

2. faithful

   Compare the message against the source passages. Every number
   (hyperparameters, dataset sizes, the baseline value), every equation,
   and every setup detail in the message must match the source exactly.
   Report each mismatch in fidelity_issues: dropped details, changed
   values, reworded formulas. Extra framing language is fine; changed
   facts are not.

Judge only what is in front of you. Do not guess the experiment's outcome.
"""


def method_value_leaks(prompt_text: str, target: dict) -> list[str]:
    """Mechanical, exact, free: the answer must not be in the question.

    Returns a reason per method value (that is not also a baseline) found
    verbatim in the prompt; empty = clean. Shared by the stage-8 audit and
    the final dataset build (which reassembles the footer and must re-check
    the exact text it ships).
    """
    meas = (target or {}).get("measurements") or []
    prompt_nums = set()
    for m in _NUM_RE.findall(prompt_text or ""):
        try:
            prompt_nums.add(float(m))
        except ValueError:
            pass
    baselines = {mm.get("baseline_value") for mm in meas}
    reasons = []
    for mm in meas:
        method = mm.get("method_value")
        if method is None or method in baselines:
            continue
        if float(method) in prompt_nums:
            reasons.append(f"method value {method} ({mm.get('metric', '?')}) "
                           f"appears in the prompt (mechanical check)")
    return reasons


def run_verify_sample(composed: dict, source_spans: list[str],
                      model: str | None = None) -> tuple[dict, dict]:
    """Verify one composed training sample. One cheap call.

    `composed` is a stage-7 output file (ok=True); `source_spans` are the
    verbatim spans the prompt was written from. The judge never sees the
    direction or method_value.
    """
    prompt_text = composed.get("prompt", "")
    user = (
        "RESEARCHER MESSAGE (the training sample):\n"
        "-----\n"
        f"{prompt_text}\n"
        "-----\n\n"
        "SOURCE PASSAGES it was written from (verbatim):\n"
        + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(source_spans, 1))
    )
    res, usage = chat_json(
        model or config.MODEL_SAMPLE_VERIFY, VERIFY_SYSTEM, user,
        VERIFY_SCHEMA, schema_name="verify",
        max_tokens=config.VERIFY_MAX_TOKENS,
        reasoning=config.reasoning_param(),
        traj=f"08_verify_{composed.get('sample_id', '')}",
    )
    validate_against_schema(res, VERIFY_SCHEMA)

    reasons: list[str] = []
    if res.get("outcome_reference"):
        reasons.append("outcome reference: "
                       + (res.get("outcome_evidence") or "(unspecified)"))
    if not res.get("faithful"):
        reasons.extend(res.get("fidelity_issues")
                       or ["(unspecified fidelity issue)"])

    # Mechanical, exact, free: the answer must not be in the question.
    reasons.extend(method_value_leaks(prompt_text,
                                      composed.get("target") or {}))

    return {
        "passed": not reasons,
        "reasons": reasons,
        "judge": res,
    }, usage
