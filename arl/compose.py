"""Stage 7 part B: compose training samples. LLM rewrite + fidelity gate.

The composer LLM rewrites stage-6's verbatim spans into a neutral,
first-person researcher prompt ("I've been brainstorming..."). Payload now
passes through next-token prediction, so every rewrite is checked
mechanically before it becomes a training sample:

  - every number in the body must already occur in the source material
    (spans, baseline value, setting/technique names) -- invented = rejected
  - the baseline value must be present; the method value must NOT be
  - every LaTeX command must occur in the spans -- equations are copied

Outcome references are handled by instruction, not by pattern: the composer's
system prompt forbids them, and stage 8 verifies the finished sample with a
cheap LLM judge (a regex cannot tell a stated goal from a stated result).
The composer is never shown the method value or stage 5's
judgement, so it cannot leak what it never saw. A failing rewrite is retried
with the failure reasons fed back, then recorded as a structured failure.
"""
from __future__ import annotations

import re
from zlib import crc32

import config
from arl.openrouter import chat
from arl.util import SchemaError, extract_json_block, validate_against_schema

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LATEX_RE = re.compile(r"\\[a-zA-Z]+")

COMPOSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["body"],
    "properties": {
        "body": {
            "type": "string",
            "description": "The first-person researcher message body.",
        },
    },
}

COMPOSE_SYSTEM = """\
You rewrite a researcher's source material into a single first-person
message: someone neutrally describing an idea they have been brainstorming
and the exact experiment they have set up, ending with their current
baseline result. A question is appended later; you write only the body.

The message must read as the researcher's own thinking, in a neutral tone:
an opener like the one you are given, then "my setup is...", "my baseline
gets...". Never present it as coming from a paper, and never name authors
or venues.

HARD RULES

1. Preserve every technical detail from the source passages: the mechanism,
   the formulation, every hyperparameter, dataset, model, resolution, epoch
   count. Losing a detail is as bad as inventing one.
2. Copy every number EXACTLY as it appears. Never introduce a number that
   is not in the source material.
3. Copy every equation and LaTeX expression EXACTLY as written.
4. State the baseline results (every metric's label and value you are
   given) as the current starting point.
5. NO outcome language: do not say or hint that the idea works, fails, or
   changes anything. You do not know the outcome. Stating the goal or
   motivation behind the design is fine; stating results is not.
6. Do not add expectations, hedges about likely results, or requests -- the
   question is appended separately.
"""


# A fixed opener would be a template fingerprint across the corpus (and a
# prominence cue in its own right); rotate deterministically by sample_id so
# the pipeline stays resumable -- recomposing the same row yields the same
# opener.
_OPENERS = (
    "brainstorming about",
    "working on",
    "ideating on",
    "developing",
    "spending cycles on",
    "mulling over",
    "sketching out",
    "iterating on",
)


def opener_for(sample_id: str) -> str:
    return _OPENERS[crc32(str(sample_id).encode("utf-8")) % len(_OPENERS)]


def _floats(text: str) -> set[float]:
    out = set()
    for m in _NUM_RE.findall(text or ""):
        try:
            out.add(float(m))
        except ValueError:
            pass
    return out


def _allowed_text(row: dict) -> str:
    return "\n".join((row.get("source_spans") or [])
                     + [str(row.get(k) or "") for k in
                        ("component", "setting", "baseline_label")]
                     + [str(m) for m in (row.get("metrics") or [])]
                     + [str(b) for b in (row.get("benchmarks") or [])])


def fidelity_check(body: str, row: dict) -> list[str]:
    """All the ways a rewrite can be unfaithful. Empty list = clean."""
    fails: list[str] = []
    allowed = _allowed_text(row)
    meas = row.get("measurements") or []
    baselines = {float(m["baseline_value"]) for m in meas}
    methods = {float(m["method_value"]) for m in meas}

    allowed_nums = _floats(allowed) | baselines
    body_nums = _floats(body)

    for v in sorted(body_nums - allowed_nums):
        fails.append(f"invented number {v:g}: it does not occur in the "
                     f"source material")
    for m in meas:
        b = float(m["baseline_value"])
        if b not in body_nums:
            fails.append(f"baseline value {b:g} ({m.get('metric_raw', '?')}) "
                         f"is missing from the body")
    for v in sorted(methods - baselines):
        if v in body_nums:
            fails.append(f"method value {v:g} leaked into the body -- it is "
                         f"part of the answer and must not appear")

    span_latex = set(_LATEX_RE.findall(allowed))
    for tok in sorted(set(_LATEX_RE.findall(body)) - span_latex):
        fails.append(f"latex command {tok} does not occur in the source "
                     f"spans -- equations must be copied exactly")

    # Outcome-reference screening is deliberately NOT done here with
    # patterns: the composer is instructed to exclude outcomes (rule 5 of
    # its system prompt), and stage 8 verifies semantically with an LLM
    # judge. A regex cannot tell a stated goal from a stated result.
    return fails


def question_footer(row: dict) -> str:
    pairs = []
    for m in row.get("measurements") or []:
        unit = (m.get("metric_unit") or "").strip()
        bench = (m.get("benchmark") or "").strip()
        label = (f"{bench} " if bench else "") + m.get("metric_raw", "?")
        pairs.append(label + (f" (in {unit})" if unit else ""))
    what = "; ".join(pairs) or "the reported results"
    # Orient the question by the row's own arm labels: "A vs B" components
    # are ambiguous about which arm the baseline is (a model should not have
    # to guess the dataset's naming convention), and "adding the X vs Y"
    # is not even grammatical. Fall back to the component when labels are
    # missing or degenerate.
    base_l = " ".join((row.get("baseline_label") or "").split())
    meth_l = " ".join((row.get("method_label") or "").split())
    if base_l and meth_l and base_l.lower() != meth_l.lower():
        change = (f"switching from {base_l} (my baseline above) "
                  f"to {meth_l}")
    else:
        change = f"adding the {row.get('component', 'technique')}"
    # One question, and it is the scored one. The direction ask that used to
    # sit in front of this was never graded; it was kept as scaffolding to make
    # the model commit to a mechanism first. For the pilot it goes: an unscored
    # question invites a verdict-shaped answer, and a model that has just
    # written "this clearly helps" is anchored when it then writes the numbers.
    return (f"\n\nMy question about this setup: after {change}, predict the "
            f"resulting value for each: {what}.")


def assemble_prompt(body: str, row: dict) -> str:
    return body.rstrip() + question_footer(row)


def compose_target(row: dict) -> dict:
    """The graded target: the measurement grid, nothing else.

    No direction, and since the prompt no longer asks for one either, target
    and question now agree: every slot the prompt asks about is a slot that
    gets scored, and nothing else is asked.
    """
    return {
        "measurements": [
            {"benchmark": m.get("benchmark", ""),
             "metric": m.get("metric_raw", ""),
             "baseline_value": m.get("baseline_value"),
             "method_value": m.get("method_value"),
             "unit": m.get("metric_unit", "")}
            for m in (row.get("measurements") or [])
        ],
    }


def run_compose_sample(row: dict, model: str | None = None,
                       audit_feedback: str = "") -> tuple[dict, dict]:
    """Compose one training sample from one labelled row.

    The composer sees spans + setting + baseline. It is never shown
    method_value or any stage-5 judgement. `audit_feedback`
    carries a previous version's stage-8 audit failure reasons so the
    recomposition can avoid them (leak phrasing, fidelity slips).
    """
    spans = row.get("source_spans") or []
    baseline_lines = "\n".join(
        f"  {(m.get('benchmark') or '').strip() or '-'} / "
        f"{m.get('metric_raw', '?')}: "
        f"{row.get('baseline_label', 'baseline')} = "
        f"{m.get('baseline_value')}{(m.get('metric_unit') or '').strip()}"
        for m in (row.get("measurements") or []))
    base_user = (
        f"TECHNIQUE: {row.get('component', '')}\n"
        f"SETTING: {row.get('setting', '')}\n"
        f"OPENER: start the message in the spirit of \"I've been "
        f"{opener_for(row.get('sample_id', ''))} ...\" (rephrase naturally, "
        f"keep first person)\n"
        f"BASELINE RESULTS (state each in the body):\n{baseline_lines}\n\n"
        f"SOURCE PASSAGES (verbatim from the researcher's notes):\n"
        + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(spans, 1))
    )
    if audit_feedback.strip():
        base_user += (
            "\n\nA PREVIOUS VERSION OF THIS MESSAGE FAILED AN INDEPENDENT "
            "AUDIT for these reasons -- write a version that avoids them:\n"
            + audit_feedback.strip())

    feedback = ""
    failures: list[str] = []
    usage: dict = {}
    for attempt in range(1, config.COMPOSE_RETRIES + 2):
        text, usage = chat(
            model or config.MODEL_COMPOSE,
            [{"role": "system", "content": COMPOSE_SYSTEM},
             {"role": "user", "content": base_user + feedback}],
            schema=COMPOSE_SCHEMA, schema_name="compose",
            max_tokens=config.COMPOSE_MAX_TOKENS,
            reasoning=config.reasoning_param(),
            traj=f"07_compose_{row.get('sample_id', '')}_try{attempt}",
        )
        try:
            res = extract_json_block(text)
            validate_against_schema(res, COMPOSE_SCHEMA)
            body = (res.get("body") or "").strip()
        except (ValueError, SchemaError):
            # Some providers intermittently return the body as plain prose
            # despite the enforced JSON schema. The envelope is packaging,
            # not protection: the content is validated by the mechanical
            # fidelity gate below and by stage 8's audit either way, so
            # accept the prose as the body rather than erroring the sample.
            body = text.strip()
        failures = fidelity_check(body, row)
        if not failures:
            return {
                "ok": True,
                "prompt": assemble_prompt(body, row),
                "target": compose_target(row),
                "body": body,
                "attempts": attempt,
                "failures": [],
            }, usage
        feedback = (
            "\n\nYOUR PREVIOUS ATTEMPT FAILED THESE MECHANICAL CHECKS:\n- "
            + "\n- ".join(failures)
            + "\n\nRewrite the body and fix every failure. Previous body:\n"
            + body)

    return {"ok": False, "failures": failures,
            "attempts": config.COMPOSE_RETRIES + 1,
            "target": compose_target(row)}, usage
