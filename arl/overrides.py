"""Human corrections as pipeline INPUTS.

Stage 7 rebuilds data/samples.jsonl out of data/extracted/, and stage 8
rebuilds data/train_samples.jsonl out of data/composed/. Both rebuild from
scratch, so a correction written into either output file survives exactly
until the next `python run_all.py` -- which is how a full-source audit turned
into a chore that had to be repeated after every run.

Two files fix that, and they are the only hand-maintained inputs in the
pipeline:

  data/label_overrides.json   {sample_id: {field: value}}   applied in stage 7
  data/prompt_overrides.json  {sample_id: "prompt body"}    applied in stage 8

A label patch lands after normalize() and BEFORE the mechanical checks, so a
correction is verified like any other row -- an override that contradicts the
paper gets flagged needs_review rather than trusted. A prompt override
replaces the composer's BODY only; the question footer stays deterministic
Python, so restyling it still reaches every sample.

Nothing here invents data. Each entry records a decision a human already made
against the paper; the point is that the decision is now replayable.
"""
from __future__ import annotations

from pathlib import Path

from arl.util import read_json


def load(path: str | Path) -> dict:
    """An absent file means no corrections, never an error: the pipeline must
    run on a fresh checkout with no override files at all."""
    return read_json(Path(path), default={}) or {}


def apply_label(row: dict, patch: dict) -> dict:
    """Return `row` with `patch` applied. Lists (measurements) are replaced
    wholesale, not merged -- a partial measurement grid is not a grid.

    Keys starting with `_` are DIRECTIVES to stage 7, not label fields: they
    are carried onto the row for provenance but never counted as corrections.
    The touched field names are recorded as `overridden` so a corrected sample
    stays identifiable downstream.
    """
    if not patch:
        return dict(row)
    out = dict(row)
    out.update(patch)
    fields = sorted(k for k in patch if not k.startswith("_"))
    if fields:
        out["overridden"] = fields
    return out


def overruled(patch: dict) -> set[str]:
    """Checks whose verdict this correction outranks.

    A mechanical check sees only the spans stage 6 sliced; a human auditing the
    correction had the whole paper open. When they disagree the human can win,
    but only by saying so here and saying why -- the check still runs and its
    complaint is still recorded on the row. Silence in either direction is the
    thing to avoid: a trusted-by-default override hides bad corrections, and a
    check that deletes an audited row hides good ones.
    """
    return set((patch or {}).get("_overrule") or [])


def body_for(sample_id: str, overrides: dict, composed_body: str) -> str:
    """The audited body if there is one, else the composer's.

    An empty override is ignored rather than honoured: blanking a prompt is
    never the intent, and a missing body would compose an empty question.
    """
    return overrides.get(sample_id) or composed_body
