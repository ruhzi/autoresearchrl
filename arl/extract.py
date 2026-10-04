"""LLM stages: the cheap gate and the strong extractor.

Division of labour matters here:

  screen   -- cheap model, high recall. Three booleans, nothing else: is this AI
              research, does it introduce something, are results attached.
              `eligible` is their AND, computed in Python.

  extract  -- strong model. Pulls (setting, metric, baseline_value, method_value)
              tuples off the tables. It does NOT decide whether a change is an
              improvement; magnitude is computed arithmetically
              in labels.py from these numbers plus the metric registry. That
              keeps every label checkable against the paper by hand.

The extractor sees the WHOLE paper including results. The sanitised context in
sanitize.py is what the *evaluated* model sees; ground truth is built with full
information.
"""
from __future__ import annotations

import config
from arl.openrouter import chat_json
from arl.util import validate_against_schema

# --------------------------------------------------------------------- schemas

# Three booleans and a reason. Nothing derived, nothing decorative.
#
# An earlier version asked for nine fields -- a 6-way enum that had to stay
# consistent with a free-text field, two metadata enums nothing consumed, and an
# `eligible` flag the model had to compute as an AND of three booleans with an
# exception clause. It got the comprehension right every time and the bookkeeping
# wrong eight times out of twenty. Every extra field is one more simultaneous
# constraint; `eligible` is arithmetic and belongs in Python.
SCREEN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_ai_research", "introduces_something", "has_results",
                 "reason"],
    "properties": {
        "is_ai_research": {"type": "boolean"},
        "introduces_something": {"type": "boolean"},
        "has_results": {"type": "boolean"},
        "reason": {
            "type": "string",
            "description": "One or two sentences. If any answer is false, say "
                           "which one and why.",
        },
    },
}

# --------------------------------------------------------------------- prompts

SCREEN_SYSTEM = """\
You screen papers for a dataset about AI research techniques. You are given a
paper's title, arXiv categories, abstract, section outline, and table captions.
Answer three questions.

1. is_ai_research

   Is this AI research? Judge the paper's contribution, not its venue -- a paper
   in an astronomy or medical venue counts if what it contributes is a model or a
   training technique. Not AI research: classical algorithm design and complexity
   theory, pure mathematics, hardware or systems work where nothing is learned.

2. introduces_something

   Does the paper introduce or meaningfully investigate something of its own
   that can be evaluated through a controlled comparison -- such as a method,
   module, pipeline, framework, architecture, training scheme, policy,
   evaluation protocol, component, modification, or experimental condition?
   This includes novel contributions as well as meaningful investigations or
   modifications of existing techniques, provided they can be compared against
   a baseline or alternative. Not: a literature survey or position paper that
   only summarises other people's work.

3. has_results

   Are results attached, in some form? The authors' own numbers, in a table, a
   figure, or the text. Not: only numbers cited from other papers, and not purely
   qualitative examples with nothing measured.

Then `reason`: one or two sentences. If any answer is false, say which and why.


"""




def _fmt_tables(tables: list[dict], limit: int) -> str:
    out, used = [], 0
    for i, t in enumerate(tables):
        block = (
            f"--- table {i} (env={t['env']}, label={t.get('label', '')!r}) ---\n"
            f"CAPTION: {t.get('caption', '')}\n"
            f"{t.get('body', '')}\n"
        )
        if used + len(block) > limit:
            out.append(f"[... {len(tables) - i} further tables omitted for length]")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


SCREEN_CRITERIA = ("is_ai_research", "introduces_something",
                   "has_results")


def run_screen(paper: dict, parsed: dict) -> tuple[dict, dict]:
    """Three-rubric eligibility screen. Cheap model, runs on every paper.

    The response is validated against SCREEN_SCHEMA before it is returned, so a
    malformed reply raises instead of becoming a verdict, and `eligible` is
    computed here rather than asked of the model.
    """
    outline = "\n".join(
        f"- {s['title'] or s['kind']}"
        for s in parsed.get("sections", [])
        if s.get("title") or s["kind"] == "frontmatter"
    ) or "(no sections detected)"

    captions = "\n".join(
        f"- {t.get('caption', '')[:300]}"
        for t in parsed.get("tables", [])
        if t.get("caption")
    ) or "(no captioned tables found)"

    user = (
        f"TITLE: {paper['title']}\n\n"
        f"ARXIV CATEGORIES: {', '.join(paper.get('cats') or [])}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"SECTION OUTLINE:\n{outline}\n\n"
        f"TABLE CAPTIONS ({parsed.get('n_tables', 0)} table environments in "
        f"the source):\n{captions}\n"
    )
    res, usage = chat_json(config.MODEL_GATE, SCREEN_SYSTEM, user,
                           SCREEN_SCHEMA, schema_name="screen",
                           max_tokens=2048,
                           traj=f"04_screen_{paper.get('base_id', '')}")
    validate_against_schema(res, SCREEN_SCHEMA)
    res["eligible"] = all(bool(res[k]) for k in SCREEN_CRITERIA)
    res["failed_criteria"] = [k for k in SCREEN_CRITERIA if not res[k]]
    return res, usage


# ---- stage 6: one call per stage-5 sample -----------------------------------
#
# Stage 5 says a comparison exists and which way it went. Stage 6 is handed ONE
# of those and asked to locate it and read the two numbers off the table.
#
# One sample per call, not a batch: hand a model forty rows to fill and it
# returns forty rows whether or not the numbers exist. Asked about a single
# comparison, "I could not find this" is a natural answer rather than a hole in
# a table it has been told to complete.
#
# Stage 6 is NOT given stage 5's location, and is deliberately not told which
# direction stage 5 judged. It re-locates the comparison itself, so a stage-5 row
# that was vague or invented shows up as `not_found` instead of being silently
# trusted -- and its numbers give an arithmetic direction that can be checked
# against stage 5's judged one.
_EXTRACTION = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_spans", "comparison_kind", "metric_unit",
        "baseline_label", "baseline_value", "method_label", "method_value",
        "isolates_single_technique", "table_ref",
        "evidence", "confidence",
    ],
    "properties": {
        # ---- SOURCE MATERIAL: the text stage 7 writes the sample from.
        #
        # ONE array, not a field per category. An earlier version required four
        # separate spans -- intuition / methodology / setup / baseline -- which
        # made a paper that simply never states its motivation look like a
        # defective extraction, and pressured the model to stretch something into
        # the empty slot. The categories are guidance in the prompt now; the
        # schema takes however many quotes the paper actually supports.
        "source_spans": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Passages quoted VERBATIM from the paper that a reader "
                           "would need in order to understand this technique and "
                           "the conditions it was measured under. Quote what the "
                           "paper actually provides -- as many spans as that "
                           "takes, and nothing invented for what it does not say. "
                           "Each span is checked literally against the paper.",
        },
        "comparison_kind": {
            "type": "string",
            "enum": ["ablation", "baseline_vs_method", "variant_sweep",
                     "external_baseline", "other"],
        },
        "metric_unit": {
            "type": "string",
            "description": "e.g. '%', 'points', 'GFLOPs', 'ms', or '' if unitless.",
        },
        "baseline_label": {
            "type": "string",
            "description": "The row WITHOUT the technique, labelled as the paper "
                           "labels it.",
        },
        "baseline_value": {"type": "number"},
        "method_label": {
            "type": "string",
            "description": "The row WITH the technique, labelled as the paper "
                           "labels it.",
        },
        "method_value": {"type": "number"},
        "isolates_single_technique": {
            "type": "boolean",
            "description": "False if the two rows differ in more than one respect "
                           "(the technique AND a different epoch budget, say), "
                           "which makes the comparison uncontrolled.",
        },
        "table_ref": {
            "type": "string",
            "description": "Table label or a fragment of its caption.",
        },
        "evidence": {
            "type": "string",
            "description": "The table row text you read the numbers from, copied "
                           "EXACTLY from the input. This is checked literally "
                           "against the source, so do not paraphrase or reformat.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}

# No `found` flag: `extraction` is null when the comparison could not be located,
# and found-ness is derived from that in Python. One field fewer to keep
# consistent with another.
EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["extraction", "not_found_reason"],
    "properties": {
        "extraction": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": _EXTRACTION["required"],
            "properties": _EXTRACTION["properties"],
            "description": "The extracted comparison, or null if you could not "
                           "find it in this paper.",
        },
        "not_found_reason": {
            "type": "string",
            "description": "Why you could not locate it. Empty string when "
                           "extraction is present.",
        },
    },
}

EXTRACT_SYSTEM = """\
You are given one machine-learning paper and ONE specific comparison that has
already been identified in it. Your job is to locate that comparison and read
its two numbers off the table.

You are not searching for comparisons. Exactly one is described to you, by the
technique it varies, the setting it runs in, and the metric it reports. Find that
one.

=== WHAT TO RETURN ===

If you find it, fill `extraction` and leave `not_found_reason` empty:

- baseline_value  the number for the variant WITHOUT the technique
- method_value    the number for the variant WITH the technique
- baseline_label / method_label  how the paper labels those two rows

For an ablation table with rows "full model" and "w/o X", where the technique is
X: baseline is "w/o X", method is "full model".

If you cannot find it, set `extraction` to null and say why in
`not_found_reason` -- the described comparison is not in this paper, or the
table reports it in a form with no absolute values, or the description is too
vague to match a specific pair of rows.

**Returning null is a correct and useful answer.** Do not guess numbers to fill
the slot. A wrong number is far worse than a null, because it becomes a training
label that nothing downstream can detect as wrong.

=== READING THE NUMBERS ===

1. Copy values VERBATIM from the table. Never estimate, infer, round, or
   reconstruct a value you cannot see.

2. If the cell holds a range, a plus/minus, or a relative delta rather than an
   absolute value, return null -- that is not extractable.

3. `evidence` must be the table row text copied EXACTLY from the input. It is
   checked literally against the source you were given, so do not paraphrase,
   reformat, or tidy the LaTeX.

4. Do NOT judge whether the change is good or bad. Report the two values and
   Do not judge whether a move is an improvement -- that depends on the
   paper's goal for the metric and is decided downstream, not here.

5. Set `isolates_single_technique` false if the two rows differ in more than one
   respect. Still extract them, but flag it.

=== SOURCE SPANS: THE TEXT THE SAMPLE GETS WRITTEN FROM ===

Besides the numbers, quote the passages a reader would need to understand this
technique and the conditions it was measured under. A later stage rewrites these
into a researcher describing the idea in their own voice, so it needs the
author's actual words -- summarising here loses detail that stage cannot recover.

Look for, and quote when the paper provides it:

  - the motivation: what problem or insight makes this technique plausible
  - the mechanism: how it works, what it changes, the formulation
  - the experimental conditions for THIS comparison: backbone, dataset,
    resolution, epochs, evaluation protocol, what was held fixed. Often stated
    once at the top of a section rather than in the table itself.
  - what the baseline variant actually is, e.g. that sigma_0 = 0 leaves the
    endpoint unchanged and recovers the standard method

That is a checklist of what to look for, NOT a required set. Plenty of papers
never state a motivation, or fold the setup into a caption. Quote what is there
and skip what is not: do not stretch something into a gap, and do not pad the
array to look complete. Two real spans are worth more than four where two were
invented.

Rules for every span:

- COPY EXACTLY from the input. Each span is checked literally against the paper,
  and one that does not occur verbatim is flagged. Do not paraphrase, summarise,
  or tidy the LaTeX. Never stitch non-contiguous sentences into one span -- use
  two spans instead.
- Keep equations as written, LaTeX and all. The formula itself is more useful
  than a prose gloss of it.
- Quote enough to stand alone. Several sentences per span is fine, and better
  than a clipped fragment.
- Quote only what is relevant to THIS sample's technique. In a paper with three
  components, a sample about the second should not quote the first's mechanism.

Set `confidence` to low if you had to choose between plausible row pairings.
"""


def run_extract_sample(paper: dict, parsed: dict, sample: dict,
                       model: str | None = None) -> tuple[dict, dict]:
    """Extract the numbers for ONE stage-5 sample. One API call per sample.

    The sample descriptor is what stage 5 identified: technique, setting, metric.
    Stage 5 hands over identity only -- technique, setting, metrics. There is
    no judgement to withhold; this stage reads cells.
    """
    method = "\n\n".join(
        f"## {s['title']}\n{s['body'][:10000]}"
        for s in parsed.get("sections", [])
        if s["kind"] in ("method", "intro")
    )[: config.MAX_CONTEXT_CHARS]

    results = "\n\n".join(
        f"## {s['title']}\n{s['body'][:10000]}"
        for s in parsed.get("sections", [])
        if s["kind"] in ("results", "ablation")
    )[: config.MAX_CONTEXT_CHARS]

    user = (
        f"THE COMPARISON YOU MUST FIND\n"
        f"  technique   : {sample.get('technique', '')}\n"
        f"  description : {sample.get('description', '')}\n"
        f"  setting     : {sample.get('setting', '')}\n"
        f"  metric      : {sample.get('metric_name', '')}\n\n"
        f"This is one of {sample.get('_n_in_paper', 'several')} comparisons "
        f"identified in the paper below. Find this one.\n\n"
        f"=== PAPER ===\n"
        f"TITLE: {paper['title']}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"=== METHOD / INTRO (LaTeX) ===\n{method}\n\n"
        f"=== RESULTS / ABLATION PROSE (LaTeX) ===\n{results}\n\n"
        f"=== TABLES (LaTeX source, {parsed.get('n_tables', 0)} found) ===\n"
        f"{_fmt_tables(parsed.get('tables', []), config.MAX_TABLE_CHARS)}\n"
    )
    res, usage = chat_json(
        model or config.MODEL_RETRIEVE, EXTRACT_SYSTEM, user, EXTRACT_SCHEMA,
        schema_name="extraction", max_tokens=8192,
    )
    validate_against_schema(res, EXTRACT_SCHEMA)

    ext = res.get("extraction")
    res["found"] = ext is not None            # derived, not asked for
    if not res["found"] and not (res.get("not_found_reason") or "").strip():
        res["not_found_reason"] = "(model gave no reason)"
    return res, usage
