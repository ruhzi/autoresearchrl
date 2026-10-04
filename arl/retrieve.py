"""Stage 6: pointer-slicing retrieval agent. Lossless by construction.

Stage 5 says a comparison exists; this agent is handed ONE of those
descriptions and must locate it in the paper. It never types paper text:
it explores with grep/read and submits POINTERS (anchor strings). Python
resolves every pointer against the PaperStore and slices the exact source
bytes -- prose spans, the evidence row, and the two numbers, which must occur
literally inside that sliced row. A hallucinated span or number cannot
survive this: it fails to resolve and goes back to the agent as an error.

`submit_not_found` stays a first-class answer and keeps its old role as the
quality meter on stage 5.
"""
from __future__ import annotations

import re

import config
from arl.openrouter import ToolError, chat_tools
from arl.paperstore import AnchorError, PaperStore

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

_COMPARISON_KINDS = {"ablation", "baseline_vs_method", "variant_sweep",
                     "external_baseline", "other"}
_CONFIDENCES = {"high", "medium", "low"}


RETRIEVE_SYSTEM = """\
You are given ONE comparison that has already been identified in a
machine-learning paper: a technique, the setting it runs in, and the metrics
it is reported on. Locate it in the paper and submit pointers to it. You are
not searching for comparisons -- exactly one is described; find that one.

THE PAPER

The paper is stored outside your context as sections s0..sN and tables
t0..tM (the outline lists every id with kind, title, and size). Explore with:

- grep(query): literal search, case-insensitive, whitespace-tolerant.
  The fastest way in: grep the metric name, the technique, or a row label.
- read(target, offset): one page of raw LaTeX from a section or table.

POINTERS, NOT TEXT

You never write paper content. You bracket it: a pointer is
{target, start_anchor, end_anchor}, where the anchors are SHORT strings
(roughly 3-8 words) copied EXACTLY from what grep/read showed you. Each
anchor must occur exactly once in that target. The system slices the text
between your anchors straight from the source -- so anchor precision is all
that matters, and any anchor that does not match comes back to you as an
error to fix.

WHAT TO SUBMIT: submit_found

- spans: pointers to the passages a reader needs to understand this
  technique and the conditions it was measured under. Look for, and bracket
  when the paper provides it: the motivation; the mechanism or formulation;
  the experimental conditions for THIS comparison (backbone, dataset,
  epochs, protocol -- often stated once at the top of a section); what the
  baseline variant actually is. Quote what exists, skip what does not --
  two real spans beat four stretched ones. Bracket only passages relevant
  to THIS sample's technique.
- evidence: an ARRAY of pointers, each bracketing a table row pair holding
  values (bracket from the first row's label to the end of the second row
  when adjacent; both rows inside the slice). Rows usually carry all the
  metrics as columns; different benchmarks often sit in different rows or
  different tables -- add one evidence pointer per row pair you read from.
- measurements: one entry PER (BENCHMARK, METRIC) pair this comparison
  reports. For each: the benchmark name, the metric name, and
  baseline_cell / method_cell -- the two values EXACTLY as they appear in
  the rows (e.g. "71.2"). Each cell must occur inside one of the evidence
  slices and contain exactly one number. baseline = the variant WITHOUT
  the technique, method = WITH it. For an ablation table where the
  technique is X: baseline is "w/o X", method is "full model". Include
  every pair you can extract; a value reported only as a range or delta
  is skipped, not guessed.
- baseline_label / method_label: how the paper labels those two rows.
- per measurement: the benchmark, the metric named as the paper names it,
  the unit, and the two cells. Do not judge whether a move is good.
- isolates_single_technique: false if the two rows differ in more than one
  respect. Still submit, but flag it.
- confidence: low if you had to choose between plausible row pairings.

WHEN TO GIVE UP: submit_not_found

Returning not_found with a reason is a correct and useful answer -- the
described comparison is not in this paper, or the values exist only as a
range, a plus/minus, or a relative delta (not extractable), or the
description is too vague to match a specific pair of rows. Do not force a
match: a wrong number becomes a training label nothing downstream can catch.

Do not judge whether the change is good or bad; that is computed
arithmetically downstream. Respond only with tool calls.
"""


def _pointer_schema() -> dict:
    return {"type": "object",
            "properties": {"target": {"type": "string"},
                           "start_anchor": {"type": "string"},
                           "end_anchor": {"type": "string"}},
            "required": ["target", "start_anchor", "end_anchor"]}


def _retrieve_tools() -> list[dict]:
    def tool(name, desc, props=None, req=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {},
                           "required": req or []}}}

    return [
        tool("grep", "Literal search across the whole paper.",
             {"query": {"type": "string"}}, ["query"]),
        tool("read", "One page of raw LaTeX from a section or table.",
             {"target": {"type": "string"},
              "offset": {"type": "integer"}}, ["target"]),
        tool("submit_found", "Submit pointers to the located comparison.", {
            "spans": {"type": "array", "items": _pointer_schema(),
                      "description": "passages needed to understand the "
                                     "technique and its conditions"},
            "evidence": {
                "type": "array",
                "items": _pointer_schema(),
                "description": "Pointers to the table row pairs holding the "
                               "values -- one per table/row-block (multiple "
                               "benchmarks often sit in different rows or "
                               "tables).",
            },
            "measurements": {
                "type": "array",
                "description": "One entry PER (benchmark, metric) pair the "
                               "paper reports for this comparison (each cell "
                               "pair must occur inside an evidence slice).",
                "items": {"type": "object", "properties": {
                    "benchmark": {"type": "string"},
                    "metric": {"type": "string"},
                    "baseline_cell": {"type": "string"},
                    "method_cell": {"type": "string"},
                    "metric_unit": {"type": "string"},
                }, "required": ["benchmark", "metric", "baseline_cell",
                                "method_cell", "metric_unit"]},
            },
            "baseline_label": {"type": "string"},
            "method_label": {"type": "string"},
            "comparison_kind": {"type": "string",
                                "enum": sorted(_COMPARISON_KINDS)},
            "isolates_single_technique": {"type": "boolean"},
            "table_ref": {"type": "string"},
            "confidence": {"type": "string", "enum": sorted(_CONFIDENCES)},
        }, ["spans", "evidence", "measurements", "baseline_label",
            "method_label", "comparison_kind",
            "isolates_single_technique", "table_ref", "confidence"]),
        tool("submit_not_found",
             "The described comparison cannot be located or its values are "
             "not extractable. Give the reason.",
             {"reason": {"type": "string"}}, ["reason"]),
    ]


def _norm(s: str) -> str:
    return " ".join((s or "").split())


class RetrieveEnv:
    """Dispatch backend for one sample's retrieval run.

    Holds the verdict in `result` after a successful terminal call. All
    payload in the accepted extraction is sliced from the store; the only
    model-typed fields are labels/metadata that stage 7 treats as such.
    """

    def __init__(self, store: PaperStore):
        self.store = store
        self.result: dict | None = None

    def dispatch(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        return handler(args)

    # -------------------------------------------------------------- explore

    def _tool_grep(self, args: dict) -> str:
        hits = self.store.grep(str(args.get("query", "")))
        if not hits:
            return "no matches"
        return "\n".join(
            f"{h['target']}"
            f"{' (CAPTION -- do not anchor here)' if h.get('in_caption') else ''}"
            f" @{h['offset']}: ...{h['context']}..."
            for h in hits)

    def _tool_read(self, args: dict) -> str:
        try:
            return self.store.peek(args.get("target", ""),
                                   int(args.get("offset") or 0))
        except (KeyError, ValueError) as e:
            raise ToolError(str(e)) from None

    # -------------------------------------------------------------- helpers

    def _slice(self, pointer, what: str) -> str:
        if not isinstance(pointer, dict):
            raise ToolError(f"{what} must be a pointer object "
                            f"{{target, start_anchor, end_anchor}}")
        try:
            return self.store.slice(pointer)
        except (AnchorError, KeyError) as e:
            raise ToolError(f"{what}: {e}") from None

    def _cell_value(self, cell: str, evidence: str, what: str) -> float:
        cell = str(cell or "")
        if not cell.strip():
            raise ToolError(f"{what} is empty")
        if _norm(cell) not in _norm(evidence):
            raise ToolError(
                f"{what} {cell!r} does not occur inside the sliced evidence "
                f"row. The cell text must be copied exactly from the row "
                f"your evidence pointer brackets.")
        nums = _NUM.findall(cell)
        if len(nums) != 1:
            raise ToolError(
                f"{what} {cell!r} contains {len(nums)} numbers; it must "
                f"contain exactly one (the value itself).")
        value = float(nums[0])
        # Substring containment alone would let "1.2" pass against "71.25".
        # The parsed value must equal one of the numbers the evidence
        # actually contains, as full tokens.
        evidence_nums = set()
        for n in _NUM.findall(evidence):
            try:
                evidence_nums.add(float(n))
            except ValueError:
                pass
        if value not in evidence_nums:
            raise ToolError(
                f"{what} {cell!r} parses to {value:g}, which is not one of "
                f"the numbers in the evidence slice -- copy the full cell "
                f"value exactly.")
        return value

    # ------------------------------------------------------------- terminal

    def _tool_submit_found(self, args: dict) -> str:
        spans = args.get("spans")
        if not isinstance(spans, list) or not spans:
            raise ToolError("`spans` must be a non-empty array of pointers "
                            "(the passages the sample is written from)")
        sliced_spans = [self._slice(p, f"spans[{i}]")
                        for i, p in enumerate(spans)]
        ev_pointers = args.get("evidence")
        if not isinstance(ev_pointers, list) or not ev_pointers:
            raise ToolError("`evidence` must be a non-empty array of "
                            "pointers to the table row pairs")
        evidences = [self._slice(p, f"evidence[{i}]")
                     for i, p in enumerate(ev_pointers)]
        combined_evidence = "\n".join(evidences)

        raw_meas = args.get("measurements")
        if not isinstance(raw_meas, list) or not raw_meas:
            raise ToolError("`measurements` must be a non-empty array: one "
                            "entry per (benchmark, metric) pair")
        measurements = []
        for i, mm in enumerate(raw_meas):
            if not isinstance(mm, dict):
                raise ToolError(f"measurements[{i}] must be an object")
            name = str(mm.get("metric") or "").strip()
            bench = str(mm.get("benchmark") or "").strip()
            if not name or not bench:
                raise ToolError(f"measurements[{i}] needs both a benchmark "
                                f"and a metric name")
            tag = f"({bench}/{name})"
            measurements.append({
                "benchmark": bench,
                "metric": name,
                "baseline_value": self._cell_value(
                    mm.get("baseline_cell"), combined_evidence,
                    f"measurements[{i}].baseline_cell {tag}"),
                "method_value": self._cell_value(
                    mm.get("method_cell"), combined_evidence,
                    f"measurements[{i}].method_cell {tag}"),
                "baseline_cell": str(mm.get("baseline_cell")),
                "method_cell": str(mm.get("method_cell")),
                "metric_unit": str(mm.get("metric_unit") or ""),
            })

        kind = str(args.get("comparison_kind") or "")
        if kind not in _COMPARISON_KINDS:
            raise ToolError(f"comparison_kind must be one of "
                            f"{sorted(_COMPARISON_KINDS)}")
        confidence = str(args.get("confidence") or "")
        if confidence not in _CONFIDENCES:
            raise ToolError(f"confidence must be one of "
                            f"{sorted(_CONFIDENCES)}")

        self.result = {
            "found": True,
            "not_found_reason": "",
            "extraction": {
                # ---- payload: sliced from source, never model-typed
                # (measurement values are parsed from cells verified to occur
                # literally inside the sliced evidence row)
                "source_spans": sliced_spans,
                "evidence": evidences,
                "measurements": measurements,
                # ---- audit trail
                "span_pointers": spans,
                "evidence_pointer": args.get("evidence"),
                # ---- model-typed metadata (labels, not payload)
                "baseline_label": str(args.get("baseline_label") or ""),
                "method_label": str(args.get("method_label") or ""),
                "comparison_kind": kind,
                "isolates_single_technique":
                    bool(args.get("isolates_single_technique")),
                "table_ref": str(args.get("table_ref") or ""),
                "confidence": confidence,
            },
        }
        return (f"accepted: {len(sliced_spans)} spans, "
                f"{len(evidences)} evidence slices, "
                f"{len(measurements)} (benchmark, metric) measurements")

    def _tool_submit_not_found(self, args: dict) -> str:
        reason = str(args.get("reason") or "").strip()
        if not reason:
            raise ToolError("submit_not_found requires a non-empty reason")
        self.result = {"found": False, "extraction": None,
                       "not_found_reason": reason}
        return "recorded as not found"


def run_retrieve_sample(paper: dict, parsed: dict, sample: dict,
                        model: str | None = None) -> tuple[dict, dict]:
    """Retrieve one stage-5 sample's verbatim material. One agent run.

    Stage 5 hands over identity only. This stage slices numbers out of the
    source; it does not judge them.
    """
    store = PaperStore(parsed, peek_chars=config.RETRIEVE_READ_CHARS)
    env = RetrieveEnv(store)

    user = (
        f"THE COMPARISON YOU MUST FIND\n"
        f"  technique   : {sample.get('technique', '')}\n"
        f"  description : {sample.get('description', '')}\n"
        f"  setting     : {sample.get('setting', '')}\n"
        f"  benchmarks  : {', '.join(sample.get('benchmarks') or [])}\n"
        f"  metrics     : {', '.join(sample.get('metrics') or [])}\n\n"
        f"This is one of {sample.get('_n_in_paper', 'several')} comparisons "
        f"identified in the paper. Find this one.\n\n"
        f"PAPER: {paper['title']}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"OUTLINE:\n{store.outline()}\n"
    )
    chat_tools(
        model or config.MODEL_RETRIEVE, RETRIEVE_SYSTEM, user,
        _retrieve_tools(), env.dispatch,
        terminal_tools={"submit_found", "submit_not_found"},
        max_turns=config.RETRIEVE_MAX_TURNS,
        max_tokens=config.RETRIEVE_MAX_TOKENS,
        reasoning=config.reasoning_param(),
        traj=f"06_retrieve_{sample.get('sample_id', '')}",
    )
    assert env.result is not None    # chat_tools only returns on success
    return env.result, {}
