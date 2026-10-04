"""Stage 5: enumerate candidate samples in a paper. Comparisons, no numbers.

The scoping pass. For each eligible paper it answers:

    how many samples can this paper yield, what are they, and what does each
    one measure?

No numeric extraction and no verdict. Values are read off the tables later, in
stage 6. The point of this stage is to learn whether the corpus yields enough
comparisons to be worth extracting -- which needs neither a number nor a
judgement.

Why there is no effect label here
---------------------------------
There used to be one, and it was the least reliable field the pipeline
produced. Whether a comparison helped is a function of the stage-6 values and
the paper's own polarity for each metric; asking a model that has not yet seen
the values to guess it meant guessing. A full-source audit found the numbers
essentially always right and that label wrong often enough to be the dominant
error source, so the label is gone. If a direction is wanted it is derived
post-hoc from the extracted values plus polarity read from the paper's own
captions -- once, mechanically, with a stated materiality threshold, rather
than per-sample by a model working blind.
"""
from __future__ import annotations

import re
import threading

import config
from arl.metrics import resolve
from arl.openrouter import ToolError, chat_json, chat_tools
from arl.paperstore import PaperStore
from arl.util import read_json, validate_against_schema, write_json

# ---------------------------------------------------------------------- schema

_SAMPLE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["technique", "description", "setting", "benchmarks",
                 "metrics"],
    "properties": {
        # ---- identity: what makes this a distinct sample.
        # A sample is ONE methodological comparison: (technique, recipe
        # setting). Where it is EVALUATED -- benchmarks AND metrics -- are
        # properties of the sample, never what makes it distinct. One recipe
        # scored on 4 benchmarks x 3 metrics is ONE sample.
        "technique": {
            "type": "string",
            "description": "Short name of the ONE technique this sample "
                           "measures, e.g. 'rotary position embedding', "
                           "'KV-cache sharing', 'retrieval instead of gold "
                           "context'.",
        },
        "description": {
            "type": "string",
            "description": "One or two sentences: what the technique does, and "
                           "what it is being compared against. General, not "
                           "verbatim -- but specific enough that a reader who has "
                           "the paper open could find the exact rows from "
                           "this alone. Name the two variants as the paper labels "
                           "them if you can see them.",
        },
        "setting": {
            "type": "string",
            "description": "The METHODOLOGICAL context the recipe runs in: "
                           "base model, scale, training budget/config. NOT "
                           "the evaluation datasets -- those go in "
                           "`benchmarks`.",
        },
        "benchmarks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "EVERY dataset/benchmark/task this comparison is "
                           "evaluated on, named as the paper names them.",
        },
        "metrics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "EVERY metric the paper reports for this "
                           "comparison, named verbatim as the paper names "
                           "them.",
        },
        # No effect/direction label. A sample is a comparison, not a verdict:
        # the verdict is a function of the stage-6 numbers plus the paper's own
        # metric polarity, so asking a model that has not yet seen the numbers
        # to guess it produced a label that was wrong far more often than the
        # numbers were. Derived post-hoc if it is wanted at all.
    },
}

# Just the list, plus a note. The count is len(samples) and the cross-metric
# tradeoffs are derivable from the samples themselves -- both are computed in
# Python (see count_samples / find_tradeoffs). Asking the model to report a
# length it must keep consistent with its own array is the same class of mistake
# that broke stage 4.
ENUMERATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["samples", "notes"],
    "properties": {
        "samples": {"type": "array", "items": _SAMPLE_ITEM},
        "notes": {
            "type": "string",
            "description": "Anything that limited the enumeration. Empty if none.",
        },
    },
}

# ---------------------------------------------------------------------- prompt

ENUMERATE_SYSTEM = """You inventory the CONTROLLED COMPARISONS a machine-learning paper contains.

You are reasoning over the whole paper to work out how many samples it can
yield and what each one is about in general terms.

Describe each comparison generally, but specifically enough that a later stage
handed your description alone could find the exact rows in the paper. Two things
are the later stage's job and must NOT appear here: metric VALUES of any kind,
and verbatim quotes. Everything else that helps identify a comparison -- what the
technique does, how the paper labels the two variants, which table it sits in --
belongs in your description.

A controlled comparison is a pair of variants differing by ONE identifiable
technique, so the difference is attributable to that technique. Each
(technique, methodological setting) pair is ONE sample. WHERE the comparison
is evaluated -- the benchmarks/datasets AND the metrics -- are properties of
the sample, NOT what makes it distinct: list them all in `benchmarks` and
`metrics`, but never emit separate samples per benchmark or per metric.

`setting` is the recipe context only (base model, scale, training config).
Evaluation datasets belong in `benchmarks`. If two rows differ only in what
they are evaluated on, they are the SAME sample.

=== HOW MANY SAMPLES ===

- An ablation table with 4 removable components, scored on 3 benchmarks and
  2 metrics, yields 4 samples -- one per component, each listing all 3
  benchmarks and both metrics.
- The same recipe evaluated on 3 datasets yields ONE sample with 3
  benchmarks -- not 3 samples.
- A sweep over a design choice yields one sample per non-default value.
- Two genuinely different training recipes (e.g. the technique applied at
  7B vs 70B, or trained under a different budget) ARE different settings,
  hence different samples.

The technique may be one the paper PROPOSES or one it merely INVESTIGATES --
an ablated component, a baseline the authors ran themselves, a varied
experimental condition (retrieval on/off, cache policy, decoding strategy).
Both count.

Do not include rows copied from other papers with no shared setup, dataset
statistics, or curves where no technique is being compared.

=== DESCRIBING EACH SAMPLE ===

For each comparison also write a `description`: one or two sentences saying what
the technique does and what it is being compared against.

This is the handle a later stage uses to find the comparison again, so it has to
be general but discriminating. "A training strategy is varied" is useless -- a
paper may vary five things. "The heat-kernel endpoint filter strength sigma_0 is
raised from 0 (unfiltered endpoint, i.e. standard flow matching) to 40" points at
exactly one pair of rows.

Name the two variants as the paper labels them when you can see the labels, e.g.
"full model vs w/o gating", "Shared linear vs Sample-level". Do not quote long
passages and do not include any numbers -- that is the later stage's job.

=== ORIENTATION: always describe ENABLING the technique ===

Describe each comparison as adding / enabling / turning ON the named technique
-- the fuller arm relative to the reduced arm -- never as removing it. When the
paper labels the two variants "w/o X" and "full" (or "X off" / "X on",
"baseline" / "+X"), the reference is the reduced arm and the technique is what
gets ADDED, even though the paper writes the ablation as a removal.

  technique "syntactic constraints", arms "w/o constraints" vs "full":
    the technique is adding the constraints; "w/o constraints" is the reference.

=== DO NOT JUDGE THE OUTCOME ===

Do not say whether a comparison helped or hurt, and do not rank the arms. You
have not seen the numbers, and whether a move is good depends on the paper's
own goal for each metric -- higher attack-success is a win in an attack paper
and a loss in a defense paper. That call is made later, from the extracted
values. Your job is to inventory WHICH comparisons exist and WHAT they measure.

=== NO NUMBERS ===

No metric values, deltas, or percentages anywhere in your output.
Digits inside names are fine ("ResNet-50", "8 layers"); digits that are
measurements are not.

=== COVERAGE ===

Inventory every comparison across all tables and settings, appendix included.
List a comparison even when you cannot tell how it turned out -- how it turned
out is not your call.
"""


def _fmt_tables(tables: list[dict], limit: int) -> str:
    out, used = [], 0
    for i, t in enumerate(tables):
        block = (
            f"--- table {i} (label={t.get('label', '')!r}) ---\n"
            f"CAPTION: {t.get('caption', '')}\n{t.get('body', '')}\n"
        )
        if used + len(block) > limit:
            out.append(f"[... {len(tables) - i} further tables omitted for length]")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


def run_enumerate(paper: dict, parsed: dict, model: str | None = None
                  ) -> tuple[dict, dict]:
    """Enumerate candidate samples for one paper. Sees the full paper."""
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
        f"TITLE: {paper['title']}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"=== METHOD / INTRO (LaTeX) ===\n{method}\n\n"
        f"=== RESULTS / ABLATION PROSE (LaTeX) ===\n{results}\n\n"
        f"=== TABLES (LaTeX source, {parsed.get('n_tables', 0)} found) ===\n"
        f"{_fmt_tables(parsed.get('tables', []), config.MAX_TABLE_CHARS)}\n"
    )
    res, usage = chat_json(
        model or config.MODEL_ENUM_SUB, ENUMERATE_SYSTEM, user,
        ENUMERATE_SCHEMA, schema_name="enumeration", max_tokens=16000,
    )
    validate_against_schema(res, ENUMERATE_SCHEMA)
    return res, usage


# -------------------------------------------------------------- stage 5 RLM
#
# The paper never enters a single context window. It lives in a PaperStore;
# a ROOT agent (cheap, never judges samples) reads the outline, batches
# related sections and tables, and dispatches ENUMERATOR subagents -- each a
# fresh chat_json call running the unchanged ENUMERATE_SYSTEM over just its
# batch. Subagent samples accumulate in the environment's `findings`; the
# root only ever sees digests, so its context stays flat.
#
# Coverage is guaranteed mechanically, not by model discipline: submit() is
# refused until every section and table id is covered or explicitly skipped
# with a recorded reason. (RLM harnesses leave decomposition AND completion
# to the model; our task has a known completion criterion, so it is enforced
# in code -- coverage by construction.)

ROOT_SYSTEM_TMPL = """\
You orchestrate the enumeration of a machine-learning paper you cannot read
directly. The goal: every CONTROLLED COMPARISON in the paper ends up in the
environment's findings list. One sample = one (technique, setting, metric)
triple. You do not judge or filter samples yourself -- enumerator subagents
do that. Your deliverable is coverage: submit() succeeds only when every
section and table has been enumerated or explicitly skipped.

THE ENVIRONMENT

The paper is stored outside your context as sections s0..sN and tables
t0..tM, in document order. The outline you are given lists every id with its
kind, title, and size in chars.

YOUR TOOLS

- peek(target, offset): one ~{peek_chars}-char page of raw text. For
  orientation only -- reading the paper yourself is not your job.
- grep(query): literal search across the whole paper (case-insensitive,
  whitespace-tolerant). Use it to find where a table is discussed.
- spawn_enumerator(targets): dispatch a subagent that reads ONLY those
  sections/tables and appends every comparison it finds to the findings
  list. You get back a digest (counts and one line per sample), never the
  full findings. A batch may carry at most {batch_chars} chars total
  (sizes are in the outline).
- coverage(): what is covered, skipped, and still uncovered.
- findings_brief(): one compressed line per finding so far.
- skip(targets, reason): mark targets that genuinely contain no comparisons
  worth enumerating (a bibliography, a code listing, an acknowledgements
  section). Not a shortcut -- every skip is recorded in the output.
- submit(): finish. Refused while anything is neither covered nor skipped.

PLAYBOOK

1. Read the outline. Group related targets into batches: keep each table in
   the same batch as the section that discusses it (grep the table's label
   or caption words when unsure). Pack batches close to the {batch_chars}-char
   budget -- fat batches, few calls.
2. Dispatch spawn_enumerator over every batch. You may issue SEVERAL
   spawn_enumerator calls in one turn; they run in parallel.
3. Results and appendix sections are where ablation studies live. They are
   never to be left uncovered, and skipping them is almost always wrong.
4. Check findings_brief() and coverage(). A results section that yielded
   zero samples is suspicious -- re-dispatch it in a different batch or peek
   to see why. Duplicated findings across batches are fine (dedupe is
   automatic downstream).
5. If an enumerator fails (its output was truncated, or an error comes
   back), its targets stay uncovered: split that batch into smaller ones
   and re-dispatch. Dense, table-heavy targets yield many samples, so give
   them small batches.
6. submit().

RULES

- No metric values or verbatim quotes anywhere -- a later stage extracts
  numbers; findings carry only comparisons and descriptions.
- Do not reconstruct the paper via repeated peeks. Peek pages are capped and
  your context is small; enumeration belongs to subagents.
- Respond only with tool calls.
"""

PART_NOTE = ("NOTE: you are seeing PART of a paper: only the sections and "
             "tables listed below. Enumerate the comparisons visible in this "
             "part; other parts are handled separately.")


def _root_tools() -> list[dict]:
    def tool(name, desc, props=None, req=None):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": props or {},
                           "required": req or []}}}

    ids = {"type": "array", "items": {"type": "string"},
           "description": "target ids from the outline, e.g. [\"s3\",\"t1\"]"}
    return [
        tool("peek", "One page of raw text from a section or table.",
             {"target": {"type": "string"},
              "offset": {"type": "integer", "description": "char offset"}},
             ["target"]),
        tool("grep", "Literal search across the whole paper.",
             {"query": {"type": "string"}}, ["query"]),
        tool("spawn_enumerator",
             "Dispatch a subagent to enumerate ONLY these targets.",
             {"targets": ids}, ["targets"]),
        tool("coverage", "Covered / skipped / uncovered target ids."),
        tool("findings_brief", "One line per finding accumulated so far."),
        tool("skip", "Mark targets as deliberately not enumerated.",
             {"targets": ids, "reason": {"type": "string"}},
             ["targets", "reason"]),
        tool("submit", "Finish. Refused while coverage is incomplete."),
    ]


class EnumerationEnv:
    """The RLM environment for one paper: findings + coverage ledger.

    `enum_fn(target_ids) -> {"samples": [...], "notes": str}` is the
    enumerator subagent; injected so tests need no LLM. All mutation happens
    under a lock because one root turn may fan out parallel spawns.
    """

    def __init__(self, store: PaperStore, enum_fn,
                 max_batch_chars: int | None = None,
                 checkpoint_path=None):
        self.store = store
        self.enum_fn = enum_fn
        self.max_batch = max_batch_chars or config.RLM_SUBAGENT_MAX_CHARS
        self.checkpoint_path = checkpoint_path
        self.findings: list[dict] = []
        self.ledger: dict[str, str] = {}      # id -> covered | skipped
        self.skipped: list[dict] = []
        self.batches: list[list[str]] = []
        self.notes: list[str] = []
        self.n_dispatches = 0
        self._lock = threading.Lock()

    # -------------------------------------------------------- checkpointing
    #
    # Every completed enumerator batch is written to disk immediately, so a
    # crash or outage mid-paper resumes from the finished batches instead of
    # redoing them. This is the RLM's state-in-the-environment property made
    # durable: the environment IS the progress, so persisting it is enough.

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        write_json(self.checkpoint_path, {
            "findings": self.findings, "ledger": self.ledger,
            "batches": self.batches, "skipped": self.skipped,
            "notes": self.notes,
        })

    def restore(self, ckpt: dict) -> None:
        """Seed the environment from a previous run's checkpoint. Only ids
        that still exist in the store are honored (a reparse could change
        them; stale ids must not satisfy the coverage gate)."""
        known = set(self.store.ids())
        self.ledger = {i: v for i, v in (ckpt.get("ledger") or {}).items()
                       if i in known}
        self.findings = list(ckpt.get("findings") or [])
        self.batches = list(ckpt.get("batches") or [])
        self.skipped = list(ckpt.get("skipped") or [])
        self.notes = list(ckpt.get("notes") or [])

    # ------------------------------------------------------------- helpers

    def _check_ids(self, ids) -> list[str]:
        if not isinstance(ids, list) or not ids or \
                not all(isinstance(i, str) for i in ids):
            raise ToolError("`targets` must be a non-empty list of target "
                            "ids from the outline")
        known = set(self.store.ids())
        bad = [i for i in ids if i not in known]
        if bad:
            raise ToolError(f"unknown target ids: {', '.join(bad)}. Valid "
                            f"ids are listed in the outline.")
        return ids

    def _uncovered(self) -> list[str]:
        return [i for i in self.store.ids() if i not in self.ledger]

    # -------------------------------------------------------------- tools

    def dispatch(self, name: str, args: dict) -> str:
        with self._lock:
            self.n_dispatches += 1
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        return handler(args)

    def _tool_peek(self, args: dict) -> str:
        try:
            return self.store.peek(args.get("target", ""),
                                   int(args.get("offset") or 0))
        except (KeyError, ValueError) as e:
            raise ToolError(str(e)) from None

    def _tool_grep(self, args: dict) -> str:
        hits = self.store.grep(str(args.get("query", "")))
        if not hits:
            return "no matches"
        return "\n".join(
            f"{h['target']}"
            f"{' (caption)' if h.get('in_caption') else ''}"
            f" @{h['offset']}: ...{h['context']}..."
            for h in hits)

    def _tool_spawn_enumerator(self, args: dict) -> str:
        ids = self._check_ids(args.get("targets"))
        size = sum(self.store.chars(i) for i in ids)
        if size > self.max_batch:
            raise ToolError(
                f"batch is {size} chars, budget is {self.max_batch}. Split "
                f"it (sizes are in the outline).")
        already = [i for i in ids if self.ledger.get(i) == "covered"]
        try:
            res = self.enum_fn(ids)
        except Exception as e:
            # The subagent failing must not kill the paper: hand the error
            # to the root, which can split the batch or retry. Targets stay
            # uncovered, so the submit gate still guarantees nothing is
            # silently dropped.
            raise ToolError(
                f"enumerator subagent failed on {ids}: "
                f"{type(e).__name__}: {str(e)[:200]}. These targets remain "
                f"uncovered -- split the batch into smaller ones and "
                f"re-dispatch.") from None
        samples = res.get("samples") or []
        with self._lock:
            for s in samples:
                s["_targets"] = list(ids)
                self.findings.append(s)
            for i in ids:
                self.ledger[i] = "covered"
            self.batches.append(list(ids))
            if res.get("notes"):
                self.notes.append(res["notes"])
            self._save_checkpoint()
        lines = [f"batch {ids}: {len(samples)} samples "
                 f"(findings total: {len(self.findings)})"]
        if already:
            lines.append(f"note: {', '.join(already)} had already been "
                         f"covered; duplicates are deduped downstream")
        lines += [f" - {s.get('technique', '?')[:40]} | "
                  f"{s.get('setting', '?')[:30]} | "
                  f"{', '.join(s.get('metrics') or [])[:30]}" for s in samples]
        return "\n".join(lines)

    def _tool_coverage(self, _args: dict) -> str:
        covered = [i for i, v in self.ledger.items() if v == "covered"]
        skipped = [i for i, v in self.ledger.items() if v == "skipped"]
        out = [f"covered ({len(covered)}): {', '.join(covered) or '-'}",
               f"skipped ({len(skipped)}): {', '.join(skipped) or '-'}"]
        unc = self._uncovered()
        if unc:
            out.append(f"UNCOVERED ({len(unc)}): " + ", ".join(
                f"{i} ({self.store.chars(i)} chars)" for i in unc))
        else:
            out.append("UNCOVERED: none -- submit() will be accepted")
        return "\n".join(out)

    def _tool_findings_brief(self, _args: dict) -> str:
        if not self.findings:
            return "no findings yet"
        return "\n".join(
            f"{n}. {s.get('technique', '?')[:40]} | "
            f"{s.get('setting', '?')[:30]} | "
            f"{', '.join(s.get('metrics') or [])[:24]}"
            for n, s in enumerate(self.findings, 1))

    def _tool_skip(self, args: dict) -> str:
        ids = self._check_ids(args.get("targets"))
        reason = str(args.get("reason") or "").strip()
        if not reason:
            raise ToolError("skip requires a non-empty reason; it is "
                            "recorded in the output")
        with self._lock:
            for i in ids:
                self.ledger[i] = "skipped"
            self.skipped.append({"targets": ids, "reason": reason})
            self._save_checkpoint()
        return f"skipped {', '.join(ids)}"

    def _tool_submit(self, _args: dict) -> str:
        unc = self._uncovered()
        if unc:
            raise ToolError(
                "coverage incomplete -- these targets are neither covered "
                "nor skipped: " + ", ".join(unc))
        return f"accepted: {len(self.findings)} findings"


def run_enumerate_subset(paper: dict, store: PaperStore,
                         target_ids: list[str]) -> tuple[dict, dict]:
    """One enumerator subagent call over a batch of sections/tables."""
    parts = []
    for tid in target_ids:
        body = store.body(tid)
        if tid.startswith("t"):
            t = store._get(tid)
            parts.append(f"--- table [{tid}] (label={t['title']!r}) ---\n"
                         f"CAPTION: {t['caption']}\n{body}")
        else:
            t = store._get(tid)
            parts.append(f"## [{tid}] {t['title']}\n{body}")

    user = (
        f"TITLE: {paper['title']}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"{PART_NOTE}\n\n"
        f"FULL PAPER OUTLINE (for orientation):\n{store.outline()}\n\n"
        f"=== YOUR PART: {', '.join(target_ids)} ===\n\n"
        + "\n\n".join(parts)
    )
    res, usage = chat_json(
        config.MODEL_ENUM_SUB, ENUMERATE_SYSTEM, user, ENUMERATE_SCHEMA,
        schema_name="enumeration", max_tokens=config.RLM_ENUM_MAX_TOKENS,
        reasoning=config.reasoning_param(),
        traj=f"05_enum_{paper.get('base_id', '')}_{'-'.join(target_ids)[:40]}",
    )
    validate_against_schema(res, ENUMERATE_SCHEMA)
    return res, usage


def merge_findings(findings: list[dict]) -> list[dict]:
    """Dedupe on normalized (technique, setting); first wins."""
    seen, out = set(), []
    for s in findings:
        k = (_key(s.get("technique", "")), _key(s.get("setting", "")))
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def run_enumerate_rlm(paper: dict, parsed: dict) -> tuple[dict, dict]:
    """Enumerate one paper without the paper ever entering one context.

    Fast path: a paper that fits one subagent budget is enumerated in a
    single call (identical to the old single-shot stage 5, appendix
    included). Otherwise the root agent decomposes and dispatches.
    """
    store = PaperStore(parsed)
    ids = store.ids()

    if config.RLM_FAST_PATH and store.total_chars <= config.RLM_SUBAGENT_MAX_CHARS:
        res, usage = run_enumerate_subset(paper, store, ids)
        samples = [dict(s, _targets=ids) for s in (res.get("samples") or [])]
        return {
            "samples": merge_findings(samples),
            "notes": res.get("notes", ""),
            "_rlm": {"fast_path": True, "batches": [ids], "skipped": [],
                     "root_dispatches": 0},
        }, usage

    def enum_fn(target_ids: list[str]) -> dict:
        res, _ = run_enumerate_subset(paper, store, target_ids)
        return res

    ckpt_path = None
    bid = paper.get("base_id") or ""
    if bid:
        ckpt_path = config.CANDIDATES_PARTIAL_DIR / f"{bid}.json"

    env = EnumerationEnv(store, enum_fn, checkpoint_path=ckpt_path)
    resume_note = ""
    if ckpt_path is not None:
        prior = read_json(ckpt_path)
        if prior:
            env.restore(prior)
            done = sorted(env.ledger)
            resume_note = (
                f"\nRESUMING an interrupted run: {len(done)} targets are "
                f"already covered or skipped ({', '.join(done)}) and "
                f"{len(env.findings)} findings are already recorded. Do NOT "
                f"re-enumerate those -- check coverage() and dispatch only "
                f"what is still uncovered.\n")

    system = ROOT_SYSTEM_TMPL.format(
        peek_chars=config.RLM_PEEK_CHARS,
        batch_chars=config.RLM_SUBAGENT_MAX_CHARS)
    user = (
        f"PAPER: {paper['title']}\n\n"
        f"ABSTRACT: {paper['abstract']}\n\n"
        f"The paper is stored in the environment: "
        f"{sum(1 for i in ids if i.startswith('s'))} sections and "
        f"{sum(1 for i in ids if i.startswith('t'))} tables, "
        f"{store.total_chars:,} chars total.\n\n"
        f"OUTLINE:\n{store.outline()}\n"
        f"{resume_note}\n"
        f"Enumerate everything, then submit."
    )
    chat_tools(
        config.MODEL_RLM_ROOT, system, user, _root_tools(), env.dispatch,
        terminal_tools={"submit"}, max_turns=config.RLM_MAX_ROOT_TURNS,
        max_tokens=config.RLM_ROOT_MAX_TOKENS,
        traj=f"05_root_{bid}",
    )

    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()          # final result supersedes the checkpoint

    notes = list(env.notes)
    for sk in env.skipped:
        notes.append(f"skipped {', '.join(sk['targets'])}: {sk['reason']}")
    return {
        "samples": merge_findings(env.findings),
        "notes": "; ".join(n for n in notes if n),
        "_rlm": {"fast_path": False, "batches": env.batches,
                 "skipped": env.skipped, "root_dispatches": env.n_dispatches},
    }, {}


# ----------------------------------------------------------- ids + aggregates

def _key(s: str) -> str:
    return " ".join(sorted(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split()))


def annotate(result: dict, base_id: str = "") -> dict:
    """Attach ids to each sample, then the aggregates.

    `sample_id` is assigned here, in Python, from the array index. Stage 6 makes
    one call per sample and echoes this id back, so extractions match stage-5
    rows exactly instead of being fuzzy-matched on token overlap.
    """
    samples = result.get("samples") or []
    for i, s in enumerate(samples):
        s["sample_id"] = f"{base_id}-s{i:03d}" if base_id else f"s{i:03d}"
    result["n_samples"] = len(samples)          # computed, not reported
    return result


def summarize(per_paper: dict) -> dict:
    """Corpus-level counts over {base_id: enumeration result}."""
    from collections import Counter
    tot = 0
    metric_names = Counter()
    per_paper_counts = {}

    for bid, res in per_paper.items():
        rows = res.get("samples") or []
        per_paper_counts[bid] = len(rows)
        tot += len(rows)
        for s in rows:
            for m in s.get("metrics") or []:
                metric_names[m[:40]] += 1

    return {
        "papers": len(per_paper),
        "total_candidate_samples": tot,
        "samples_per_paper": per_paper_counts,
        "metric_mentions": dict(metric_names.most_common(25)),
    }
