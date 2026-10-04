"""The paper as an environment, not as prompt filler.

PaperStore wraps one stage-3 parsed JSON and is the single source of truth for
paper text. Stage 5's root agent orients itself through `outline`/`peek`/`grep`
(small, capped returns) and Python builds subagent prompts from `body` -- the
full paper never enters any single context window.

Ids are positional and stable: sections s0..sN, tables t0..tM, in document
order. Sizes are computed from the bodies themselves, not trusted from the
parse metadata.

"External storage" here means external to the context window: local JSON plus
deterministic accessors. No embeddings -- grep is literal (case-insensitive,
whitespace-normalized) so retrieval stays auditable.
"""
from __future__ import annotations

import re

import config


class AnchorError(Exception):
    """A pointer that cannot be resolved to exactly one span: anchor not
    found, anchor ambiguous, or end before start. Structured refusal --
    the retrieval agent gets this back verbatim and repairs its pointer."""


class PaperStore:
    def __init__(self, parsed: dict, peek_chars: int | None = None,
                 grep_max_hits: int | None = None,
                 grep_context_chars: int | None = None):
        self._peek_chars = peek_chars or config.RLM_PEEK_CHARS
        self._grep_max_hits = grep_max_hits or config.RLM_GREP_MAX_HITS
        self._grep_ctx = grep_context_chars or config.RLM_GREP_CONTEXT_CHARS

        self._targets: dict[str, dict] = {}
        for i, s in enumerate(parsed.get("sections") or []):
            self._targets[f"s{i}"] = {
                "kind": s.get("kind", "other"),
                "title": s.get("title") or "",
                "body": s.get("body") or "",
                "caption": "",
            }
        for i, t in enumerate(parsed.get("tables") or []):
            self._targets[f"t{i}"] = {
                "kind": "table",
                "title": t.get("label") or "",
                "body": t.get("body") or "",
                "caption": t.get("caption") or "",
            }

    # ---------------------------------------------------------------- access

    def ids(self) -> list[str]:
        return list(self._targets)

    def _get(self, target_id: str) -> dict:
        try:
            return self._targets[target_id]
        except KeyError:
            raise KeyError(
                f"unknown target {target_id!r}; valid ids are s0..s(N) and "
                f"t0..t(M) as listed in the outline") from None

    def body(self, target_id: str) -> str:
        return self._get(target_id)["body"]

    def chars(self, target_id: str) -> int:
        return len(self._get(target_id)["body"])

    @property
    def total_chars(self) -> int:
        return sum(len(t["body"]) for t in self._targets.values())

    # --------------------------------------------------------------- outline

    def outline(self) -> str:
        """Compact TOC: every target with id, kind, title/label, size.

        Sizes are included so the root can budget spawn batches without
        peeking. Captions are truncated -- the outline must fit in any prompt.
        """
        lines = []
        for tid, t in self._targets.items():
            if t["kind"] == "table":
                cap = t["caption"][:120]
                if len(t["caption"]) > 120:
                    cap += "..."
                lines.append(f"{tid} [table] label={t['title']!r} "
                             f"({len(t['body'])} chars): {cap}")
            else:
                lines.append(f"{tid} [{t['kind']}] {t['title']!r} "
                             f"({len(t['body'])} chars)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ peek

    def peek(self, target_id: str, offset: int = 0) -> str:
        """One page of raw body from `offset`. For orientation, not reading."""
        body = self._get(target_id)["body"]
        n = len(body)
        start = max(0, min(int(offset), n))
        end = min(start + self._peek_chars, n)
        return (f"({target_id}: showing {start}..{end} of {n} chars)\n"
                f"{body[start:end]}")

    # ----------------------------------------------------------------- slice

    def _find_anchor(self, body: str, anchor: str, role: str,
                     target_id: str) -> re.Match:
        toks = (anchor or "").split()
        if not toks:
            raise AnchorError(f"{role} is empty")
        # Same whitespace-tolerance as grep (LaTeX line-wraps freely), but
        # case-sensitive: anchors are copied from tool output, so case must
        # already agree with the source.
        pat = re.compile(r"\s+".join(re.escape(t) for t in toks))
        matches = list(pat.finditer(body))
        if not matches:
            raise AnchorError(f"{role} {anchor!r} not found in {target_id}. "
                              f"Copy it exactly from the tool output.")
        if len(matches) > 1:
            raise AnchorError(f"{role} {anchor!r} found {len(matches)} times "
                              f"in {target_id}; use a longer, unique anchor.")
        return matches[0]

    def slice(self, pointer: dict) -> str:
        """Resolve a pointer to the EXACT source bytes it brackets.

        pointer = {target, start_anchor, end_anchor}. Each anchor must occur
        exactly once in the target's body. The returned text is a literal
        substring of the source -- this is the losslessness core: payload is
        copied from disk here, never typed by a model.
        """
        tid = pointer.get("target", "")
        body = self.body(tid)                       # KeyError on bad id
        start = self._find_anchor(body, pointer.get("start_anchor", ""),
                                  "start_anchor", tid)
        end = self._find_anchor(body, pointer.get("end_anchor", ""),
                                "end_anchor", tid)
        if end.end() <= start.start():
            raise AnchorError(
                f"end_anchor occurs before start_anchor in {tid}; swap them "
                f"or pick anchors that bracket the passage.")
        return body[start.start():end.end()]

    # ------------------------------------------------------------------ grep

    def grep(self, query: str) -> list[dict]:
        """Literal search across all bodies and table captions.

        Case-insensitive; whitespace in the query matches any whitespace run
        in the source (LaTeX line-wraps freely). Regex metacharacters in the
        query are literal. Hits are capped at grep_max_hits total.

        Bodies and captions are searched SEPARATELY so body offsets always
        index the body itself (what peek/slice operate on). Caption hits are
        marked in_caption -- useful for finding the right table, but anchors
        must be copied from the body, not the caption.
        """
        toks = query.split()
        if not toks:
            return []
        pat = re.compile(r"\s+".join(re.escape(t) for t in toks),
                         re.IGNORECASE)
        hits: list[dict] = []

        def scan(tid: str, text: str, in_caption: bool) -> bool:
            for m in pat.finditer(text):
                lo = max(0, m.start() - self._grep_ctx)
                hi = min(len(text), m.end() + self._grep_ctx)
                hit = {"target": tid, "offset": m.start(),
                       "context": text[lo:hi]}
                if in_caption:
                    hit["in_caption"] = True
                hits.append(hit)
                if len(hits) >= self._grep_max_hits:
                    return True
            return False

        for tid, t in self._targets.items():
            if scan(tid, t["body"], False):
                return hits
            if t["caption"] and scan(tid, t["caption"], True):
                return hits
        return hits
