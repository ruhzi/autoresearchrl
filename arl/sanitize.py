"""Build the leak-reduced context the model is allowed to see.

Two kinds of leak, handled differently:

1. Explicit numeric leaks -- "we improve accuracy by 2.3 points" in the abstract.
   Handled by deleting whole result-bearing sections plus a regex scrubber over
   what remains. Every redaction is counted so the cost is auditable.

2. Prominence leaks -- authors don't write three paragraphs about a component
   they ablated away as useless, so the amount of text a component gets encodes
   that it survived. A regex cannot fix this. What we do here is *measure* it:
   `leak_probe_context()` returns section titles and lengths with all technical
   content stripped, so a model given only that establishes the floor score.
   Whatever the probe achieves is the number a real system has to beat.
"""
from __future__ import annotations

import re

from arl.metrics import METRIC_WORDS

# Sections that may leak results and are dropped wholesale.
DROP_KINDS = {"results", "ablation", "conclusion", "appendix"}
KEEP_KINDS = {"frontmatter", "intro", "method", "related", "other"}

# Content guard for sections still classified `other` after title matching and
# parent inheritance. Measured on 20 real papers, `other` was 60% of the context
# and 34 of those sections carried tables or result percentages -- titles like
# "7.1 The gap is large and stable" state the finding outright. A section whose
# body looks like it reports numbers is treated as results regardless of title.
_TABLE_BEGIN_RE = re.compile(r"\\begin\{table")
_PCT_NUM_RE = re.compile(r"\d+\.\d+\s*\\?%")
_TABLE_ROW_RE = re.compile(r"\d+\.\d+\s*&")
OTHER_RESULT_HITS = 3


def looks_like_results(body: str) -> tuple[bool, int]:
    """Does this section body report numeric results?"""
    hits = (len(_TABLE_BEGIN_RE.findall(body))
            + len(_PCT_NUM_RE.findall(body))
            + len(_TABLE_ROW_RE.findall(body)))
    return hits >= OTHER_RESULT_HITS, hits

REDACT = "[REDACTED]"

_ENV_DROP = ("figure", "figure*", "table", "table*", "longtable", "tabular",
             "tabularx", "threeparttable", "wrapfigure", "subfigure")

# Multi-word superlative phrases are redacted FIRST. Regex alternation is
# leftmost-first, not longest-first, so "achieves state" would otherwise match
# before "state-of-the-art" and leave "-of-the-art" behind as a live hint.
_PHRASE_FIRST = [
    r"state[- ]of[- ]the[- ]art",
    r"\bSOTA\b",
    r"\bnew best\b",
    r"\bbest[- ]performing\b",
]
_PHRASE_FIRST_RE = re.compile("|".join(_PHRASE_FIRST), re.IGNORECASE)

_CLAIM_PATTERNS = [
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bsurpass(?:es|ed|ing)?\b",
    r"\bexceed(?:s|ed|ing)?\b",
    r"\bbeat(?:s|en|ing)?\b",
    r"\bimprove(?:s|d|ment|ments)?\s+(?:of|by)\b",
    r"\bgain(?:s)?\s+of\b",
    r"\breduction\s+of\b",
    r"\bspeedup\s+of\b",
    r"\brelative(?:ly)?\s+(?:improvement|reduction)\b",
    # "achieves the highest accuracy" / "yields the lowest WER" -- a superlative
    # verb phrase leaks the ranking with no number attached at all. Found by
    # auditing real contexts: the earlier pattern required the superlative to
    # follow the verb directly, so an intervening article defeated it.
    r"\b(?:achieve|yield|attain|obtain|reach|deliver|record|report)(?:s|ed|ing)?"
    r"\s+(?:the\s+|a\s+|new\s+)*(?:state|best|worst|highest|lowest|strongest|"
    r"weakest|largest|smallest|top)\b",
    # bare superlative sitting on a metric: "the highest downstream accuracy"
    r"\b(?:highest|lowest|best|worst|strongest|weakest)\s+(?:\w+\s+){0,3}"
    r"(?:accuracy|performance|score|result|results|f1|bleu|perplexity|ppl|wer|"
    r"cer|auc|iou|fid|reward|success|throughput|latency|error)\b",
]
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_PCT_RE = re.compile(rf"{_NUM}\s*\\?%")
_POINTS_RE = re.compile(rf"{_NUM}\s*(?:percentage\s+)?(?:points?|pts?|pp)\b", re.I)
_XFOLD_RE = re.compile(rf"{_NUM}\s*(?:x|×|-fold|\s*fold)\b(?=\s*(?:faster|slower|speedup|smaller|larger|fewer|more|less|reduction))", re.I)

# metric-adjacent numbers, both orders, with a short window between
_METRIC_ALT = "|".join(
    sorted((re.escape(w) for w in METRIC_WORDS if len(w) >= 2), key=len, reverse=True)
)
_NUM_THEN_METRIC = re.compile(
    rf"(?P<num>{_NUM})(?P<mid>(?:\s+\w+){{0,3}}\s+(?:{_METRIC_ALT}))\b", re.I
)
_METRIC_THEN_NUM = re.compile(
    rf"(?P<met>(?:{_METRIC_ALT}))(?P<mid>(?:\s+\w+){{0,3}}\s+(?:of|is|to|at|reaches?|=)?\s*)(?P<num>{_NUM})\b",
    re.I,
)


def _drop_environments(tex: str) -> str:
    for env in _ENV_DROP:
        tex = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            " ",
            tex,
            flags=re.S,
        )
    return tex


def _tidy_latex(tex: str) -> str:
    tex = re.sub(r"\\(?:cite[a-zA-Z]*|citep|citet)\s*(?:\[[^\]]*\])*\s*\{[^}]*\}",
                 "[CITE]", tex)
    tex = re.sub(r"\\(?:ref|autoref|cref|Cref|eqref)\s*\{[^}]*\}", "[REF]", tex)
    tex = re.sub(r"\\label\s*\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(?:footnote|thanks)\s*\{", " (", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?", " ", tex)  # remaining commands
    tex = tex.replace("{", " ").replace("}", " ").replace("&", " ")
    # Strip math delimiters so numbers inside them are bare and the scrubber can
    # see them. Found by audit: "achieves an accuracy of $0.683$" survived
    # because the metric-adjacent pattern expects a plain number after "of".
    tex = tex.replace("$$", " ").replace("$", " ")
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


def scrub(text: str) -> tuple[str, dict]:
    """Redact result-bearing numbers and comparative claims. Returns counts."""
    counts = {"pct": 0, "points": 0, "xfold": 0, "metric_adjacent": 0, "claims": 0}

    def _sub(pattern, key, repl=REDACT):
        nonlocal text
        text, n = pattern.subn(repl, text)
        counts[key] += n

    _sub(_PHRASE_FIRST_RE, "claims", "[CLAIM-REDACTED]")
    _sub(_PCT_RE, "pct")
    _sub(_POINTS_RE, "points")
    _sub(_XFOLD_RE, "xfold")

    def _num_only(m: re.Match) -> str:
        return REDACT + m.group("mid")

    text, n = _NUM_THEN_METRIC.subn(_num_only, text)
    counts["metric_adjacent"] += n

    def _num_only2(m: re.Match) -> str:
        return m.group("met") + m.group("mid") + REDACT

    text, n = _METRIC_THEN_NUM.subn(_num_only2, text)
    counts["metric_adjacent"] += n

    _sub(_CLAIM_RE, "claims", "[CLAIM-REDACTED]")

    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return text, counts


def build_context(parsed: dict, abstract: str = "", max_chars: int = 60_000) -> dict:
    """Assemble the model-visible context from a parsed paper."""
    kept, dropped = [], []

    if abstract:
        kept.append(("abstract", abstract))

    for s in parsed.get("sections", []):
        if s["kind"] in DROP_KINDS or s["kind"] not in KEEP_KINDS:
            dropped.append({"title": s["title"], "kind": s["kind"],
                            "chars": s["chars"], "reason": "kind"})
            continue

        # Last line of defence: an unclassified section that reports numbers is
        # results content whatever its title says.
        if s["kind"] == "other":
            is_res, hits = looks_like_results(s.get("body", ""))
            if is_res:
                dropped.append({"title": s["title"], "kind": "other",
                                "chars": s["chars"], "reason": "content_guard",
                                "hits": hits})
                continue

        body = s.get("body", "")
        if s["kind"] == "frontmatter":
            # The abstract is added explicitly above; leaving the abstract
            # environment in the frontmatter would duplicate it, which both
            # wastes context and doubles the weight of its (redacted) claims.
            body = re.sub(
                r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", body, flags=re.S
            )
        if not body.strip():
            continue
        header = s["title"] or s["kind"]
        kept.append((header, body))

    chunks, all_counts = [], {}
    for header, body in kept:
        body = _tidy_latex(_drop_environments(body))
        body, counts = scrub(body)
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v
        chunks.append(f"## {header}\n\n{body}")

    context = "\n\n".join(chunks).strip()
    truncated = False
    if len(context) > max_chars:
        context = context[:max_chars]
        truncated = True

    return {
        "context": context,
        "context_chars": len(context),
        "truncated": truncated,
        "kept_sections": [h for h, _ in kept],
        "dropped_sections": dropped,
        "redactions": all_counts,
    }


def leak_probe_context(parsed: dict) -> dict:
    """Structure-only view: section titles and lengths, no technical content.

    A model scored on this alone cannot know anything about the *method* -- only
    about how the paper is shaped and which component got the most ink. Its
    score is the floor. If it lands near the score of a model that sees the full
    context, the benchmark is measuring paper layout, not research taste.
    """
    lines = []
    for s in parsed.get("sections", []):
        if s["kind"] in DROP_KINDS:
            continue
        lines.append(f"- {s['kind']}: \"{s['title']}\" ({s['chars']} chars)")
    return {
        "probe_context": "Section outline of a paper (content withheld):\n"
        + "\n".join(lines),
        "n_sections": len(lines),
    }
