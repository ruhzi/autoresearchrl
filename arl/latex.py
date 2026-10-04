"""LaTeX parsing: flatten includes, split sections, pull out tables.

This module is deliberately dumb about semantics. Its only job is to turn a
directory of .tex files into (a) a section-labelled document and (b) a list of
table environments with captions. Everything that requires judgement happens
later, in the LLM extraction stage.
"""
from __future__ import annotations

import re
from pathlib import Path

# ------------------------------------------------------------------ utilities

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)
# (?![a-zA-Z]) guards against \includegraphics, \includeonly, \inputminted
# etc. -- only the literal \input / \include file-inlining commands match.
_INPUT_RE = re.compile(
    r"\\(?:input|include)(?![a-zA-Z])\s*(?:\{([^}]*)\}|([^\s{}\\]+))")
_DOCCLASS_RE = re.compile(r"\\documentclass")


def strip_comments(tex: str) -> str:
    """Remove LaTeX line comments, preserving escaped \\%."""
    return _COMMENT_RE.sub("", tex)


def read_tex(path: Path) -> str:
    """Read a .tex file, tolerating non-UTF-8 sources.

    A fair number of arXiv submissions are latin-1 / cp1252. Decoding those as
    UTF-8 with errors='replace' injects U+FFFD into the text, which shows up
    inside section titles and defeats keyword matching. Try strict UTF-8 first,
    then latin-1, and only then give up and replace.
    """
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def find_main(root: Path) -> Path | None:
    """Pick the .tex file that looks like the document root.

    Prefers a file containing \\documentclass; among several, prefers the
    largest (submissions often ship a stub alongside the real paper).
    """
    candidates = list(root.rglob("*.tex"))
    if not candidates:
        return None
    with_class = []
    for p in candidates:
        try:
            head = read_tex(p)
        except Exception:
            continue
        if _DOCCLASS_RE.search(head):
            with_class.append((len(head), p))
    if with_class:
        return max(with_class)[1]
    return max(((p.stat().st_size, p) for p in candidates))[1]


def flatten(main: Path, root: Path, _depth: int = 0, _seen: set | None = None) -> str:
    """Inline \\input / \\include recursively so the paper is one string."""
    if _seen is None:
        _seen = set()
    if _depth > 8 or main in _seen:
        return ""
    _seen.add(main)

    try:
        tex = read_tex(main)
    except Exception:
        return ""
    tex = strip_comments(tex)

    def repl(m: re.Match) -> str:
        name = (m.group(1) or m.group(2) or "").strip()
        if not name:
            return ""
        for cand in (name, name + ".tex"):
            for base in (main.parent, root):
                p = (base / cand)
                if p.is_file():
                    return "\n" + flatten(p, root, _depth + 1, _seen) + "\n"
        return ""  # missing include -- drop it

    return _INPUT_RE.sub(repl, tex)


# ------------------------------------------------------------------- sections

_SEC_RE = re.compile(
    r"\\(?P<cmd>section|subsection|subsubsection)\*?\s*\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\}"
)

_KIND_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    # order matters: first match wins
    ("ablation", ("ablation", "ablations")),
    ("results", (
        "result", "results", "experiment", "experiments", "experimental",
        "evaluation", "evaluations", "empirical", "benchmark", "benchmarks",
        "analysis", "comparison", "comparisons", "findings", "performance",
        "main result", "quantitative",
    )),
    ("method", (
        "method", "methods", "methodology", "approach", "our approach", "model",
        "architecture", "framework", "design", "algorithm", "formulation",
        "proposed", "implementation", "preliminaries", "background", "setup",
        "training", "objective", "notation",
    )),
    ("related", ("related work", "related", "prior work", "literature")),
    ("intro", ("introduction", "intro", "overview", "motivation")),
    ("conclusion", (
        "conclusion", "conclusions", "discussion", "limitations",
        "future work", "broader impact", "ethics", "acknowledg",
    )),
    ("appendix", ("appendix", "appendices", "supplementary", "supplement")),
]


def _strip_numbering(title: str) -> str:
    """Drop a manual number prefix: '4.3 The distortion...' -> 'The distortion...'.

    Authors number sections by hand often enough that the digits reach the title
    string, and a replacement char from a non-UTF-8 source can sit in front of
    them ('\\ufffd4 Axis I'). Both defeat naive keyword matching.
    """
    return re.sub(r"^[^A-Za-z]*(?:\d+(?:\.\d+)*)?[\s.):-]*", "", title).strip()


def classify_section(title: str) -> str:
    t = re.sub(r"[^a-z ]+", " ", _strip_numbering(title).lower()).strip()
    t = re.sub(r"\s+", " ", t)
    for kind, keys in _KIND_PATTERNS:
        for k in keys:
            if k in t:
                return kind
    return "other"


_LEVEL_RANK = {"section": 0, "subsection": 1, "subsubsection": 2}

# A subsection under a results/ablation heading is results content even when its
# own title carries no keyword ("Out-of-distribution generalization" under
# "Results"). Measured on 20 real papers, 160 of 190 unclassified sections were
# sub-level with a correctly-classified parent -- so inheritance, not a longer
# keyword list, is what plugs the leak. Only inherit a kind that actually
# matters; an `other` parent teaches us nothing.
_INHERITABLE = {"results", "ablation", "method", "intro", "related",
                "conclusion", "appendix"}


def split_sections(tex: str) -> list[dict]:
    """Split a flattened document into labelled sections."""
    # Restrict to the document body when the preamble is present.
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    m = re.search(r"\\end\{document\}", tex)
    if m:
        tex = tex[: m.start()]

    marks = list(_SEC_RE.finditer(tex))
    sections: list[dict] = []

    if marks and marks[0].start() > 0:
        sections.append(
            {"level": "pre", "title": "", "kind": "frontmatter",
             "body": tex[: marks[0].start()]}
        )

    in_appendix_at = tex.find("\\appendix")
    ancestors: dict[int, str] = {}  # depth rank -> kind of the enclosing heading
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(tex)
        title = re.sub(r"\s+", " ", mk.group("title")).strip()
        level = mk.group("cmd")
        rank = _LEVEL_RANK.get(level, 1)
        kind = classify_section(title)
        inherited = False

        # Own keywords win; otherwise take the nearest meaningful ancestor's kind.
        if kind == "other":
            for r in range(rank - 1, -1, -1):
                parent = ancestors.get(r)
                if parent in _INHERITABLE:
                    kind, inherited = parent, True
                    break

        # Anything after \appendix is appendix regardless of its own title.
        if in_appendix_at != -1 and mk.start() > in_appendix_at:
            kind, inherited = "appendix", False

        ancestors[rank] = kind
        for r in list(ancestors):  # entering a shallower heading clears deeper ones
            if r > rank:
                del ancestors[r]

        sections.append(
            {
                "level": level,
                "title": title,
                "kind": kind,
                "inherited_kind": inherited,
                "body": tex[mk.end(): end],
            }
        )

    if not sections:
        sections = [{"level": "pre", "title": "", "kind": "other", "body": tex}]
    return sections


def extract_abstract(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\abstract\s*\{(.*?)\n\n", tex, re.S)
    return m.group(1).strip() if m else ""


# --------------------------------------------------------------------- tables

_TABLE_ENVS = ("table", "table*", "tabular", "tabularx", "longtable", "threeparttable")


def _find_envs(tex: str, env: str) -> list[tuple[int, int, str]]:
    """Locate \\begin{env}...\\end{env} spans, handling nesting."""
    out = []
    begin = re.compile(r"\\begin\{" + re.escape(env) + r"\}")
    end_pat = re.compile(r"\\end\{" + re.escape(env) + r"\}")
    for bm in begin.finditer(tex):
        depth = 1
        pos = bm.end()
        while depth > 0:
            nb = begin.search(tex, pos)
            ne = end_pat.search(tex, pos)
            if ne is None:
                break
            if nb is not None and nb.start() < ne.start():
                depth += 1
                pos = nb.end()
            else:
                depth -= 1
                pos = ne.end()
        if depth == 0:
            out.append((bm.start(), pos, tex[bm.start(): pos]))
    return out


_CAPTION_RE = re.compile(r"\\caption\*?\s*\{", re.S)
_LABEL_RE = re.compile(r"\\label\s*\{([^}]*)\}")


def _balanced_after(tex: str, open_idx: int) -> str:
    """Return the contents of a {...} group whose opening brace is at open_idx."""
    depth = 0
    for i in range(open_idx, len(tex)):
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
            if depth == 0:
                return tex[open_idx + 1: i]
    return tex[open_idx + 1:]


def extract_caption(block: str) -> str:
    m = _CAPTION_RE.search(block)
    if not m:
        return ""
    body = _balanced_after(block, m.end() - 1)
    return re.sub(r"\s+", " ", body).strip()


def extract_tables(tex: str) -> list[dict]:
    """Return every table-ish environment with its caption and label.

    Outer `table` floats subsume their inner `tabular`, so we take floats first
    and then add any free-standing tabular that wasn't already covered.
    """
    found: list[dict] = []
    covered: list[tuple[int, int]] = []

    for env in ("table*", "table", "longtable", "threeparttable"):
        for s, e, block in _find_envs(tex, env):
            if any(s >= cs and e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            lab = _LABEL_RE.search(block)
            found.append(
                {
                    "env": env,
                    "start": s,
                    "caption": extract_caption(block),
                    "label": lab.group(1) if lab else "",
                    "body": block,
                }
            )

    for env in ("tabular", "tabularx"):
        for s, e, block in _find_envs(tex, env):
            if any(s >= cs and e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            found.append(
                {"env": env, "start": s, "caption": "", "label": "", "body": block}
            )

    found.sort(key=lambda d: d["start"])
    for d in found:
        d.pop("start", None)
    return found


# --------------------------------------------------------------------- driver

def parse_paper(src_dir: Path) -> dict:
    """Turn an unpacked source directory into a structured document."""
    main = find_main(src_dir)
    if main is None:
        return {"ok": False, "reason": "no_tex_file"}

    tex = flatten(main, src_dir)
    if len(tex.strip()) < 500:
        return {"ok": False, "reason": "flatten_too_short", "chars": len(tex)}

    sections = split_sections(tex)
    tables = extract_tables(tex)

    return {
        "ok": True,
        "main_tex": str(main.relative_to(src_dir)).replace("\\", "/"),
        "chars": len(tex),
        "abstract": extract_abstract(tex),
        "sections": [
            {"level": s["level"], "title": s["title"], "kind": s["kind"],
             "inherited_kind": s.get("inherited_kind", False),
             "chars": len(s["body"]), "body": s["body"]}
            for s in sections
        ],
        "section_kinds": sorted({s["kind"] for s in sections}),
        "n_tables": len(tables),
        "tables": tables,
    }
