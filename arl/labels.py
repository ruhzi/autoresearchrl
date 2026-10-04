"""Turn extracted numeric tuples into verifiable samples.

A sample is a comparison plus its numbers: identity from stage 5, values from
stage 6, and provenance on every row so any figure traces back to a specific
table cell.

There is deliberately no direction/effect label. Whether a comparison helped
is a function of these values and the paper's own polarity for each metric --
derivable afterwards, mechanically, with a stated materiality threshold. Baking
a model's guess at it into the sample made it the pipeline's dominant error
source while the values themselves were essentially always right.
"""
from __future__ import annotations

import re

import config
from arl.metrics import resolve

# Below this relative change we call it neutral rather than a tiny win/loss --
# differences this small are usually inside seed variance.
NEUTRAL_REL = 0.002


def _toks(s: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rel_close(x: float, y: float, tol: float) -> bool:
    denom = max(abs(x), abs(y), 1e-9)
    return abs(x - y) / denom <= tol


# ------------------------------------------------------------------ normalising

def normalize(s5: dict, ext: dict) -> tuple[dict | None, str]:
    """Combine a stage-5 sample with its stage-6 extraction into one label.

    Identity (technique / setting / metric) comes from stage 5; numbers and
    verbatim detail from stage 6. The two are matched by sample_id upstream, so
    this function never has to guess which extraction belongs to which sample.
    """
    # No direction label. The sample IS the comparison plus its numbers; a
    # verdict over those numbers is derived later if it is wanted, from the
    # values and the paper's own metric polarity. Nothing here rejects a
    # sample for lacking one -- that rejection used to discard rows whose
    # values were fine.
    raw_meas = ext.get("measurements") or []
    measurements = []
    for mm in raw_meas:
        try:
            b = float(mm["baseline_value"])
            m = float(mm["method_value"])
        except (KeyError, TypeError, ValueError):
            continue
        met = resolve(mm.get("metric", ""))
        measurements.append({
            "benchmark": mm.get("benchmark", ""),
            "metric_raw": mm.get("metric", ""),
            "metric": met["canonical"] if met else
                " ".join(str(mm.get("metric", "")).lower().split()),
            "metric_family": met["family"] if met else "unmapped",
            "baseline_value": b,
            "method_value": m,
            "delta_abs": m - b,
            "delta_rel": (m - b) / abs(b) if b != 0 else None,
            "metric_unit": (mm.get("metric_unit") or "").strip(),
        })
    if not measurements:
        return None, "nonnumeric_value"

    sample = {
        # ---- identity, from stage 5
        "sample_id": s5.get("sample_id", ""),
        "component": s5.get("technique", ""),
        "setting": s5.get("setting", ""),
        "benchmarks": sorted({m["benchmark"] for m in measurements if
                              m["benchmark"]}),
        "metrics": sorted({m["metric_raw"] for m in measurements}),
        # ---- source material, verbatim from the paper
        "source_spans": list(ext.get("source_spans") or []),
        # ---- detail, from stage 6
        "comparison_kind": ext.get("comparison_kind", "other"),
        "baseline_label": ext.get("baseline_label", ""),
        "method_label": ext.get("method_label", ""),
        "measurements": measurements,
        # ---- provenance / quality
        "table_ref": ext.get("table_ref", ""),
        "evidence": ext.get("evidence", ""),
        "extractor_confidence": ext.get("confidence", "medium"),
        "isolates_single_technique": bool(
            ext.get("isolates_single_technique", True)),
        "needs_review": (
            ext.get("confidence") == "low"
            or not ext.get("isolates_single_technique", True)
        ),
    }
    return sample, ""


# ------------------------------------------------------------------- checking
#
# The two-pass fuzzy matcher that used to live here is gone. With one stage-6
# call per stage-5 sample, extractions key to stage-5 rows by sample_id exactly,
# and four cheaper checks replace a second model pass:
#
#   1. evidence_is_verbatim   -- mechanical: did the quoted row really occur in
#                                the source we sent? catches fabrication
#   2. internal_consistency   -- mechanical: samples sharing metric+setting must
#                                agree on the baseline value
#   3. target_matches_evidence -- mechanical: the target's own numbers must be
#                                visible in the quoted evidence, and never only
#                                as the numerator of an "X/<baseline>" cell
#                                (the resolved-count-read-as-result signature)
#
# (The former checks 3/4 -- registry-polarity conflict and judged-vs-computed
# direction agreement -- went with the direction label itself. A sample now
# carries a comparison and its numbers; there is no verdict to cross-check.)


def _norm_ws(t: str) -> str:
    return " ".join((t or "").split())


def evidence_is_verbatim(evidence: str, source: str,
                         min_len: int = 12) -> bool | None:
    """Did the quoted evidence actually occur in the text we sent the model?

    Whitespace-normalised substring test. Returns None when the evidence is too
    short to be meaningful either way, so a trivial quote is not scored as a pass.
    """
    ev = _norm_ws(evidence)
    if len(ev) < min_len:
        return None
    return ev in _norm_ws(source)


_TNUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _float_tokens(text: str) -> set[float]:
    out = set()
    for m in _TNUM_RE.findall(text or ""):
        try:
            out.add(float(m))
        except ValueError:
            pass
    return out


def _spellings(v: float) -> set[str]:
    """String spellings a value might take in LaTeX source."""
    out = {f"{v:g}"}
    if float(v).is_integer():
        out.add(str(int(v)))
        out.add(f"{v:.1f}")
    return out


def target_matches_evidence(row: dict) -> list[str]:
    """Mechanical: audit the target's numbers against the quoted evidence.

    Two failure modes, both zero-cost to detect and both real extraction
    errors rather than judgement calls:

    - a baseline/method value that occurs NOWHERE in evidence+spans was not
      read off this paper's table at all
    - a method value whose every occurrence is the numerator of a two-part
      fraction "X/Y" with Y equal to one of this sample's baselines was read
      out of a resolved-count/ratio cell ("26/33 resolved"), not a
      post-intervention result cell. Slash-chains like "44.2/45.8/45.7"
      (multi-config cells) are exempt: chain members are not fractions.

    Returns a list of failure descriptions; empty = clean. Rows with no
    evidence text are uncheckable and pass vacuously (evidence presence is
    check 1's job).
    """
    evs = row.get("evidence") or []
    if isinstance(evs, str):
        evs = [evs]
    ev = ("\n".join(evs)).replace("−", "-")
    spans = ("\n".join(row.get("source_spans") or [])).replace("−", "-")
    both = ev + "\n" + spans
    if not both.strip():
        return []

    allnum = _float_tokens(both)
    baselines = {float(mm["baseline_value"])
                 for mm in row.get("measurements") or []
                 if mm.get("baseline_value") is not None}
    fails: list[str] = []
    for mm in row.get("measurements") or []:
        for key in ("baseline_value", "method_value"):
            v = mm.get(key)
            if v is not None and float(v) not in allnum:
                fails.append(f"{key} {float(v):g} "
                             f"({mm.get('metric_raw', '?')}) occurs nowhere "
                             f"in the evidence or source spans")
        v = mm.get("method_value")
        if v is None or float(v) not in allnum:
            continue
        occs: list[bool] = []
        for s in _spellings(float(v)):
            for mo in re.finditer(r"(?<![\d.])" + re.escape(s) + r"(?![\d.])",
                                  both):
                head = both[max(0, mo.start() - 4):mo.start()]
                if re.search(r"/[\s{]*$", head):
                    occs.append(False)   # itself a denominator/chain member
                    continue
                tail = both[mo.end():mo.end() + 16]
                dm = re.match(r"[\s}]*/[\s{]*(?:\\[a-zA-Z]+\{?)?"
                              r"([\d.]+)(?![\d.]*\s*/)", tail)
                occs.append(bool(dm) and float(dm.group(1)) in baselines)
        if occs and all(occs):
            fails.append(f"method_value {float(v):g} "
                         f"({mm.get('metric_raw', '?')}) only occurs in the "
                         f"evidence as the numerator of X/<baseline> -- "
                         f"likely a resolved-count/ratio cell, not the "
                         f"post-intervention value")
    return fails


def dedup(samples: list[dict], keep: set | None = None) -> list[dict]:
    """Drop duplicate rows within one paper.

    `keep` holds sample_ids a human decided are not duplicates (via
    `_overrule: ["dedup"]` in data/label_overrides.json). They survive every
    rule below but still register, so other rows continue to dedupe against
    them.

    Three rules, any match means duplicate:
    - wording: same normalized (component, setting)
    - values:  identical measurement grid. Stage-5 batch fragmentation
      re-lists the same comparison in different words; the numbers cannot
      be paraphrased.
    - containment: A's grid strictly inside B's larger grid means A is a
      duplicate SLICE of B (papers repeat one comparison's cells across a
      per-task table and a config-stack table); keep B regardless of which
      was enumerated first. Benchmark names are excluded from the tuple
      (their casing varies between tables); full containment of the
      (metric, baseline, method) set is discriminating enough.
    """
    def grid(s: dict) -> frozenset:
        return frozenset(
            (" ".join(str(mm["metric"]).lower().split()),
             round(mm["baseline_value"], 6), round(mm["method_value"], 6))
            for mm in s.get("measurements") or [])

    grids = [grid(s) for s in samples]

    seen: set[tuple] = set()
    seen_values: set[frozenset] = set()
    out = []
    for i, s in enumerate(samples):
        key = (
            " ".join(sorted(_toks(s["setting"]))),
            " ".join(sorted(_toks(s["component"]))),
        )
        if not (keep and s.get("sample_id") in keep):
            if key in seen or (grids[i] and grids[i] in seen_values):
                continue
            if grids[i] and any(grids[i] < g for g in grids):
                continue                    # proper subset of a larger grid
        seen.add(key)
        if grids[i]:
            seen_values.add(grids[i])
        out.append(s)
    return out


def internal_consistency(samples: list[dict]) -> list[dict]:
    """Flag rows whose baseline value contradicts another row's for the same
    labeled row.

    The invariant is per LABELED ROW: the same (benchmark, metric, setting,
    baseline_label) must report one baseline number -- if two extractions
    disagree there, one of them read the wrong cell. The label is part of
    the key on purpose: papers with grid-style results (role x counterparty
    x model) legitimately carry many different baselines for the same
    metric+setting, distinguished only by which row is the baseline.
    """
    by_key: dict[tuple, set[float]] = {}
    for s in samples:
        skey = " ".join(sorted(_toks(s["setting"])))
        blabel = " ".join(sorted(_toks(s.get("baseline_label", ""))))
        for mm in s.get("measurements") or []:
            by_key.setdefault(
                (mm.get("benchmark", ""), mm["metric"], skey, blabel),
                set()).add(round(mm["baseline_value"], 6))

    for s in samples:
        skey = " ".join(sorted(_toks(s["setting"])))
        blabel = " ".join(sorted(_toks(s.get("baseline_label", ""))))
        bad = any(
            len(by_key.get((mm.get("benchmark", ""), mm["metric"], skey,
                            blabel), set())) > 1
            for mm in s.get("measurements") or [])
        s["baseline_inconsistent"] = bad
        if bad:
            s["needs_review"] = True
    return samples
