"""Offline self-test: parser, sanitiser, and label arithmetic on synthetic input.

No network, no API key, no cost. Run this after touching arl/latex.py,
arl/sanitize.py, arl/metrics.py, or arl/labels.py.

    python scripts/selftest.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl.labels import (  # noqa: E402
    evidence_is_verbatim, normalize, target_matches_evidence,
)
from arl.latex import parse_paper  # noqa: E402
from arl.metrics import resolve  # noqa: E402
from arl.sanitize import build_context, leak_probe_context  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


MAIN_TEX = r"""
\documentclass{article}
\begin{document}
\title{Gated Cross-Attention Helps Sometimes}

\begin{abstract}
We introduce gated cross-attention (GCA). GCA improves accuracy by 2.3 points
and achieves state-of-the-art results, reducing FLOPs by 18\%.
\end{abstract}

\section{Introduction}
Prior work~\cite{foo} struggles with long contexts. Our method reaches 91.4
accuracy on WikiText, a 12\% relative improvement over the baseline, and
outperforms all prior systems. We use 12 layers and a hidden size of 768.

\input{method}

\section{Experiments}
Table~\ref{tab:main} shows our main results. We reach 91.4\% accuracy.

\begin{table}[t]
\caption{Ablation of GCA components on WikiText-103.}
\label{tab:abl}
\begin{tabular}{lcc}
Variant & Accuracy ($\uparrow$) & PPL ($\downarrow$) \\
Full model      & 91.4 & 12.1 \\
w/o GCA         & 89.1 & 13.6 \\
w/o aux loss    & 91.3 & 12.2 \\
\end{tabular}
\end{table}

\section{Conclusion}
GCA is great.
\end{document}
"""

METHOD_TEX = r"""
\section{Method}
Gated cross-attention inserts a learned scalar gate on each cross-attention
head, letting the model suppress uninformative context. We also add an
auxiliary load-balancing loss with weight 0.01.
"""


def test_latex() -> dict:
    print("\nlatex.parse_paper")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "main.tex").write_text(MAIN_TEX, encoding="utf-8")
        (d / "method.tex").write_text(METHOD_TEX, encoding="utf-8")
        parsed = parse_paper(d)

    check("parses", parsed.get("ok") is True, str(parsed.get("reason", "")))
    kinds = {s["kind"] for s in parsed["sections"]}
    check("finds intro", "intro" in kinds, str(sorted(kinds)))
    check("resolves \\input into a method section", "method" in kinds)
    check("classifies Experiments as results", "results" in kinds)
    check("classifies Conclusion", "conclusion" in kinds)
    check("extracts abstract", "gated cross-attention" in parsed["abstract"].lower())
    check("finds the table", parsed["n_tables"] >= 1,
          f"n_tables={parsed['n_tables']}")
    caps = [t["caption"] for t in parsed["tables"] if t["caption"]]
    check("captures the caption", any("Ablation" in c for c in caps), str(caps))
    check("captures the label",
          any(t["label"] == "tab:abl" for t in parsed["tables"]))
    check("table body has the numbers",
          any("89.1" in t["body"] for t in parsed["tables"]))
    return parsed


def test_sanitize(parsed: dict) -> None:
    print("\nsanitize.build_context")
    ctx = build_context(parsed, abstract=parsed["abstract"])
    c = ctx["context"]

    check("drops the results section",
          "Table" not in c or "91.4" not in c,
          f"dropped={[d['kind'] for d in ctx['dropped_sections']]}")
    check("no bare result value 91.4 survives", "91.4" not in c)
    check("no bare result value 2.3 survives", "2.3" not in c)
    check("percentages redacted", "18" not in c and "12%" not in c)
    check("comparative claims redacted",
          "state-of-the-art" not in c.lower() and "outperform" not in c.lower())
    check("keeps method mechanism text", "gate" in c.lower())
    check("keeps architecture detail (12 layers / 768)",
          "768" in c, "architecture numbers should survive scrubbing")
    check("redaction counter non-zero", ctx["redactions"].get("total", 0) > 0,
          str(ctx["redactions"]))
    check("table environments stripped from context", "tabular" not in c)

    probe = leak_probe_context(parsed)
    check("leak probe emits an outline", probe["n_sections"] > 0)
    check("leak probe leaks no technical content",
          "gate" not in probe["probe_context"].lower())
    print("\n  --- context preview -------------------------------------")
    for line in c[:600].splitlines():
        print("  | " + line)
    print("  ---------------------------------------------------------")


def test_metrics() -> None:
    print("\nmetrics.resolve")
    cases = [
        ("Accuracy ($\\uparrow$)", "accuracy", +1),
        ("Top-1 Acc.", "accuracy", +1),
        ("PPL ($\\downarrow$)", "perplexity", -1),
        ("test perplexity", "perplexity", -1),
        ("WER (%)", "wer", -1),
        ("GFLOPs", "flops", -1),
        ("tokens/s", "throughput", +1),
        ("Macro F1", "f1", +1),
        ("mIoU", "iou", +1),
        ("FID", "fid", -1),
        ("peak memory (GB)", "memory", -1),
    ]
    for raw, want_canon, want_pol in cases:
        got = resolve(raw)
        ok = got is not None and got["canonical"] == want_canon \
            and got["polarity"] == want_pol
        check(f"{raw!r} -> {want_canon}", ok,
              "" if ok else f"got {got}")
    check("unknown metric returns None (dropped, not guessed)",
          resolve("Blorptastic Index") is None)


def test_labels() -> None:
    print()
    print("labels.normalize  (identity from stage 5, numbers from stage 6)")

    def call(metric, base, meth, unit="", conf="high", iso=True, **s5_extra):
        s5 = {"technique": "gated cross-attention", "setting": "WikiText-103",
              "metrics": [metric], "sample_id": "x-s000", **s5_extra}
        ext = {"comparison_kind": "ablation",
               "baseline_label": "w/o GCA", "method_label": "Full model",
               "measurements": [{"benchmark": "WikiText-103", "metric": metric,
                                 "baseline_value": base, "method_value": meth,
                                 "metric_unit": unit}],
               "isolates_single_technique": iso, "source_spans": ["span"],
               "table_ref": "tab:abl", "evidence": "w/o GCA & 89.1 & 13.6",
               "confidence": conf}
        return normalize(s5, ext)

    s1, why = call("Accuracy", 89.1, 91.4, unit="%")
    check("a comparison with numbers normalises", s1 is not None, why)
    check("identity taken from stage 5",
          s1 and s1["component"] == "gated cross-attention"
          and s1["metrics"] == ["Accuracy"])
    check("numbers taken from stage 6",
          s1 and s1["measurements"][0]["baseline_value"] == 89.1
          and s1["measurements"][0]["method_value"] == 91.4)
    check("delta computed in Python",
          s1 and abs(s1["measurements"][0]["delta_abs"] - 2.3) < 1e-9)
    check("registry canonicalises the metric name",
          s1 and s1["measurements"][0]["metric"] == "accuracy")

    check("NO direction label on the sample",
          s1 is not None and "direction" not in s1 and "stage5" not in s1)
    check("NO polarity field on a measurement",
          s1 is not None
          and "direction_per_paper" not in s1["measurements"][0])

    for eff in ("improves", "degrades", "no_change", "mixed", "unclear"):
        r, w = call("Accuracy", 89.1, 91.4, unit="%", effect=eff)
        check(f"legacy effect={eff!r} neither rejects nor leaks through",
              r is not None and "direction" not in r, w)

    s_unmapped, _ = call("Blorptastic Index", 1.0, 2.0)
    check("unknown metric is kept, not dropped",
          s_unmapped is not None
          and s_unmapped["measurements"][0]["metric_family"] == "unmapped")

    s_bad, why = normalize(
        {"technique": "t", "setting": "s", "sample_id": "x-s000"},
        {"measurements": [{"benchmark": "b", "metric": "Acc",
                           "baseline_value": "n/a", "method_value": "n/a"}]})
    check("non-numeric values are the only rejection",
          s_bad is None and why == "nonnumeric_value", why)

    s_low, _ = call("Accuracy", 80.0, 90.0, unit="%", conf="low")
    check("low extractor confidence flags needs_review",
          s_low and s_low["needs_review"])
    s_multi, _ = call("Accuracy", 80.0, 90.0, unit="%", iso=False)
    check("uncontrolled comparison flags needs_review",
          s_multi and s_multi["needs_review"])


def test_checks() -> None:
    print()
    print("labels checks  (what replaced the second extraction pass)")

    src = (r"egin{tabular}{lcc} Full model & 91.4 & 12.1 \ "
           r"w/o GCA & 89.1 & 13.6 \ \end{tabular}")
    check("verbatim evidence accepted",
          evidence_is_verbatim("w/o GCA & 89.1 & 13.6", src) is True)
    check("fabricated evidence rejected",
          evidence_is_verbatim("w/o GCA & 77.7 & 99.9", src) is False)
    check("whitespace differences tolerated",
          evidence_is_verbatim("w/o GCA   &   89.1   &   13.6", src) is True)
    check("too-short evidence returns None, not a pass",
          evidence_is_verbatim("89.1", src) is None)

    clean = {"evidence": ["w/o GCA & 89.1 & 91.4"], "source_spans": [],
             "measurements": [{"metric_raw": "Acc", "baseline_value": 89.1,
                               "method_value": 91.4}]}
    check("target numbers present in the evidence -> clean",
          target_matches_evidence(clean) == [])
    invented = {"evidence": ["w/o GCA & 89.1 & 91.4"], "source_spans": [],
                "measurements": [{"metric_raw": "Acc", "baseline_value": 89.1,
                                  "method_value": 77.7}]}
    check("a target number absent from the evidence is flagged",
          len(target_matches_evidence(invented)) == 1)


def main() -> None:
    print("=" * 62)
    print("autoresearchrl offline self-test")
    print("=" * 62)
    parsed = test_latex()
    if parsed.get("ok"):
        test_sanitize(parsed)
    test_metrics()
    test_labels()
    test_checks()

    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")
    print("=" * 62)


if __name__ == "__main__":
    main()
