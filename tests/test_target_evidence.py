"""Stage-7A check 3: the target's numbers must be backed by the quoted
evidence. Pure Python, no LLM. The motivating bug is real (2608.08037-s001):
a "26/33 resolved" table cell read as method_value=26, when the
post-intervention failure count is 7."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl.labels import target_matches_evidence


def row(measurements, evidence, spans=()):
    return {"measurements": measurements, "evidence": evidence,
            "source_spans": list(spans)}


def m(baseline, method, metric="Acc"):
    return {"baseline_value": baseline, "method_value": method,
            "metric_raw": metric, "metric": metric}


def test_clean_row_passes():
    r = row([m(71.2, 74.9)], "baseline & 71.2 \\\\ ours & 74.9 \\\\")
    assert target_matches_evidence(r) == []


def test_value_absent_from_evidence_fails():
    r = row([m(71.2, 74.9)], "baseline & 71.2 \\\\ ours & 75.0 \\\\")
    fails = target_matches_evidence(r)
    assert len(fails) == 1 and "74.9" in fails[0]


def test_resolved_count_fraction_fails():
    # the s001 signature: method value only exists as X/<baseline>
    ev = ("subtotal & \\textbf{33} & \\textbf{91.7\\%} & \\textbf{26/33} \\\\")
    r = row([m(33.0, 26.0, "failures (count)")], ev)
    fails = target_matches_evidence(r)
    assert len(fails) == 1 and "numerator" in fails[0]


def test_fraction_in_braces_with_span_restatement_still_fails():
    # evidence has {26/33}; a span restates it as \textbf{26}/33 -- both are
    # fraction contexts, neither is a standalone result
    ev = "subtotal & \\textbf{33} & \\textbf{26/33} \\\\"
    sp = ["this intervention resolves \\textbf{26}/33 knowledge failures"]
    r = row([m(33.0, 26.0)], ev, sp)
    assert target_matches_evidence(r)


def test_slash_chain_cells_are_exempt():
    # multi-config cells "a/b/c" are pairs of results, not fractions
    ev = ("Key cosine & 44.2/45.8/45.7 \\\\ baseline & 44.2 & 45.8 & 45.7 \\\\")
    r = row([m(44.2, 45.8, "mIoU")], ev)
    assert target_matches_evidence(r) == []


def test_standalone_occurrence_elsewhere_clears_the_fraction():
    ev = "ours reaches 26 (26/33 of failures resolved), baseline 33"
    r = row([m(33.0, 26.0)], ev)
    assert target_matches_evidence(r) == []


def test_no_evidence_passes_vacuously():
    r = row([m(1.0, 2.0)], "", [])
    assert target_matches_evidence(r) == []


def test_unicode_minus_normalised():
    r = row([m(-0.1926, -0.15228, "dCE")],
            "locked \u22120.1926 nats, fresh \u22120.15228 nats")
    assert target_matches_evidence(r) == []
