"""Stage 5's printers must speak the current sample schema.

The no-direction refactor rewrote arl/candidates.summarize() to stop emitting
`direction_distribution` and `tradeoff_count` -- a sample is a comparison, not
a verdict. scripts/05_enumerate.py was not updated with it, so the run crashed
on the summary AFTER paying for the enumeration. These tests pin the printers
against the schema the enumerator actually produces today.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arl.candidates import summarize  # noqa: E402


def _stage5():
    """scripts/05_enumerate.py -- not importable by name (leading digit)."""
    spec = importlib.util.spec_from_file_location(
        "stage5", ROOT / "scripts" / "05_enumerate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample(i, technique="gating", metrics=("Acc", "F1")):
    """One candidate exactly as ENUMERATE_SCHEMA defines it: no direction,
    no effect, no derived block."""
    return {
        "sample_id": f"2608.00001-s{i:03d}",
        "description": "w/o gating vs full model",
        "technique": technique,
        "setting": "ViT-B, 100 epochs",
        "benchmarks": ["WidgetBench"],
        "metrics": list(metrics),
    }


PER_PAPER = {
    "2608.00001": {"samples": [sample(0), sample(1, "routing")], "notes": ""},
    "2608.00002": {"samples": [sample(0, "pruning", ("PSNR",))],
                   "notes": "tables only in the appendix"},
}


def test_summary_prints_for_the_current_schema(capsys):
    _stage5().print_summary(summarize(PER_PAPER))
    out = capsys.readouterr().out
    assert "papers enumerated        2" in out
    assert "TOTAL candidate samples  3" in out


def test_summary_does_not_report_a_direction_balance(capsys):
    """Stage 5 no longer produces a direction, so it must not report one --
    a printed 'DIRECTION BALANCE' block was reading keys that are gone."""
    _stage5().print_summary(summarize(PER_PAPER))
    assert "DIRECTION" not in capsys.readouterr().out.upper()


def test_summary_survives_an_empty_corpus(capsys):
    _stage5().print_summary(summarize({}))
    assert "papers enumerated        0" in capsys.readouterr().out


def test_paper_listing_shows_the_fields_a_sample_still_has(capsys):
    _stage5().print_paper("2608.00001", "Gated Widgets", PER_PAPER["2608.00001"])
    out = capsys.readouterr().out
    assert "2 candidate samples" in out
    assert "gating" in out and "routing" in out
    assert "Acc, F1" in out
    assert "ViT-B, 100 epochs" in out


def test_paper_listing_has_no_placeholder_direction_column(capsys):
    """A '?' in the listing means the printer asked for a field the schema
    dropped."""
    _stage5().print_paper("2608.00001", "Gated Widgets", PER_PAPER["2608.00001"])
    assert "?" not in capsys.readouterr().out
