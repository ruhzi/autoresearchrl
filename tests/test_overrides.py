"""Human corrections must be pipeline INPUTS, not edits to its outputs.

Stage 7 rebuilds samples.jsonl from data/extracted/, and stage 8 rebuilds
train_samples.jsonl from data/composed/ -- so a correction written into either
output file is reverted by the next `python run_all.py`. An audit that has to
be re-applied after every run is not a fix, it is a chore.

These override layers make the corrections survive: stage 7 applies label
patches after normalize() and BEFORE the mechanical checks, so a correction is
verified like any other row rather than trusted; stage 8 prefers an audited
prompt body over the composer's.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl.labels import dedup  # noqa: E402
from arl.overrides import apply_label, body_for, load, overruled  # noqa: E402


def row():
    return {
        "sample_id": "2608.00001-s000",
        "component": "gating",
        "table_ref": "Table 1",
        "measurements": [
            {"benchmark": "WidgetBench", "metric_raw": "Acc",
             "baseline_value": 46.0, "method_value": 78.0},
        ],
    }


# ------------------------------------------------------------ loading

def test_a_missing_override_file_loads_as_empty(tmp_path):
    assert load(tmp_path / "nope.json") == {}


def test_an_override_file_loads_as_a_mapping(tmp_path):
    p = tmp_path / "ov.json"
    p.write_text(json.dumps({"x-s000": {"table_ref": "Table 3"}}), encoding="utf-8")
    assert load(p) == {"x-s000": {"table_ref": "Table 3"}}


# ------------------------------------------------------------ label patches

def test_a_scalar_field_is_replaced():
    out = apply_label(row(), {"table_ref": "Table 4"})
    assert out["table_ref"] == "Table 4"


def test_measurements_are_replaced_wholesale():
    """The audit's value_fix changed a baseline the extractor read from the
    wrong cell: 46.0 -> 55.4."""
    fixed = [{"benchmark": "WidgetBench", "metric_raw": "Acc",
              "baseline_value": 55.4, "method_value": 78.0}]
    out = apply_label(row(), {"measurements": fixed})
    assert out["measurements"] == fixed


def test_the_patch_records_which_fields_it_touched():
    """Provenance: a corrected row must be identifiable as corrected."""
    out = apply_label(row(), {"table_ref": "Table 4", "component": "routing"})
    assert out["overridden"] == ["component", "table_ref"]


def test_an_empty_patch_leaves_the_row_untouched():
    assert apply_label(row(), {}) == row()


def test_an_empty_patch_does_not_mark_the_row_as_overridden():
    assert "overridden" not in apply_label(row(), {})


def test_a_patch_can_add_a_field_the_extractor_never_produced():
    out = apply_label(row(), {"audit_note": "baseline read from the wrong row"})
    assert out["audit_note"] == "baseline read from the wrong row"


# ------------------------------------------------------------ overruling a check

def test_nothing_is_overruled_by_default():
    assert overruled({"measurements": []}) == set()


def test_a_directive_names_the_checks_it_overrules():
    assert overruled({"_overrule": ["target_evidence"]}) == {"target_evidence"}


def test_a_directive_is_not_applied_as_a_data_field():
    """`_overrule` tells stage 7 something; it is not part of the label."""
    out = apply_label(row(), {"_overrule": ["dedup"], "table_ref": "Table 9"})
    assert out["overridden"] == ["table_ref"]


def test_a_directive_stays_visible_on_the_row():
    out = apply_label(row(), {"_overrule": ["dedup"], "_why": "checked by hand"})
    assert out["_overrule"] == ["dedup"] and out["_why"] == "checked by hand"


def test_a_row_that_overrules_dedup_survives_containment():
    """A human kept a row the containment rule calls a duplicate slice."""
    big = {"sample_id": "p-s000", "setting": "a", "component": "x",
           "measurements": [{"metric": "acc", "baseline_value": 1.0,
                             "method_value": 2.0},
                            {"metric": "f1", "baseline_value": 3.0,
                             "method_value": 4.0}]}
    slice_of_it = {"sample_id": "p-s001", "setting": "b", "component": "y",
                   "measurements": [{"metric": "acc", "baseline_value": 1.0,
                                     "method_value": 2.0}]}
    assert [s["sample_id"] for s in dedup([big, slice_of_it])] == ["p-s000"]
    kept = dedup([big, slice_of_it], keep={"p-s001"})
    assert [s["sample_id"] for s in kept] == ["p-s000", "p-s001"]


# ------------------------------------------------------------ prompt bodies

def test_an_audited_body_wins_over_the_composed_one():
    ov = {"x-s000": "the audited body, with the leaked paragraph removed"}
    assert body_for("x-s000", ov, "the composer's body") == ov["x-s000"]


def test_a_sample_with_no_override_keeps_the_composed_body():
    assert body_for("x-s001", {"x-s000": "other"}, "composed") == "composed"


def test_an_empty_override_is_ignored_rather_than_blanking_the_prompt():
    assert body_for("x-s000", {"x-s000": ""}, "composed") == "composed"
