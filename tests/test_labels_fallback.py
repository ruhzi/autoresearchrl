"""A sample is one (technique, setting) comparison carrying ALL its metrics
and their numbers. There is no direction label and nothing is rejected for
lacking one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl.labels import normalize

S5 = {"sample_id": "x-s000", "technique": "gadget", "setting": "bench",
      "metrics": ["GlorpScore", "Acc"]}

EXT = {"measurements": [
           {"benchmark": "bench-A", "metric": "GlorpScore",
            "baseline_value": 1.0, "method_value": 2.0, "metric_unit": ""},
           {"benchmark": "bench-B", "metric": "Acc",
            "baseline_value": 71.2, "method_value": 74.9,
            "metric_unit": "%"},
       ],
       "source_spans": ["span"], "evidence": "row & 1.0 & 71.2",
       "baseline_label": "base", "method_label": "gadget",
       "comparison_kind": "ablation", "table_ref": "t0",
       "confidence": "high", "isolates_single_technique": True}


def test_sample_carries_all_measurements():
    row, why = normalize(S5, dict(EXT))
    assert row is not None, why
    assert len(row["measurements"]) == 2
    assert row["measurements"][1]["baseline_value"] == 71.2


def test_no_direction_label_anywhere():
    row, _ = normalize(S5, dict(EXT))
    assert "direction" not in row
    assert "direction_detail" not in row
    assert "stage5" not in row
    assert all("direction_per_paper" not in m for m in row["measurements"])


def test_unknown_metric_never_drops():
    row, _ = normalize(S5, dict(EXT))
    assert row is not None
    assert row["measurements"][0]["metric_family"] == "unmapped"
    assert row["measurements"][1]["metric"] == "accuracy"   # registry metadata


def test_stage5_effect_is_ignored_if_present():
    """Legacy rows may still carry an effect; it must not reach the sample."""
    for eff in ("improves", "degrades", "no_change", "mixed", "unclear"):
        row, why = normalize(dict(S5, effect=eff, effect_detail="x"), dict(EXT))
        assert row is not None, f"{eff} was rejected: {why}"
        assert "direction" not in row


def test_missing_effect_is_not_a_rejection():
    """The old stage5_direction_unclear gate discarded rows whose values were
    fine. Nothing may reject a sample for lacking a judgement."""
    row, why = normalize(S5, dict(EXT))          # S5 has no effect key at all
    assert row is not None and why == ""


def test_only_nonnumeric_values_reject():
    row, why = normalize(S5, {**EXT, "measurements": [
        {"benchmark": "b", "metric": "Acc",
         "baseline_value": "n/a", "method_value": "n/a"}]})
    assert row is None and why == "nonnumeric_value"


def test_benchmarks_collected():
    row, _ = normalize(S5, dict(EXT))
    assert row["benchmarks"] == ["bench-A", "bench-B"]


def test_grid_papers_not_flagged_when_baseline_labels_differ():
    from arl.labels import internal_consistency
    # same benchmark+metric+setting, DIFFERENT baseline rows (grid paper):
    # legitimately different baselines, must NOT be flagged
    a = {"setting": "catering", "baseline_label": "RE policy",
         "measurements": [{"benchmark": "b", "metric": "utility",
                           "baseline_value": 10.0, "method_value": 12.0}],
         "needs_review": False}
    b = {"setting": "catering", "baseline_label": "RCC policy",
         "measurements": [{"benchmark": "b", "metric": "utility",
                           "baseline_value": 55.0, "method_value": 57.0}],
         "needs_review": False}
    out = internal_consistency([a, b])
    assert not out[0]["baseline_inconsistent"]
    assert not out[1]["baseline_inconsistent"]
    # same labeled row disagreeing on the value: a real contradiction
    c = dict(a, measurements=[{"benchmark": "b", "metric": "utility",
                               "baseline_value": 99.0,
                               "method_value": 12.0}], needs_review=False)
    out = internal_consistency([dict(a, needs_review=False), c])
    assert out[0]["baseline_inconsistent"] and out[1]["baseline_inconsistent"]


def test_dedup_catches_paraphrased_duplicates_by_values():
    from arl.labels import dedup
    a = {"setting": "catering domain", "component": "RCC vs RE counterparty",
         "measurements": [{"benchmark": "b", "metric": "utility",
                           "baseline_value": 10.0, "method_value": 12.0}]}
    b = {"setting": "synthetic catering benchmark",
         "component": "LLM agent versus Rational Exploiter baseline",
         "measurements": [{"benchmark": "b", "metric": "utility",
                           "baseline_value": 10.0, "method_value": 12.0}]}
    c = {"setting": "catering domain", "component": "something else",
         "measurements": [{"benchmark": "b", "metric": "utility",
                           "baseline_value": 3.0, "method_value": 4.0}]}
    out = dedup([a, b, c])
    assert len(out) == 2                    # b collapsed into a by values
    assert out[0] is a and out[1] is c


def _meas(*tuples):
    return [{"benchmark": b, "metric": m, "baseline_value": bv,
             "method_value": mv} for b, m, bv, mv in tuples]


def test_dedup_drops_strict_subset_slice_keeps_larger():
    from arl.labels import dedup
    small = {"setting": "config stack, colorization",
             "component": "encoder fine-tuning added on top",
             "measurements": _meas(("colorization", "FID", 47.21, 45.97),
                                   ("colorization", "LPIPS", 0.3845, 0.3654))}
    large = {"setting": "per-task ablation",
             "component": "Encoder fine-tuning",
             "measurements": _meas(("Colorization", "FID", 47.21, 45.97),
                                   ("Colorization", "LPIPS", 0.3845, 0.3654),
                                   ("Denoising", "FID", 30.0, 28.0),
                                   ("Denoising", "LPIPS", 0.4, 0.35))}
    # subset listed FIRST: keep-larger must win regardless of order
    out = dedup([small, large])
    assert out == [large]


def test_dedup_keeps_partial_overlap():
    from arl.labels import dedup
    a = {"setting": "s1", "component": "c1",
         "measurements": _meas(("b", "FID", 56.27, 47.21),
                               ("b", "PSNR", 23.11, 21.88))}   # PSNR novel
    b = {"setting": "s2", "component": "c2",
         "measurements": _meas(("b", "FID", 56.27, 47.21),
                               ("b", "LPIPS", 0.36, 0.38),
                               ("b", "SSIM", 0.7, 0.72))}
    out = dedup([a, b])
    assert len(out) == 2                    # 50% overlap is not containment


def test_no_numeric_measurements_drops():
    row, why = normalize(S5, dict(EXT, measurements=[
        {"benchmark": "b", "metric": "Acc", "baseline_value": "n/a",
         "method_value": None,
         "metric_unit": "", "direction_per_paper": "unclear"}]))
    assert row is None
    assert why == "nonnumeric_value"
