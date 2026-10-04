"""Stage 7 (standalone): assemble training samples from stage 6's verbatim
material. Two halves of one job:

  A. TARGET (free): join stage 5 + stage 6. Identity from stage 5,
     judgement (improves/degrades, defined as BETTER/WORSE in its prompt);
     the numbers are stage 6's verbatim values; magnitudes are computed from
     those numbers. Two mechanical checks run (verbatim spans, baseline
     consistency). Writes data/samples.jsonl + data/rejects.json. Re-runs
     every time, it costs nothing.

  B. PROMPT (LLM, one call per row): a composer model rewrites the verbatim
     spans into a neutral first-person researcher message ("I've been
     brainstorming...; my exact setup...; my baseline gets X") and Python
     appends the one scored question: predict the value for each metric.
     The composer never sees the answer, and every rewrite must pass the
     mechanical fidelity gate in arl/compose.py (no invented numbers,
     baseline present, method value absent, LaTeX copied exactly, no outcome
     copied exactly). Outcome references are excluded by instruction in the
     composer's system prompt -- stage 8 audits that semantically. Failing
     rewrites are retried with reasons fed back. Writes one file per sample
     to data/composed/ (idempotent, resumable); stage 8 builds the final
     dataset from these.

By default only rows passing all checks (needs_review false) are composed;
--include-flagged composes the rest too, marked as such.

    python scripts/07_assemble.py
    python scripts/07_assemble.py --labels-only     # part A only, no API calls
    python scripts/07_assemble.py --strict          # drop rows failing any check
    python scripts/07_assemble.py --limit 5         # compose only N rows (smoke)
    python scripts/07_assemble.py --force           # recompose cached rows
    python scripts/07_assemble.py --include-flagged
"""
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.compose import run_compose_sample  # noqa: E402
from arl.labels import (  # noqa: E402
    dedup, evidence_is_verbatim, internal_consistency, normalize,
    target_matches_evidence,
)
from arl.openrouter import totals  # noqa: E402
from arl.overrides import (  # noqa: E402
    apply_label, load as load_overrides, overruled,
)
from arl.util import log, read_json, write_json, write_jsonl  # noqa: E402


def paper_source(parsed: dict) -> str:
    """Everything stage 6 sliced from -- what every verbatim span is checked
    against. A span that does not occur here literally was not sliced from
    this paper, so it gets flagged rather than trusted."""
    parts = [(t.get("body") or "") for t in parsed.get("tables", [])]
    parts += [(sec.get("body") or "") for sec in parsed.get("sections", [])]
    parts.append(parsed.get("abstract") or "")
    return "\n".join(parts)


# ------------------------------------------------------- part A: the target

def build_labels(strict: bool, write: bool = True) -> list[dict]:
    meta = {p["base_id"]: p for p in read_json(config.SAMPLE_JSON, default=[])}
    all_samples, rejects = [], []
    drops = Counter()
    checks = Counter()
    # Human corrections, replayed on every rebuild. Applied before the
    # mechanical checks below, so a correction is verified rather than
    # trusted -- see arl/overrides.py.
    label_ov = load_overrides(config.LABEL_OVERRIDES)
    ov_used = set()

    for f in sorted(config.CANDIDATES_DIR.glob("*.json")):
        bid = f.stem
        cand = read_json(f)
        parsed = read_json(config.PARSED_DIR / f"{bid}.json")
        if not cand or not parsed:
            continue
        src = paper_source(parsed)

        paper_rows = []
        for s5 in cand.get("samples") or []:
            sid = s5.get("sample_id", "")
            ext_res = read_json(config.EXTRACT_DIR / f"{sid}.json")

            if ext_res is None:
                drops["no_extraction_yet"] += 1
                continue
            if ext_res.get("error"):
                drops["extraction_error"] += 1
                continue
            if not ext_res.get("found"):
                drops["not_found_in_paper"] += 1
                rejects.append({"sample_id": sid, "reason": "not_found",
                                "detail": ext_res.get("not_found_reason", ""),
                                "stage5": s5})
                continue

            row, why = normalize(s5, ext_res["extraction"])
            if row is None:
                drops[why.split(":")[0]] += 1
                rejects.append({"sample_id": sid, "reason": why, "stage5": s5})
                continue

            if sid in label_ov:
                row = apply_label(row, label_ov[sid])
                ov_used.add(sid)
                checks["label_override_applied"] += 1

            # ---- check 1: every sliced span really came from the paper
            evs = row.get("evidence") or []
            if isinstance(evs, str):
                evs = [evs]
            ev_oks = [evidence_is_verbatim(e, src) for e in evs]
            ev_ok = (False if any(v is False for v in ev_oks)
                     else (None if all(v is None for v in ev_oks) else True))
            row["evidence_is_verbatim"] = ev_ok
            if ev_ok is False:
                checks["not_verbatim:evidence"] += 1
                row["needs_review"] = True
            elif ev_ok is None:
                checks["evidence_too_short_to_check"] += 1

            spans = row.get("source_spans") or []
            bad = [i for i, sp in enumerate(spans)
                   if evidence_is_verbatim(sp, src) is False]
            row["nonverbatim_span_indices"] = bad
            if bad:
                checks["not_verbatim:source_spans"] += len(bad)
                row["needs_review"] = True
            if not spans:
                # nothing to compose the prompt from -- a real problem,
                # unlike any single category being absent
                checks["no_source_spans"] += 1
                row["needs_review"] = True

            # ---- check 3: the target's numbers must be visible in the
            #      evidence, and never only as an X/<baseline> numerator
            tv_fails = target_matches_evidence(row)
            row["target_evidence_issues"] = tv_fails
            if tv_fails:
                checks["target_not_backed_by_evidence"] += len(tv_fails)
                # A corrected value read from the whole paper can be absent
                # from the spans stage 6 happened to slice. The complaint stays
                # on the row; only the exclusion is waived, and only where the
                # override says so explicitly.
                if "target_evidence" in overruled(label_ov.get(sid)):
                    checks["target_evidence_overruled"] += 1
                else:
                    row["needs_review"] = True

            if not row["isolates_single_technique"]:
                checks["not_single_technique"] += 1

            row["base_id"] = bid
            p = meta.get(bid, {})
            row["arxiv_id"] = p.get("id", bid)
            row["title"] = p.get("title", "")
            row["primary_cat"] = p.get("primary_cat")
            row["published"] = p.get("published")
            paper_rows.append(row)

        before = len(paper_rows)
        keep = {s for s in label_ov
                if "dedup" in overruled(label_ov.get(s))}
        paper_rows = dedup(paper_rows, keep=keep)
        if before != len(paper_rows):
            drops["duplicate_row"] += before - len(paper_rows)
        # ---- check 2: baselines must agree within a paper
        paper_rows = internal_consistency(paper_rows)
        all_samples.extend(paper_rows)

    inconsistent = sum(1 for s in all_samples if s.get("baseline_inconsistent"))
    if inconsistent:
        checks["baseline_inconsistent"] = inconsistent

    if strict:
        kept = [s for s in all_samples if not s["needs_review"]
                and not s.get("nonverbatim_span_indices")
                and s.get("evidence_is_verbatim") is not False]
        drops["failed_a_check_strict"] = len(all_samples) - len(kept)
        all_samples = kept

    if write:
        n = write_jsonl(config.LABELS_JSONL, all_samples)
        write_json(config.DATA / "rejects.json", rejects)
    else:
        n = len(all_samples)

    fams = Counter(mm.get("metric_family", "unmapped")
                   for s in all_samples
                   for mm in s.get("measurements") or [])
    review = sum(1 for s in all_samples if s["needs_review"])

    print("=" * 68)
    print(f"  LABELLED SAMPLES            {n}")
    print(f"  flagged needs_review        {review}")
    print("=" * 68)
    spread = Counter()
    for s in all_samples:
        ds = [abs(mm["delta_rel"]) for mm in (s.get("measurements") or [])
              if mm.get("delta_rel") is not None]
        mx = max(ds) if ds else None
        spread["unknown" if mx is None else
               "<2%" if mx < 0.02 else "2-5%" if mx < 0.05 else
               "5-20%" if mx < 0.20 else ">20%"] += 1
    print("  effect size (largest relative move per sample):")
    for k, v in spread.most_common():
        print(f"    {k:<10} {v:>5}   {100.0 * v / max(n, 1):>5.1f}%")
    print(f"\n  metric families: {dict(fams)}")
    print()
    print("  CHECKS  (each is a free second opinion on a label):")
    for k, v in checks.most_common():
        print(f"    {v:>5}  {k}")
    if not checks:
        print("    (nothing flagged)")
    stale = sorted(set(label_ov) - ov_used)
    if stale:
        # A correction that matches no row is a correction that silently does
        # nothing -- the exact failure the override layer exists to prevent.
        log(f"\n  WARNING: {len(stale)} label overrides matched no sample "
            f"(re-enumeration renumbered them?): {', '.join(stale[:6])}"
            f"{' ...' if len(stale) > 6 else ''}")
    print()
    print("  drops:")
    for k, v in drops.most_common():
        print(f"    {v:>5}  {k}")
    if not drops:
        print("    (none)")
    print(f"\n  -> {config.LABELS_JSONL}")
    print(f"  -> {config.DATA / 'rejects.json'}")
    return all_samples


# ------------------------------------------------------- part B: the prompt

def compose_all(labelled: list[dict], force: bool, include_flagged: bool,
                limit: int | None) -> None:
    work = [r for r in labelled
            if include_flagged or not r.get("needs_review")]
    skipped_flagged = len(labelled) - len(work)
    if limit:
        work = work[:limit]

    if not work:
        print("\nno rows to compose "
              "(everything is flagged, or there are no labels yet)")
        return

    print(f"\ncomposing {len(work)} rows "
          f"({skipped_flagged} flagged rows excluded"
          f"{'' if include_flagged else ', pass --include-flagged to add'})")
    print(f"model: {config.MODEL_COMPOSE}   workers: {config.MAX_WORKERS}\n")

    def task(row):
        sid = row.get("sample_id") or ""
        out = config.COMPOSED_DIR / f"{sid}.json"
        # Positional ids can be reused after re-enumeration; require the
        # identity to match before trusting a cached composition.
        ident = {"technique": row.get("component", ""),
                 "setting": row.get("setting", "")}
        audit_feedback = ""
        if out.exists() and not force:
            cached = read_json(out)
            if cached and cached.get("_identity") == ident:
                # One recompose attempt when the stage-8 audit rejected the
                # cached composition: feed the audit reasons back. The
                # prompt hash change invalidates the old verdict downstream.
                verdict = read_json(config.VERIFIED_DIR / f"{sid}.json")
                if (verdict and verdict.get("passed") is False
                        and cached.get("ok")
                        and not cached.get("_audit_retry")):
                    audit_feedback = "\n".join(verdict.get("reasons") or [])
                else:
                    return sid, cached, "cached"
        try:
            res, _ = run_compose_sample(row, audit_feedback=audit_feedback)
        except Exception as e:
            # No file written: next run retries this sample.
            return sid, {"error": f"{type(e).__name__}: {str(e)[:200]}"}, "error"
        res["sample_id"] = sid
        res["_model"] = config.MODEL_COMPOSE
        res["_identity"] = ident
        if audit_feedback:
            res["_audit_retry"] = 1
        res["meta"] = {
            "base_id": row.get("base_id", ""),
            "title": row.get("title", ""),
            "component": row.get("component", ""),
            "setting": row.get("setting", ""),
            "flagged": bool(row.get("needs_review")),
        }
        write_json(out, res)
        return sid, res, "fresh"

    stats = Counter()
    fail_reasons = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(task, r) for r in work]
        for n, fut in enumerate(as_completed(futures), 1):
            sid, res, how = fut.result()
            if res.get("error"):
                stats["error"] += 1
                log(f"  {sid} ERROR: {res['error']}")
            elif res.get("ok"):
                stats["ok"] += 1
            else:
                stats["gate_failed"] += 1
                fail_reasons.append(
                    (sid, "; ".join(res.get("failures") or [])[:100]))
            stats[how] += 1
            if n % 5 == 0 or n == len(work):
                print(f"  {n}/{len(work)}  ok={stats['ok']} "
                      f"gate_failed={stats['gate_failed']} "
                      f"error={stats['error']}")

    print()
    print("=" * 68)
    print(f"  composed (passed gate)     {stats['ok']:>4}/{len(work)}")
    print(f"  failed the fidelity gate   {stats['gate_failed']:>4}/{len(work)}")
    print(f"  errored (no file written)  {stats['error']:>4}/{len(work)}")
    print(f"  cached from earlier run    {stats['cached']:>4}")
    print("=" * 68)
    if fail_reasons:
        print("\n  gate failures (composer could not stay faithful):")
        for sid, why in fail_reasons[:10]:
            print(f"    {sid}  {why}")
        if len(fail_reasons) > 10:
            print(f"    ... and {len(fail_reasons) - 10} more")

    t = totals()
    if t["calls"]:
        print(f"\n  {t['calls']} API calls, "
              f"{t['prompt_tokens']:,}+{t['completion_tokens']:,} tokens, "
              f"cost ${t['cost']:.4f}")
    print(f"\n  per-sample -> {config.COMPOSED_DIR}")


def main() -> None:
    config.ensure_dirs()
    strict = "--strict" in sys.argv
    labels_only = "--labels-only" in sys.argv
    force = "--force" in sys.argv
    include_flagged = "--include-flagged" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    labelled = build_labels(strict)

    if labels_only:
        print("\n--labels-only: skipping prompt composition (no API calls)")
    elif labelled:
        compose_all(labelled, force, include_flagged, limit)

    print("  next: python scripts/08_verify.py   (audit + final dataset)")


if __name__ == "__main__":
    main()
