"""Stage 8 (standalone): verify every composed sample, build the FINAL dataset.

One cheap LLM call per composed sample. The judge (arl/verify.py) audits the
finished researcher prompt against the source passages it was written from:

  - outcome_reference: does anything in the prompt state, imply, or hint at
    how the experiment turned out? (The composer was instructed not to; this
    is the semantic check that it obeyed.)
  - faithful: are all numbers, setup details, and equations exactly what the
    sources say?

Plus one mechanical check in Python: the target's method_value must not
appear in the prompt. Only samples passing everything enter
data/train_samples.jsonl -- the final dataset this pipeline exists to
produce.

Reads data/composed/ (stage 7) + data/samples.jsonl (for the source spans).
Writes one verdict per sample to data/verified/<sample_id>.json (idempotent,
resumable), then rebuilds train_samples.jsonl from all passing samples.

    python scripts/08_verify.py
    python scripts/08_verify.py --force
    python scripts/08_verify.py --limit 5      # smoke test
"""
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.openrouter import totals  # noqa: E402
from arl.util import log, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from arl.compose import assemble_prompt, compose_target  # noqa: E402
from arl.overrides import body_for, load as load_overrides  # noqa: E402
from arl.verify import method_value_leaks, run_verify_sample  # noqa: E402


def _droplist() -> set:
    """sample_ids a human reviewer has permanently excluded (unsalvageable:
    wrong/fabricated values, or malformed non-comparisons). Absent file =
    nothing dropped, so the mechanism can never silently shrink the dataset.
    The samples themselves are preserved in data/quarantine/ for analysis."""
    return set(read_json(config.DATA / "dropped_sample_ids.json", default=[]))


def _current_rows() -> dict:
    """sample_id -> full labels row for CLEAN rows in the CURRENT labels file.

    Compositions whose sample_id is no longer in samples.jsonl are zombies
    from an earlier run (re-enumeration or re-flagging) and are excluded
    from both verification and the final dataset. Rows flagged needs_review
    by a stage-7A check are excluded the same way: a label that failed a
    mechanical check is not a label, however good its composition was.
    """
    return {r.get("sample_id"): r
            for r in read_jsonl(config.LABELS_JSONL)
            if not r.get("needs_review")}


def _final_prompt(composed: dict, row: dict, prompt_ov: dict) -> str:
    """The prompt as the final dataset will carry it.

    The question footer is deterministic Python, not composer output, so it
    is reassembled from the CURRENT labels row: editing question_footer()
    restyles the whole corpus without recomposing (no API calls). Falls
    back to the stored prompt for pre-refactor compositions with no body.

    An audited body from data/prompt_overrides.json wins over the composer's.
    That body carries a HUMAN verdict, which is why it can outrank the stage-8
    judge -- every override in the file exists because the judge passed a body
    a person then read against the paper and rejected. The stored verdict still
    describes the composer's body, so the row is tagged prompt_overridden and
    the mechanical leak check re-runs on the text that actually ships.
    """
    body = body_for(composed.get("sample_id") or "", prompt_ov,
                    composed.get("body") or "")
    return assemble_prompt(body, row) if body else composed.get("prompt", "")


def work_list(limit: int | None) -> list[tuple[dict, dict]]:
    """(composed sample, its labels row) for every ok, current composition."""
    rows_by_sid = _current_rows()
    work = []
    for f in sorted(config.COMPOSED_DIR.glob("*.json")):
        d = read_json(f)
        if not d or not d.get("ok"):
            continue
        sid = d.get("sample_id") or f.stem
        if sid not in rows_by_sid:
            continue                      # zombie or flagged label
        work.append((d, rows_by_sid[sid]))
    return work[:limit] if limit else work


def build_dataset(write: bool = True) -> tuple[list[dict], int, int]:
    """FINAL dataset = composed ok AND verified passed AND still cleanly
    labelled AND not on the human droplist AND the reassembled prompt leaks
    no method value.

    The target is rebuilt from the CURRENT labels row (compose_target), not
    read from the frozen composed file, so a label-layer correction -- a
    fixed direction, a fixed value -- propagates at build time with no
    recompose and no API call. Same reasoning as the question footer.

    Returns (rows, dropped_by_leak_check, dropped_by_droplist). `write=False`
    builds the dataset without touching disk, for comparing what a rebuild
    would produce against what shipped.
    """
    current = _current_rows()
    dropped = _droplist()
    prompt_ov = load_overrides(config.PROMPT_OVERRIDES)
    rows, leaked, quarantined = [], 0, 0
    for f in sorted(config.COMPOSED_DIR.glob("*.json")):
        d = read_json(f)
        if not d or not d.get("ok"):
            continue
        sid = d.get("sample_id") or f.stem
        if sid not in current:
            continue                      # zombie or flagged label
        if sid in dropped:
            quarantined += 1
            continue                      # human-excluded, see data/quarantine/
        verdict = read_json(config.VERIFIED_DIR / f"{sid}.json")
        if not verdict or not verdict.get("passed"):
            continue
        target = compose_target(current[sid])
        prompt = _final_prompt(d, current[sid], prompt_ov)
        # The stored verdict audited the body; the footer is rebuilt here,
        # so the mechanical leak check must re-run on the final prompt.
        if method_value_leaks(prompt, target):
            leaked += 1
            continue
        meta = dict(d.get("meta", {}))
        if sid in prompt_ov:
            meta["prompt_overridden"] = True
        rows.append({
            "sample_id": sid,
            "prompt": prompt,
            "target": target,
            "meta": meta,
        })
    if write:
        write_jsonl(config.TRAIN_JSONL, rows)
    return rows, leaked, quarantined


def main() -> None:
    config.ensure_dirs()
    force = "--force" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    work = work_list(limit)
    if not work:
        raise SystemExit(
            "nothing to verify. Run scripts/07_assemble.py first "
            f"({config.COMPOSED_DIR} has no successful compositions).")

    print(f"{len(work)} composed samples to verify")
    print(f"model: {config.MODEL_SAMPLE_VERIFY}   "
          f"workers: {config.MAX_WORKERS}\n")

    def task(item):
        composed, row = item
        sid = composed.get("sample_id") or ""
        out = config.VERIFIED_DIR / f"{sid}.json"
        # A verdict is only valid for the exact BODY it audited: if the
        # composition was re-run, the old verdict must not be trusted. The
        # footer is deterministic Python rebuilt at dataset-build time, so
        # keying on the body (not the full prompt) means restyling the
        # footer never invalidates verdicts. Legacy verdicts carry
        # _prompt_sha1 over the stored prompt; both keys are honoured.
        bhash = hashlib.sha1(
            (composed.get("body") or "").encode("utf-8")).hexdigest()[:16]
        phash = hashlib.sha1(
            (composed.get("prompt") or "").encode("utf-8")).hexdigest()[:16]
        if out.exists() and not force:
            cached = read_json(out)
            if cached and (cached.get("_body_sha1") == bhash
                           or cached.get("_prompt_sha1") == phash):
                return sid, cached, "cached"
        try:
            res, _ = run_verify_sample(composed, row.get("source_spans") or [])
        except Exception as e:
            # No file written: next run retries this sample.
            return sid, {"error": f"{type(e).__name__}: {str(e)[:200]}"}, "error"
        res["sample_id"] = sid
        res["_model"] = config.MODEL_SAMPLE_VERIFY
        res["_body_sha1"] = bhash
        res["_prompt_sha1"] = phash
        write_json(out, res)
        return sid, res, "fresh"

    stats = Counter()
    failures = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(task, it) for it in work]
        for n, fut in enumerate(as_completed(futures), 1):
            sid, res, how = fut.result()
            if res.get("error"):
                stats["error"] += 1
                log(f"  {sid} ERROR: {res['error']}")
            elif res.get("passed"):
                stats["passed"] += 1
            else:
                stats["failed"] += 1
                failures.append((sid, "; ".join(res.get("reasons") or [])[:110]))
            stats[how] += 1
            if n % 5 == 0 or n == len(work):
                print(f"  {n}/{len(work)}  passed={stats['passed']} "
                      f"failed={stats['failed']} error={stats['error']}")

    final_rows, n_leaked, n_dropped = build_dataset()
    n_final = len(final_rows)

    print()
    print("=" * 68)
    print(f"  verified clean             {stats['passed']:>4}/{len(work)}")
    print(f"  failed the audit           {stats['failed']:>4}/{len(work)}")
    print(f"  errored (no file written)  {stats['error']:>4}/{len(work)}")
    print(f"  cached from earlier run    {stats['cached']:>4}")
    print("=" * 68)
    print(f"\n  FINAL DATASET: {n_final} training samples -> {config.TRAIN_JSONL}")
    if n_leaked:
        print(f"  dropped at build: {n_leaked} whose reassembled prompt "
              f"leaked a method value")
    if n_dropped:
        print(f"  excluded by droplist: {n_dropped} "
              f"(quarantined in data/quarantine/)")
    if failures:
        print("\n  audit failures (fix upstream or drop):")
        for sid, why in failures[:10]:
            print(f"    {sid}  {why}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    t = totals()
    if t["calls"]:
        print(f"\n  {t['calls']} API calls, "
              f"{t['prompt_tokens']:,}+{t['completion_tokens']:,} tokens, "
              f"cost ${t['cost']:.4f}")
    print(f"\n  verdicts -> {config.VERIFIED_DIR}")


if __name__ == "__main__":
    main()
