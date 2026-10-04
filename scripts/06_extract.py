"""Stage 6 (standalone): retrieve verbatim material, ONE agent run per sample.

Stage 5 said a comparison exists and which way it went. This stage is handed one
of those at a time; a retrieval agent locates it in the paper with grep/read
tools and submits anchor POINTERS. Python slices the exact source bytes those
pointers bracket -- the prose spans, the evidence row, and the two numbers
(which must occur literally inside the sliced row). Nothing in the payload is
model-typed, so text and numbers are verbatim by construction.

One sample per run rather than a batch, because a model handed forty rows to
fill returns forty rows whether or not they exist. Asked about a single
comparison, `not_found` is a natural answer.

Stage 6 is NOT told where stage 5 found the comparison, and NOT told which
direction stage 5 judged. It re-locates the comparison itself, so:
  - a vague or invented stage-5 row surfaces as not_found instead of being trusted
  - the arithmetic direction from these numbers is an independent check on stage 5

Reads data/candidates/ (stage 5) + data/parsed/. Writes one file per sample to
data/extracted/<sample_id>.json. Idempotent per sample -- already-extracted
samples are skipped unless --force, so an interrupted run resumes for free.

    python scripts/06_extract.py
    python scripts/06_extract.py --force
    python scripts/06_extract.py --paper 2608.05811     # one paper only
    python scripts/06_extract.py --limit 10             # first N samples, for a smoke test
"""
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.retrieve import run_retrieve_sample  # noqa: E402
from arl.openrouter import totals  # noqa: E402
from arl.util import log, read_json, write_json  # noqa: E402


def work_list(only_paper: str | None) -> list[tuple[dict, dict, dict]]:
    """(paper, parsed, stage-5 sample) for every enumerated sample."""
    sample_meta = read_json(config.SAMPLE_JSON, default=[])
    if not sample_meta:
        raise SystemExit("no data/sample.json -- run scripts/01_sample.py first")
    by_id = {p["base_id"]: p for p in sample_meta}

    work = []
    for f in sorted(config.CANDIDATES_DIR.glob("*.json")):
        bid = f.stem
        if only_paper and bid != only_paper:
            continue
        cand = read_json(f)
        if not cand or cand.get("error"):
            continue
        paper = by_id.get(bid)
        parsed = read_json(config.PARSED_DIR / f"{bid}.json")
        if not (paper and parsed and parsed.get("ok")):
            continue
        rows = cand.get("samples") or []
        for s in rows:
            s = dict(s)
            s["_n_in_paper"] = len(rows)
            work.append((paper, parsed, s))
    return work


def main() -> None:
    config.ensure_dirs()
    force = "--force" in sys.argv
    only_paper = None
    limit = None
    if "--paper" in sys.argv:
        only_paper = sys.argv[sys.argv.index("--paper") + 1]
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    work = work_list(only_paper)
    if not work:
        raise SystemExit(
            "nothing to extract. Run scripts/05_enumerate.py first "
            "(data/candidates/ is empty or holds only errors).")
    if limit:
        work = work[:limit]

    papers = len({w[0]["base_id"] for w in work})
    print(f"{len(work)} samples across {papers} papers")
    print(f"model: {config.MODEL_RETRIEVE}   workers: {config.MAX_WORKERS}\n")

    def task(item):
        paper, parsed, s5 = item
        sid = s5.get("sample_id") or ""
        out = config.EXTRACT_DIR / f"{sid}.json"
        # sample ids are positional (s000, s001...), so a re-enumerated paper
        # can reuse an id for a DIFFERENT comparison. A cache hit therefore
        # requires the identity to match, not just the filename.
        ident = {"technique": s5.get("technique", ""),
                 "setting": s5.get("setting", "")}
        if out.exists() and not force:
            cached = read_json(out)
            if cached and cached.get("_identity") == ident:
                return sid, s5, cached, "cached"
        try:
            res, _ = run_retrieve_sample(paper, parsed, s5)
        except Exception as e:
            # No file written: a failed call is not a result, and the next run
            # retries this sample.
            return sid, s5, {"error": f"{type(e).__name__}: {str(e)[:200]}"}, "error"
        res["_model"] = config.MODEL_RETRIEVE
        res["_identity"] = ident
        res["sample_id"] = sid
        write_json(out, res)
        return sid, s5, res, "fresh"

    stats = Counter()
    reasons = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(task, it) for it in work]
        for n, fut in enumerate(as_completed(futures), 1):
            sid, s5, res, how = fut.result()
            if res.get("error"):
                stats["error"] += 1
                log(f"  {sid} ERROR: {res['error']}")
            elif res.get("found"):
                stats["found"] += 1
            else:
                stats["not_found"] += 1
                reasons.append((sid, s5.get("technique", "")[:34],
                                (res.get("not_found_reason") or "")[:80]))
            stats[how] += 1
            if n % 5 == 0 or n == len(work):
                print(f"  {n}/{len(work)}  found={stats['found']} "
                      f"not_found={stats['not_found']} error={stats['error']}")

    print()
    print("=" * 68)
    n = len(work)
    print(f"  extracted (numbers found)  {stats['found']:>4}/{n}")
    print(f"  not_found                  {stats['not_found']:>4}/{n}")
    print(f"  errored (no file written)  {stats['error']:>4}/{n}")
    print(f"  cached from earlier run    {stats['cached']:>4}")
    print("=" * 68)
    if reasons:
        print("\n  why samples could not be located "
              "(this is a quality measure on STAGE 5):")
        for sid, tech, why in reasons[:15]:
            print(f"    {sid}  {tech:<34} {why}")
        if len(reasons) > 15:
            print(f"    ... and {len(reasons) - 15} more")

    t = totals()
    if t["calls"]:
        print(f"\n  {t['calls']} API calls, "
              f"{t['prompt_tokens']:,}+{t['completion_tokens']:,} tokens, "
              f"cost ${t['cost']:.4f}")
    print(f"\n  per-sample -> {config.EXTRACT_DIR}")
    print("  next: python scripts/07_assemble.py")


if __name__ == "__main__":
    main()
