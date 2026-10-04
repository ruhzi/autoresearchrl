"""Stage 5 (standalone): enumerate candidate samples per eligible paper.

For every paper that passed the stage-4 screen, ask the strong model:
  - how many controlled comparisons does this paper contain?
  - what is each one, in outline: technique / setting / metric
  - does the technique IMPROVE or DEGRADE the system on that metric?

Scoping only. Per-sample detail (exact variants, mechanism, table location,
numbers) is stage 6's job -- this stage exists to answer "how many samples and
is the direction balanced" as cheaply as possible.

No numbers are extracted. Reads data/screen/ + data/parsed/, writes one file per
paper to data/candidates/<id>.json plus a corpus summary to
data/candidates_summary.json. Idempotent -- skips done papers unless --force.

Runs as a recursive language model (RLM): the paper lives in a PaperStore
environment, a root agent batches sections/tables and dispatches enumerator
subagents, and submit() is refused until every part of the paper is covered --
so long papers no longer lose their appendix to a context cap. Papers that fit
one subagent budget take a single call (fast path).

    python scripts/05_enumerate.py
    python scripts/05_enumerate.py --force
    python scripts/05_enumerate.py --all          # ignore the screen, do every paper
    python scripts/05_enumerate.py --paper 2608.07911   # one paper only
    python scripts/05_enumerate.py --limit 4            # first N eligible papers
    python scripts/05_enumerate.py --show 2608.09296
    python scripts/05_enumerate.py --summary      # re-print from cached files
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.candidates import annotate, run_enumerate_rlm, summarize  # noqa: E402
from arl.openrouter import totals  # noqa: E402
from arl.util import log, read_json, write_json  # noqa: E402

SUMMARY_JSON = config.DATA / "candidates_summary.json"


def eligible_papers(use_all: bool) -> list[tuple[dict, dict]]:
    sample = read_json(config.SAMPLE_JSON, default=[])
    if not sample:
        raise SystemExit("no data/sample.json -- run scripts/01_sample.py first")

    out, skipped = [], 0
    for p in sample:
        bid = p["base_id"]
        parsed = read_json(config.PARSED_DIR / f"{bid}.json")
        if not (parsed and parsed.get("ok")):
            continue
        if not use_all:
            verdict = read_json(config.SCREEN_DIR / f"{bid}.json")
            if verdict is None:
                skipped += 1
                continue
            if not verdict.get("eligible"):
                continue
        out.append((p, parsed))

    if skipped:
        log(f"note: {skipped} papers have no screen verdict yet "
            f"(run scripts/04_screen.py, or pass --all to ignore the screen)")
    return out


def load_cached() -> dict:
    res = {}
    for f in sorted(config.CANDIDATES_DIR.glob("*.json")):
        d = read_json(f)
        if d:
            res[f.stem] = d
    return res


def print_paper(bid: str, title: str, res: dict) -> None:
    rows = res.get("samples") or []
    print(f"\n{'-' * 74}")
    print(f"{bid}  {title[:60]}")
    print(f"  {len(rows)} candidate samples"
          + (f"   [{res['error']}]" if res.get("error") else ""))
    # No direction column: stage 5 says which comparisons exist, not how they
    # turned out (see ENUMERATE_SCHEMA). Printing one meant printing '?'.
    for i, s in enumerate(rows, 1):
        print(f"   {i:>3}. {s.get('technique', '')[:36]:<36} | "
              f"{', '.join(s.get('metrics') or [])[:26]:<26} | "
              f"{s.get('setting', '')[:30]:<30}")
    if res.get("notes"):
        print(f"   notes: {res['notes'][:160]}")


def print_summary(summ: dict) -> None:
    print("\n" + "=" * 74)
    print("CANDIDATE SAMPLE SUMMARY")
    print("=" * 74)
    print(f"  papers enumerated        {summ['papers']}")
    print(f"  TOTAL candidate samples  {summ['total_candidate_samples']}"
          f"   (one per technique x setting; metrics ride along)")
    if summ["papers"]:
        print(f"  samples per paper        "
              f"{summ['total_candidate_samples'] / summ['papers']:.1f} avg")
        counts = sorted(summ["samples_per_paper"].values(), reverse=True)
        print(f"  per-paper spread         {counts}")
    # The direction balance that used to print here is gone with the label:
    # stage 5 no longer guesses how a comparison turned out, so there is
    # nothing to balance. The effect-size histogram in scripts/report.py is
    # what replaced it, and it runs on real numbers from stage 6.
    if summ.get("metric_mentions"):
        print("\n  most-reported metrics across samples:")
        for m, c in list(summ["metric_mentions"].items())[:12]:
            print(f"    {c:>3}  {m}")


def main() -> None:
    config.ensure_dirs()
    force = "--force" in sys.argv
    use_all = "--all" in sys.argv

    if "--show" in sys.argv:
        bid = sys.argv[sys.argv.index("--show") + 1]
        d = read_json(config.CANDIDATES_DIR / f"{bid}.json")
        print(json.dumps(d, indent=2) if d else f"nothing for {bid}")
        return

    if "--summary" in sys.argv:
        cached = load_cached()
        if not cached:
            print("no cached enumerations yet")
            return
        print_summary(summarize(cached))
        return

    pairs = eligible_papers(use_all)
    if "--paper" in sys.argv:
        only = sys.argv[sys.argv.index("--paper") + 1]
        pairs = [pp for pp in pairs if pp[0]["base_id"] == only]
    if "--limit" in sys.argv:
        pairs = pairs[:int(sys.argv[sys.argv.index("--limit") + 1])]
    if not pairs:
        raise SystemExit(
            "no eligible papers. Run scripts/04_screen.py first, "
            "or pass --all to enumerate every parsed paper regardless.")
    print(f"enumerating {len(pairs)} papers  "
          f"(root: {config.MODEL_RLM_ROOT}, "
          f"enumerators: {config.MODEL_ENUM_SUB})\n")

    def task(item):
        p, parsed = item
        bid = p["base_id"]
        out = config.CANDIDATES_DIR / f"{bid}.json"
        if out.exists() and not force:
            cached = read_json(out)
            # Error results from older runs must not satisfy the cache:
            # a failed paper retries instead of silently vanishing.
            if cached and not cached.get("error"):
                return bid, p, cached, "cached"
        try:
            res, _ = run_enumerate_rlm(p, parsed)
            res = annotate(res, bid)
        except Exception as e:
            # No file written: a failed paper is not a result. The RLM
            # checkpoint in candidates_partial/ keeps the finished batches,
            # so the next run resumes rather than re-billing them.
            return bid, p, {"error": f"{type(e).__name__}: {str(e)[:200]}",
                            "samples": []}, "error"
        write_json(out, res)
        return bid, p, res, "fresh"

    results = {}
    rows = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(task, it) for it in pairs]
        for fut in as_completed(futures):
            bid, p, res, how = fut.result()
            results[bid] = res
            rows.append((bid, p["title"], res))
            if res.get("error"):
                log(f"  {bid} ERROR: {res['error']}")

    for bid, title, res in sorted(rows, key=lambda r: -len(r[2].get("samples") or [])):
        print_paper(bid, title, res)

    summ = summarize(results)
    write_json(SUMMARY_JSON, summ)
    print_summary(summ)

    t = totals()
    if t["calls"]:
        print(f"\n  {t['calls']} API calls, "
              f"{t['prompt_tokens']:,}+{t['completion_tokens']:,} tokens, "
              f"cost ${t['cost']:.4f}, errors={t['errors']}")
    print(f"\n  per-paper -> {config.CANDIDATES_DIR}")
    print(f"  summary   -> {SUMMARY_JSON}")
    print("  next: python scripts/06_extract.py   (numeric values)")


if __name__ == "__main__":
    main()
