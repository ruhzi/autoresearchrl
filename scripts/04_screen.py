"""Stage 4 (standalone): two-question eligibility screen.

  1. is_ai_research         is this AI research
  2. introduces_something   does it introduce something of its own
  3. has_results            are results attached

`eligible` is the AND of the three, computed in Python -- the model is never
asked to derive it. Responses are validated against the schema first: a
malformed reply writes no file and is retried on the next run.

Runs a cheap model once per parsed paper. Reads data/parsed/, writes one verdict
per paper to data/screen/<id>.json. Idempotent: already-screened papers are
skipped unless --force.

    python scripts/04_screen.py
    python scripts/04_screen.py --force        # re-screen everything
    python scripts/04_screen.py --show 2608.09296   # print one verdict in full
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.extract import run_screen  # noqa: E402
from arl.openrouter import totals  # noqa: E402
from arl.util import log, read_json, write_json  # noqa: E402


def load_pairs() -> list[tuple[dict, dict]]:
    """(paper metadata, parsed document) for every successfully parsed paper."""
    sample = read_json(config.SAMPLE_JSON, default=[])
    if not sample:
        raise SystemExit("no data/sample.json -- run scripts/01_sample.py first")
    pairs = []
    for p in sample:
        parsed = read_json(config.PARSED_DIR / f"{p['base_id']}.json")
        if parsed and parsed.get("ok"):
            pairs.append((p, parsed))
    if not pairs:
        raise SystemExit("no parsed papers -- run scripts/03_parse.py first")
    return pairs


def main() -> None:
    config.ensure_dirs()
    force = "--force" in sys.argv

    if "--show" in sys.argv:
        bid = sys.argv[sys.argv.index("--show") + 1]
        v = read_json(config.SCREEN_DIR / f"{bid}.json")
        print(json.dumps(v, indent=2) if v else f"no verdict for {bid}")
        return

    pairs = load_pairs()
    print(f"screening {len(pairs)} parsed papers with {config.MODEL_GATE}\n")

    def task(item):
        p, parsed = item
        bid = p["base_id"]
        out = config.SCREEN_DIR / f"{bid}.json"
        if out.exists() and not force:
            return bid, p, read_json(out), "cached"
        try:
            res, _ = run_screen(p, parsed)
        except Exception as e:
            # Deliberately write NOTHING. A failed or malformed call is not a
            # verdict -- writing one made a schema violation look like a
            # decision, and left errors=0 while records were corrupt. With no
            # file on disk, the next run retries this paper.
            return bid, p, {"error": f"{type(e).__name__}: {str(e)[:200]}"}, "error"
        # Stamp provenance -- a floating alias like ~...-latest can change model
        # underneath you between runs, and then verdicts are unattributable.
        res["_model"] = config.MODEL_GATE
        write_json(out, res)
        return bid, p, res, "fresh"

    rows = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(task, it) for it in pairs]
        for fut in as_completed(futures):
            bid, p, res, how = fut.result()
            rows.append((bid, p["title"], res, how))
            if res.get("error"):
                log(f"  {bid} ERROR: {res['error']}")

    rows.sort(key=lambda r: (not r[2].get("eligible"), r[0]))
    for bid, title, res, how in rows:
        mark = "PASS" if res.get("eligible") else "drop"
        print(f"  {mark:<4} {bid}  "
              f"ai={str(res.get('is_ai_research')):<5} "
              f"introduces={str(res.get('introduces_something')):<5} "
              f"results={str(res.get('has_results')):<5}"
              + (f"  failed={res['failed_criteria']}"
                 if res.get("failed_criteria") else ""))
        print(f"       {title[:74]}")
        if not res.get("eligible"):
            print(f"       why: {res.get('reason', '')[:110]}")

    n = len(rows)
    q1 = sum(1 for _, _, r, _ in rows if r.get("is_ai_research"))
    q2 = sum(1 for _, _, r, _ in rows if r.get("introduces_something"))
    q3 = sum(1 for _, _, r, _ in rows if r.get("has_results"))
    ok = sum(1 for _, _, r, _ in rows if r.get("eligible"))

    print()
    print("=" * 62)
    print(f"  1  is AI research         {q1:>3}/{n}")
    print(f"  2  introduces something   {q2:>3}/{n}")
    print(f"  3  has results            {q3:>3}/{n}")
    print(f"  ELIGIBLE (all three)      {ok:>3}/{n}")
    print("=" * 62)
    fails = Counter(c for _, _, r, _ in rows for c in (r.get("failed_criteria") or []))
    print(f"  failed criteria: {dict(fails) if fails else 'none'}")
    errs = [b for b, _, r, _ in rows if r.get("error")]
    if errs:
        print(f"  ERRORED (no verdict written): {errs}")

    t = totals()
    if t["calls"]:
        print(f"\n  {t['calls']} API calls, "
              f"{t['prompt_tokens']:,}+{t['completion_tokens']:,} tokens, "
              f"cost ${t['cost']:.4f}, errors={t['errors']}")
    print(f"\n  verdicts -> {config.SCREEN_DIR}")
    print("  next: python scripts/05_enumerate.py")


if __name__ == "__main__":
    main()
