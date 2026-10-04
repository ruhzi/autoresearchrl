"""Stage 1: build the arXiv candidate pool and draw a reproducible sample."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.arxiv_api import build_pool, draw_sample  # noqa: E402
from arl.util import log, read_json, write_json  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    pool = read_json(config.POOL_JSON, default=None)
    if pool and "--refresh" not in sys.argv:
        log(f"pool: reusing {len(pool)} cached papers "
            f"({config.POOL_JSON}); pass --refresh to rebuild")
    else:
        pool = build_pool()
        write_json(config.POOL_JSON, pool)
        # The final pool supersedes the page checkpoint.
        config.POOL_PARTIAL_JSON.unlink(missing_ok=True)

    if not pool:
        raise SystemExit("pool is empty -- check the arXiv query window in config.py")

    # The previous draw is a manifest, not a cache: every paper in it has
    # already been paid for through stage 8, so it is carried forward whatever
    # happened to the pool or the query window. --redraw abandons it (and
    # orphans that work) -- it exists so the choice is explicit, never a
    # side effect of editing config.py.
    keep = [] if "--redraw" in sys.argv else read_json(config.SAMPLE_JSON,
                                                      default=[])
    in_pool = {p["base_id"] for p in pool}
    orphans = [r["base_id"] for r in keep if r["base_id"] not in in_pool]

    sample = draw_sample(pool, config.SAMPLE_SIZE, config.SAMPLE_SEED,
                         keep=keep)
    write_json(config.SAMPLE_JSON, sample)

    dates = sorted(p["published"] for p in pool)
    print(f"\npool: {len(pool)} papers, submitted {dates[0]} .. {dates[-1]}")
    print(f"categories queried: {', '.join(config.CATEGORIES)}")
    print(f"primary category mix in pool: "
          f"{dict(Counter(p['primary_cat'] for p in pool).most_common(8))}")
    print(f"\nsample: {len(sample)} papers (seed={config.SAMPLE_SEED}) -- "
          f"{len(keep)} carried forward, {len(sample) - len(keep)} new\n")
    for i, s in enumerate(sample, 1):
        tag = "     " if i <= len(keep) else " NEW "
        print(f"{i:3d}.{tag}{s['base_id']}  [{(s.get('primary_cat') or '?'):<12}] "
              f"{s['title'][:72]}")

    if len(sample) < config.SAMPLE_SIZE:
        log(f"\nNOTE: asked for {config.SAMPLE_SIZE} but the pool only has "
            f"{len(pool)} papers, so the draw is short. Raise "
            f"ARL_POOL_TARGET (and --refresh) to widen it.")
    if len(keep) > config.SAMPLE_SIZE:
        log(f"\nNOTE: SAMPLE_SIZE={config.SAMPLE_SIZE} is below the "
            f"{len(keep)} papers already drawn. The draw never shrinks -- "
            f"those papers are kept. Pass --redraw to start over.")
    if orphans:
        log(f"\nNOTE: {len(orphans)} carried-forward papers are no longer in "
            f"the pool (the query window moved): {', '.join(orphans[:5])}"
            f"{' ...' if len(orphans) > 5 else ''}. They stay in the sample "
            f"because their downstream work is already paid for.")

    if dates[0][:10] != config.DATE_FROM[:8]:
        log(f"\nNOTE: the pool only spans {dates[0][:10]} .. {dates[-1][:10]}. "
            f"POOL_TARGET={config.POOL_TARGET} was filled by the most recent "
            f"submissions before reaching the start of the configured window, so "
            f"this is 'random within those days', not across the full window. "
            f"Raise ARL_POOL_TARGET to widen it.")


if __name__ == "__main__":
    main()
