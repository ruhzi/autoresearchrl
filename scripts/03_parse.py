"""Stage 3: parse LaTeX into sections + tables, and build the sanitised context.

No API calls here -- this is all local, so run it freely while iterating on the
parser. It writes two things per paper:

  parsed/<id>.json   full structure (sections with bodies, table source)
  parsed/<id>.ctx.json  the model-visible context: results sections deleted,
                        remaining numbers scrubbed, plus the structure-only
                        leak probe for the floor baseline.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.latex import parse_paper  # noqa: E402
from arl.sanitize import build_context, leak_probe_context  # noqa: E402
from arl.util import read_json, write_json  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    sample = read_json(config.SAMPLE_JSON, default=[])
    if not sample:
        raise SystemExit("no sample found -- run scripts/01_sample.py first")

    ok, failed = 0, Counter()
    table_counts, ctx_chars, redactions = [], [], Counter()

    for i, p in enumerate(sample, 1):
        bid = p["base_id"]
        src = config.SRC_DIR / bid
        if not src.is_dir():
            failed["no_source_dir"] += 1
            print(f"{i:3d}/{len(sample)} {bid} SKIP no source")
            continue

        parsed = parse_paper(src)
        if not parsed.get("ok"):
            failed[parsed.get("reason", "parse_failed")] += 1
            print(f"{i:3d}/{len(sample)} {bid} FAIL {parsed.get('reason')}")
            write_json(config.PARSED_DIR / f"{bid}.json", parsed)
            continue

        abstract = parsed.get("abstract") or p.get("abstract", "")
        ctx = build_context(parsed, abstract=abstract,
                            max_chars=config.MAX_CONTEXT_CHARS)
        ctx.update(leak_probe_context(parsed))
        ctx["base_id"] = bid
        ctx["title"] = p["title"]

        write_json(config.PARSED_DIR / f"{bid}.json", parsed)
        write_json(config.PARSED_DIR / f"{bid}.ctx.json", ctx)

        ok += 1
        table_counts.append(parsed["n_tables"])
        ctx_chars.append(ctx["context_chars"])
        for k, v in ctx["redactions"].items():
            redactions[k] += v

        kinds = Counter(s["kind"] for s in parsed["sections"])
        print(f"{i:3d}/{len(sample)} {bid} ok  tables={parsed['n_tables']:<3} "
              f"ctx={ctx['context_chars']:>6}  "
              f"redacted={ctx['redactions'].get('total', 0):<3} "
              f"sections={dict(kinds)}")

    print(f"\nparsed OK: {ok}/{len(sample)}")
    if failed:
        print(f"failures: {dict(failed)}")
    if table_counts:
        table_counts.sort()
        print(f"tables per paper: min={table_counts[0]} "
              f"median={table_counts[len(table_counts) // 2]} "
              f"max={table_counts[-1]}  "
              f"papers with 0 tables={sum(1 for t in table_counts if t == 0)}")
        print(f"context chars: median="
              f"{sorted(ctx_chars)[len(ctx_chars) // 2]}")
        print(f"redactions across corpus: {dict(redactions)}")


if __name__ == "__main__":
    main()
