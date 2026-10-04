"""Stage 2: download and unpack arXiv LaTeX source for the sampled papers."""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.fetch_source import fetch_one  # noqa: E402
from arl.util import log, read_json, write_json  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    sample = read_json(config.SAMPLE_JSON, default=[])
    if not sample:
        raise SystemExit("no sample found -- run scripts/01_sample.py first")

    report = []
    for i, p in enumerate(sample, 1):
        bid = p["base_id"]
        rec = fetch_one(bid, config.SRC_DIR / bid)
        rec["title"] = p["title"]
        report.append(rec)
        print(f"{i:3d}/{len(sample)} {bid} {rec['status']:<14} "
              f"tex={len(rec.get('tex_files', []))}")
        if rec["status"] != "cached":
            time.sleep(config.EPRINT_DELAY_S)

    write_json(config.FETCH_REPORT, report)
    counts = Counter(r["status"] for r in report)
    print(f"\nstatus counts: {dict(counts)}")
    usable = sum(v for k, v in counts.items()
                 if k in ("tar", "single_tex", "cached"))
    print(f"usable LaTeX source: {usable}/{len(report)}")
    if counts.get("pdf_only"):
        log(f"note: {counts['pdf_only']} papers are PDF-only and are dropped. "
            f"That is the intended behaviour -- PDF table extraction is where "
            f"this kind of pipeline usually breaks.")


if __name__ == "__main__":
    main()
