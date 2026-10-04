"""Pipeline driver. One-shot, or any single stage, or any range.

Every stage reads the previous stage's saved output from data/, so each one runs
standalone as long as the stage before it has completed at least once. This
driver is just an ordered runner over the same scripts.

The pipeline ends at stage 8 with the final dataset, data/train_samples.jsonl.
(scripts/report.py and scripts/reward_diagnose.py are separate inspection
utilities, not pipeline stages.)

Stages 5-8 run DEPTH-FIRST per paper: each eligible paper is taken all the
way through to the final dataset before the next paper starts. Stop the run
at any point and you have N fully finished papers, not five papers stranded
at stage 6. Pass --by-stage for the old breadth-first order.

    python run_all.py                 # everything, stages 1-8
    python run_all.py --limit 2       # first 2 eligible papers, end to end
    python run_all.py --paper 2608.10544   # one paper, end to end
    python run_all.py --dry-run       # print the plan, execute nothing
    python run_all.py --local         # stages 1-3 only: no API key, no cost
    python run_all.py --only 4        # just the screen
    python run_all.py --only 5        # just the enumeration
    python run_all.py --from 4        # screen onward
    python run_all.py --from 4 --to 5 # screen + enumerate, stop before numbers
    python run_all.py --list          # show the stages and exit

Flags passed through to the stages that understand them:
    --force     redo work that is already cached on disk
    --refresh   stage 1 only: rebuild the arXiv pool
    --all       stage 5 only: enumerate every parsed paper, ignoring the screen
    --redo-done revisit papers listed in data/papers_done.json (by default a
                paper that has finished stage 8 once is skipped, so samples
                that fail deterministically are not re-billed every run)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (stage number, script, one-line description, costs money?)
STAGES = [
    (1, "01_sample.py", "draw a reproducible sample from the arXiv pool", False),
    (2, "02_fetch.py", "download + unpack e-print LaTeX source", False),
    (3, "03_parse.py", "sections, tables, sanitised context", False),
    (4, "04_screen.py", "is-AI-research + introduces-method-with-results", True),
    (5, "05_enumerate.py", "RLM enumeration: which comparisons exist, no numbers", True),
    (6, "06_extract.py", "retrieval agent: verbatim spans + numbers per sample", True),
    (7, "07_assemble.py", "labels + LLM-composed researcher prompts", True),
    (8, "08_verify.py", "LLM audit: no outcome refs, faithful -> final dataset", True),
]

# which stages accept which pass-through flags
FLAG_STAGES = {
    "--refresh": {1},
    "--force": {4, 5, 6, 7, 8},
    "--all": {5},
    "--strict": {7},
    "--labels-only": {7},
    "--limit": {5, 6, 7, 8},
    "--paper": {5, 6},
    "--include-flagged": {7},
}

# flags that carry a value which must be forwarded with them
VALUE_FLAGS = {"--limit", "--paper"}


def run(script: str, extra: list[str], dry: bool = False) -> None:
    print(f"\n{'=' * 66}\n>>> {script} {' '.join(extra)}\n{'=' * 66}")
    if dry:
        return
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra])
    if r.returncode != 0:
        raise SystemExit(f"{script} exited {r.returncode} -- stopping")


def eligible_paper_ids() -> list[str]:
    """Papers that passed the screen and parsed OK, in sample order."""
    sys.path.insert(0, str(ROOT))
    import config
    from arl.util import read_json

    out = []
    for p in read_json(config.SAMPLE_JSON, default=[]):
        bid = p["base_id"]
        parsed = read_json(config.PARSED_DIR / f"{bid}.json")
        verdict = read_json(config.SCREEN_DIR / f"{bid}.json")
        if parsed and parsed.get("ok") and verdict and verdict.get("eligible"):
            out.append(bid)
    return out


def done_manifest_path() -> Path:
    sys.path.insert(0, str(ROOT))
    import config
    return config.DATA / "papers_done.json"


def load_done() -> list[str]:
    from arl.util import read_json
    return list(read_json(done_manifest_path(), default=[]))


def mark_done(bid: str) -> None:
    """Record that this paper has finished stage 8 once, so later runs skip it
    instead of re-billing the samples that fail deterministically."""
    from arl.util import write_json
    done = load_done()
    if bid not in done:
        done.append(bid)
        write_json(done_manifest_path(), done)


def dataset_size() -> int:
    sys.path.insert(0, str(ROOT))
    import config
    try:
        with open(config.TRAIN_JSONL, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def arg_int(name: str, default: int) -> int:
    if name in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(name) + 1])
        except (IndexError, ValueError):
            raise SystemExit(f"{name} needs a stage number")
    return default


def main() -> None:
    args = sys.argv[1:]

    if "--list" in args:
        print("stage  script                   cost  description")
        for n, s, d, paid in STAGES:
            print(f"  {n}    {s:<24} {'$' if paid else '-':<5} {d}")
        return

    lo, hi = 1, 8
    if "--local" in args:
        lo, hi = 1, 3
    if "--only" in args:
        lo = hi = arg_int("--only", 1)
    else:
        lo = arg_int("--from", lo)
        hi = arg_int("--to", hi)

    selected = [s for s in STAGES if lo <= s[0] <= hi]
    if not selected:
        raise SystemExit(f"no stages in range {lo}-{hi}")

    paid = [s[0] for s in selected if s[3]]
    print(f"running stages {lo}-{hi}: "
          + ", ".join(f"{n}:{s.replace('.py', '')}" for n, s, _, _ in selected))
    print(f"stages that make API calls: {paid or 'none'}")

    dry = "--dry-run" in args
    scripts = {n: s for n, s, _, _ in STAGES}

    def flags_for(num: int, skip: set | None = None) -> list[str]:
        extra = []
        for f, stages in FLAG_STAGES.items():
            if f in args and num in stages and f not in (skip or set()):
                extra.append(f)
                if f in VALUE_FLAGS:
                    try:
                        extra.append(args[args.index(f) + 1])
                    except IndexError:
                        raise SystemExit(f"{f} needs a value")
        return extra

    # Stages 1-4 are corpus-wide and cheap; run them once, in order.
    for num, script, _, _ in selected:
        if num <= 4:
            run(script, flags_for(num), dry)

    # Stages 5-8 run DEPTH-FIRST per paper: each eligible paper goes all the
    # way to the final dataset before the next paper starts, so a stopped run
    # leaves N complete papers, never five papers stranded mid-stage.
    # (Stages 7/8 are corpus-wide but per-sample cached, so re-running them
    # after each paper only pays for that paper's new samples.)
    # Pass --by-stage for the old breadth-first order.
    deep = [n for n, _, _, _ in selected if n >= 5]
    if deep:
        if "--by-stage" in args:
            for num in deep:
                run(scripts[num], flags_for(num), dry)
        else:
            papers = eligible_paper_ids()
            # A paper that has been through stage 8 once is skipped: its
            # outputs are cached, and the only work left would be re-billing
            # the samples that fail every time. --redo-done, --force or an
            # explicit --paper revisit it.
            if not ({"--redo-done", "--force", "--paper"} & set(args)):
                done = set(load_done())
                skipped = [b for b in papers if b in done]
                papers = [b for b in papers if b not in done]
                if skipped:
                    print(f"\nskipping {len(skipped)} papers already through "
                          f"stage 8 (data/papers_done.json); pass "
                          f"--redo-done to revisit them")
            if "--paper" in args:
                only = args[args.index("--paper") + 1]
                papers = [b for b in papers if b == only]
            if "--limit" in args:
                papers = papers[:int(args[args.index("--limit") + 1])]
            if not papers and not dry:
                raise SystemExit("no eligible papers (run stages 1-4 first)")
            for i, bid in enumerate(papers, 1):
                print(f"\n{'#' * 66}\n# PAPER {i}/{len(papers)}: {bid}"
                      f"\n{'#' * 66}")
                # --limit/--paper are consumed at this level: forwarding them
                # would cap samples inside a stage instead of papers.
                skip = {"--limit", "--paper"}
                for num in deep:
                    extra = flags_for(num, skip)
                    if num in (5, 6):
                        extra += ["--paper", bid]
                    run(scripts[num], extra, dry)
                if not dry and 8 in deep:
                    mark_done(bid)
                    print(f"\n# paper {bid} complete -- train_samples.jsonl "
                          f"now holds {dataset_size()} samples")

    if hi <= 3:
        print("\nLocal stages done -- no API calls were made.")
    print(f"\nfinished stages {lo}-{hi}" + (" (dry run)" if dry else ""))


if __name__ == "__main__":
    main()
