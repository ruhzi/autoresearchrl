
A pipeline that mines **controlled numeric comparisons** from recent arXiv
machine-learning papers into training samples. Each sample is one comparison
from one table: a technique, the setting it was measured in, and the value
every reported metric took with and without it. The task the samples pose
is: given the paper's introduction and method with all results removed,
predict each metric's value after the technique is applied. Values are the
target; whether a change "helps" is derived downstream from the value and
the metric's polarity, never asked of a model.

Comparisons are within-paper (ablations, same-setting rows), so the delta
reflects the technique rather than differences in compute, tuning, or seeds
across papers.

## The pipeline

| Stage | Script | What it does | Cost |
|---|---|---|---|
| 1 | `scripts/01_sample.py` | draw a reproducible sample from the arXiv pool | free |
| 2 | `scripts/02_fetch.py` | download + unpack e-print LaTeX source | free |
| 3 | `scripts/03_parse.py` | sections, tables, sanitised context | free |
| 4 | `scripts/04_screen.py` | is-AI-research + introduces-method-with-results | LLM |
| 5 | `scripts/05_enumerate.py` | RLM enumeration: which comparisons exist, no numbers | LLM |
| 6 | `scripts/06_extract.py` | retrieval agent: verbatim spans + numbers per sample | LLM |
| 7 | `scripts/07_assemble.py` | labels + LLM-composed researcher prompts | LLM |
| 8 | `scripts/08_verify.py` | LLM audit: no outcome refs, faithful -> final dataset | LLM |

Stages 5-8 run depth-first per paper: stop at any point and you have N
finished papers, not a pile of half-processed ones.

What each LLM stage is asked, and not asked:

- **Stage 5** reads the paper through grep/peek tools and lists which
  comparisons exist and what they measure. It never reports numbers.
- **Stage 6** is handed one comparison at a time and locates it with
  grep/read. It submits *pointers* (anchor strings), and Python slices the
  prose spans, the evidence rows, and the two cell values straight out of
  the LaTeX source. The agent never types paper text. "Not found" is a
  first-class answer.
- **Stage 7** joins 5 and 6 into labelled samples and composes the
  researcher-facing prompt from the paper with results removed.
- **Stage 8** audits each prompt for leaked outcomes and faithfulness and
  writes the survivors to the final dataset.

## Requirements

- Python 3.12 or newer (3.10.12+ and 3.11.4+ also work). No third-party
  packages at runtime; `pytest` for the offline tests.
- Network access to arXiv (stages 1-2) and OpenRouter (stages 4-8).
- One `OPENROUTER_API_KEY`, as an environment variable or a line in a
  `.env` file at the repo root (`OPENROUTER_API_KEY=sk-or-...`).

## Quickstart

```powershell
python run_all.py --local              # stages 1-3: no key, no cost
python scripts/00_list_models.py anthropic z-ai deepseek  # confirm the default model slugs exist
python run_all.py --limit 2            # 2 eligible papers through stages 5-8 (stage 4 screens all sampled papers)
python run_all.py --list               # show the stages and exit
python run_all.py --dry-run            # print the plan, execute nothing
```

Output lands in `data/`; the final dataset is `data/train_samples.jsonl`.
Every stage reads the previous stage's output from disk, so any stage can be
re-run alone (`python run_all.py --only 6`) or from a point (`--from 4`).

## Configuration

All knobs are environment variables read by `config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | (required for stages 4-8) | OpenRouter key |
| `ARL_DATA_DIR` | `<repo>/data` | where every stage reads and writes |
| `ARL_DATE_FROM` | `202607010000` | arXiv submittedDate window start, YYYYMMDDHHMM |
| `ARL_DATE_TO` | `202608260000` | window end, exclusive |
| `ARL_SAMPLE_SIZE` | `50` | papers processed; raising it grows the corpus |
| `ARL_SAMPLE_SEED` | `20260812` | draw seed |
| `ARL_MAX_WORKERS` | `8` | concurrent LLM calls |
| `ARL_MODEL_GATE` | `~deepseek/deepseek-v4-flash-latest` | stage 4 |
| `ARL_MODEL_RLM_ROOT` | `anthropic/claude-sonnet-5` | stage 5 root agent |
| `ARL_MODEL_ENUM_SUB` | `anthropic/claude-sonnet-5` | stage 5 enumerator subagents |
| `ARL_MODEL_RETRIEVE` | `z-ai/glm-5.2` | stage 6 retrieval agent |
| `ARL_MODEL_COMPOSE` | `anthropic/claude-sonnet-5` | stage 7 composer |
| `ARL_MODEL_SAMPLE_VERIFY` | `anthropic/claude-sonnet-5` | stage 8 judge |

Model slugs are OpenRouter slugs, current as of 2026-10-01. Check them with
`scripts/00_list_models.py` before a run; providers rename and retire models.


Corrections are pipeline inputs so they survive re-runs. Three optional
files in `ARL_DATA_DIR`:

- `label_overrides.json` — `{sample_id: {field: value}}`, applied in stage 7
  (a `measurements` list replaces the whole grid).
- `prompt_overrides.json` — `{sample_id: "prompt body"}`, applied in stage 8.
- `dropped_sample_ids.json` — `[sample_id, ...]`, excluded from the final dataset in stage 8.

## Reproducibility

- LLM output is not deterministic, even at temperature 0. Two runs on the
  same papers produce overlapping, not identical, sample sets.
- `data/sample.json` is a manifest: once a paper is drawn it stays, and
  raising `ARL_SAMPLE_SIZE` only adds papers. Delete it to redraw.
  Changing `ARL_DATE_FROM`/`ARL_DATE_TO` after a run has no effect on an
  existing `pool.json` and `sample.json`; use a fresh `ARL_DATA_DIR`, or
  delete those two files, or pass `--refresh --redraw` to stage 1 (`--refresh`
  rebuilds the pool, `--redraw` abandons the previous draw), to redraw.
- The arXiv pool changes daily. Pin the date window, and keep
  `ARL_DATE_FROM` past the pretraining cutoff of any model you will
  evaluate on the output, or you are measuring recall of a memorised paper.
- Every per-sample output file records the model slug that produced it.

## Offline checks

```powershell
python -m pytest tests
python scripts/selftest.py
```

## License

MIT. See `LICENSE`.
