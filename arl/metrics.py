"""Metric registry: canonical name, family, and polarity.

Canonical metric names and families, so the same metric written five ways
collapses to one key. (The polarity column is retained for post-hoc direction
derivation; nothing in the pipeline reads it.)
The LLM's job is to read (metric_name, baseline_value, method_value) off a
table; whether an increase counts as better is decided here, by lookup, so any
label can be audited by hand against the paper.

polarity: +1 = higher is better, -1 = lower is better.
family:   quality | efficiency | loss   -- lets us ask the tradeoff question
          ("helps FLOPs, hurts downstream accuracy") separately per axis.

An unrecognised metric name is DROPPED, not guessed. Dropping costs samples;
guessing costs label integrity, which is the thing we cannot buy back.
"""
from __future__ import annotations

import re

# canonical -> (family, polarity, aliases)
REGISTRY: dict[str, tuple[str, int, tuple[str, ...]]] = {
    # ---- quality, higher is better
    "accuracy":     ("quality", +1, ("acc", "accuracy", "top1", "top 1", "top-1",
                                      "top1 accuracy", "top-1 accuracy", "test accuracy",
                                      "val accuracy", "validation accuracy", "overall accuracy",
                                      "mean accuracy", "avg accuracy", "average accuracy")),
    "top5_accuracy": ("quality", +1, ("top5", "top 5", "top-5", "top5 accuracy",
                                      "top-5 accuracy")),
    "f1":           ("quality", +1, ("f1", "f1 score", "f-1", "macro f1", "micro f1",
                                      "weighted f1", "f measure")),
    "exact_match":  ("quality", +1, ("em", "exact match", "exact-match")),
    "precision":    ("quality", +1, ("precision", "prec")),
    "recall":       ("quality", +1, ("recall", "rec")),
    "bleu":         ("quality", +1, ("bleu", "sacrebleu", "bleu4", "bleu-4")),
    "rouge":        ("quality", +1, ("rouge", "rouge l", "rouge-l", "rouge1",
                                      "rouge-1", "rouge2", "rouge-2")),
    "meteor":       ("quality", +1, ("meteor",)),
    "chrf":         ("quality", +1, ("chrf", "chrf++")),
    "comet":        ("quality", +1, ("comet",)),
    "auc":          ("quality", +1, ("auc", "auroc", "roc auc", "auc roc")),
    "auprc":        ("quality", +1, ("auprc", "pr auc", "average precision", "ap")),
    "map":          ("quality", +1, ("map", "mean average precision")),
    "mrr":          ("quality", +1, ("mrr", "mean reciprocal rank")),
    "ndcg":         ("quality", +1, ("ndcg", "ndcg@10", "ndcg@5")),
    "iou":          ("quality", +1, ("iou", "miou", "mean iou", "jaccard")),
    "dice":         ("quality", +1, ("dice", "dice score", "dsc")),
    "psnr":         ("quality", +1, ("psnr",)),
    "ssim":         ("quality", +1, ("ssim", "ms-ssim", "ms ssim")),
    "pass_at_k":    ("quality", +1, ("pass@1", "pass@10", "pass@k", "pass at 1",
                                      "pass rate")),
    "success_rate": ("quality", +1, ("success rate", "sr", "task success",
                                      "success", "solve rate", "solved")),
    "win_rate":     ("quality", +1, ("win rate", "winrate", "preference rate")),
    "elo":          ("quality", +1, ("elo", "elo rating")),
    "reward":       ("quality", +1, ("reward", "return", "mean reward",
                                      "average return", "episode return", "score")),
    "correlation":  ("quality", +1, ("spearman", "pearson", "kendall",
                                      "correlation", "rho")),

    # ---- loss-like, lower is better
    "loss":         ("loss", -1, ("loss", "training loss", "train loss",
                                   "test loss", "val loss", "validation loss",
                                   "eval loss", "final loss", "ce loss",
                                   "cross entropy", "cross-entropy")),
    "perplexity":   ("loss", -1, ("ppl", "perplexity", "test ppl", "val ppl",
                                   "validation perplexity", "wikitext ppl")),
    "nll":          ("loss", -1, ("nll", "negative log likelihood",
                                   "negative log-likelihood")),
    "bpc":          ("loss", -1, ("bpc", "bits per character", "bpb",
                                   "bits per byte")),
    "wer":          ("quality", -1, ("wer", "word error rate")),
    "cer":          ("quality", -1, ("cer", "character error rate")),
    "error_rate":   ("quality", -1, ("error rate", "err", "error", "test error",
                                      "classification error")),
    "fid":          ("quality", -1, ("fid", "fid score")),
    "kid":          ("quality", -1, ("kid",)),
    "lpips":        ("quality", -1, ("lpips",)),
    "niqe":         ("quality", -1, ("niqe",)),
    "musiq":        ("quality", +1, ("musiq",)),
    "mse":          ("quality", -1, ("mse", "mean squared error")),
    "rmse":         ("quality", -1, ("rmse", "root mean squared error")),
    "mae":          ("quality", -1, ("mae", "mean absolute error")),
    "hallucination_rate": ("quality", -1, ("hallucination rate",
                                            "hallucination", "toxicity")),

    # ---- agent / game-theoretic evaluation (higher-utility conventions)
    "utility":      ("quality", +1, ("utility", "customer utility", "profit",
                                      "mean utility", "avg utility")),
    "compliance_rate": ("quality", +1, ("compliance rate", "compliance",
                                         "compl", "conditional compliance rate")),
    "defection_rate": ("quality", -1, ("defection rate", "defection", "defect",
                                        "unilateral defection rate", "unilat")),
    "regret":       ("quality", -1, ("regret", "ex post regret",
                                      "ex-post regret", "compliance regret")),
    "recovery_rate": ("quality", +1, ("recovery rate", "dsat sat recovery rate",
                                       "dsat to sat recovery rate")),
    "failed_roundtrips": ("quality", -1, ("failed roundtrips",
                                           "failed roundtrip")),

    # ---- efficiency
    "flops":        ("efficiency", -1, ("flops", "flop", "gflops", "tflops",
                                         "pflops", "macs", "gmacs", "compute")),
    "params":       ("efficiency", -1, ("params", "parameters", "param count",
                                         "model size", "num params",
                                         "trainable params")),
    "memory":       ("efficiency", -1, ("memory", "mem", "gpu memory", "vram",
                                         "peak memory", "memory footprint")),
    "latency":      ("efficiency", -1, ("latency", "inference time", "runtime",
                                         "wall clock", "wall-clock", "time",
                                         "training time", "ttft", "step time",
                                         "seconds", "gpu hours")),
    "throughput":   ("efficiency", +1, ("throughput", "tokens per second",
                                         "tokens/s", "tok/s", "samples/s",
                                         "images/s", "fps", "qps",
                                         "speedup", "speed up")),
    "energy":       ("efficiency", -1, ("energy", "power", "watts", "joules",
                                         "co2", "carbon")),
    "cost":         ("efficiency", -1, ("cost", "dollar cost", "api cost",
                                         "price")),
    "sparsity":     ("efficiency", +1, ("sparsity", "compression ratio",
                                         "pruning ratio")),
}

_ALIAS: dict[str, str] = {}
for _canon, (_fam, _pol, _aliases) in REGISTRY.items():
    _ALIAS[_canon] = _canon
    for _a in _aliases:
        _ALIAS[_a] = _canon


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\\[a-zA-Z]+", " ", s)          # latex commands
    s = re.sub(r"[$\\{}^_~]", " ", s)
    s = re.sub(r"\(.*?\)", " ", s)               # parenthetical units
    s = re.sub(r"[^a-z0-9@+/. -]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Direction arrows are never part of a metric's name; textual up/down is
    # stripped later, in resolve(), only AFTER an exact-alias miss -- so
    # "speedup" and "speed up" (real aliases) survive intact.
    s = re.sub(r"\s*[↑↓]\s*$", "", s).strip()
    return s


def resolve(metric_name: str) -> dict | None:
    """Map a free-text metric name onto the registry. None if unrecognised."""
    if not metric_name:
        return None
    n = _norm(metric_name)
    if not n:
        return None

    if n not in _ALIAS:
        # "Accuracy up" -> "accuracy"; tried only after the exact name missed
        # so alias names that genuinely end in up/down are not mangled.
        stripped = re.sub(r"\s+(?:up|down)\s*$", "", n).strip()
        if stripped in _ALIAS:
            n = stripped

    if n in _ALIAS:
        canon = _ALIAS[n]
    else:
        # Try progressively: drop trailing units/qualifiers, then substring match.
        canon = None
        toks = n.split()
        for cut in range(len(toks), 0, -1):
            cand = " ".join(toks[:cut])
            if cand in _ALIAS:
                canon = _ALIAS[cand]
                break
        if canon is None:
            for cut in range(0, len(toks)):
                cand = " ".join(toks[cut:])
                if cand in _ALIAS:
                    canon = _ALIAS[cand]
                    break
        if canon is None:
            # longest alias appearing as a whole-word substring
            best = ""
            for alias in _ALIAS:
                if len(alias) < 3:
                    continue
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", n):
                    if len(alias) > len(best):
                        best = alias
            if best:
                canon = _ALIAS[best]
        if canon is None:
            return None

    family, polarity, _ = REGISTRY[canon]
    return {
        "canonical": canon,
        "family": family,
        "polarity": polarity,
        "raw": metric_name,
    }


# Words that signal a results claim -- used by the sanitiser to find leaks in
# the intro/abstract, not for labelling.
METRIC_WORDS: set[str] = set()
for _canon, (_fam, _pol, _aliases) in REGISTRY.items():
    METRIC_WORDS.add(_canon.replace("_", " "))
    METRIC_WORDS.update(_aliases)
