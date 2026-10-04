"""arXiv Atom API: build a candidate pool and draw a reproducible sample."""
from __future__ import annotations

import hashlib
import time
import urllib.parse
import xml.etree.ElementTree as ET

import config
from arl.util import http_get, log, read_json, write_json

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _query_string() -> str:
    cats = " OR ".join(f"cat:{c}" for c in config.CATEGORIES)
    return f"({cats}) AND submittedDate:[{config.DATE_FROM} TO {config.DATE_TO}]"


def _page(start: int) -> bytes:
    params = urllib.parse.urlencode(
        {
            "search_query": _query_string(),
            "start": start,
            "max_results": config.POOL_PAGE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    return http_get("http://export.arxiv.org/api/query?" + params, timeout=60)


def _parse_entry(e) -> dict:
    aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
    prim = e.find("arxiv:primary_category", NS)
    return {
        "id": aid,
        "base_id": aid.split("v")[0],
        "title": " ".join(e.find("a:title", NS).text.split()),
        "published": e.find("a:published", NS).text,
        "updated": e.find("a:updated", NS).text,
        "primary_cat": prim.get("term") if prim is not None else None,
        "cats": [c.get("term") for c in e.findall("a:category", NS)],
        "abstract": " ".join(e.find("a:summary", NS).text.split()),
        "authors": [
            a.find("a:name", NS).text for a in e.findall("a:author", NS)
        ][:20],
    }


def build_pool() -> list[dict]:
    """Page through the API until POOL_TARGET distinct papers are collected.

    Every fetched page is checkpointed to POOL_PARTIAL_JSON, so an outage
    mid-build resumes from the last completed page instead of restarting.
    The caller deletes the checkpoint once the final pool.json is written.
    """
    partial = read_json(config.POOL_PARTIAL_JSON, default=None)
    if partial:
        pool: list[dict] = partial["pool"]
        start: int = partial["start"]
        log(f"pool: resuming from checkpoint -- {len(pool)} papers, "
            f"next page start={start}")
    else:
        pool = []
        start = 0
    seen: set[str] = {p["base_id"] for p in pool}

    while len(pool) < config.POOL_TARGET:
        root = ET.fromstring(_page(start))
        entries = root.findall("a:entry", NS)
        if not entries:
            log(f"pool: API returned no entries at start={start}; stopping early")
            break
        for e in entries:
            rec = _parse_entry(e)
            if rec["base_id"] in seen:
                continue
            seen.add(rec["base_id"])
            pool.append(rec)
        start += config.POOL_PAGE
        write_json(config.POOL_PARTIAL_JSON, {"pool": pool, "start": start})
        log(f"pool: start={start - config.POOL_PAGE} -> {len(pool)} papers "
            f"(checkpointed)")
        time.sleep(config.API_DELAY_S)
    return pool


def _draw_key(base_id: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{base_id}".encode("utf-8")).hexdigest()


def draw_sample(pool: list[dict], n: int, seed: int,
                keep: list[dict] | None = None) -> list[dict]:
    """Reproducible draw that GROWS the corpus and never redraws it.

    The order is per-paper: md5(seed:base_id). A paper's position depends on
    its own id and nothing else, so refreshing the pool INSERTS new papers
    rather than reordering the ones already processed. The previous seeded
    shuffle was stable across n but not across the pool -- Fisher-Yates
    consumes the list length, so a single inserted paper re-randomised
    everything and kept 3 of 25 already-processed papers.

    `keep` is the previous draw (data/sample.json). Those papers are always
    in the result, first and in their stored form, even when a moved query
    window has dropped them from the pool: every one has already been
    fetched, screened, enumerated, extracted, composed and verified, and
    dropping it orphans its samples. So n is a floor on work already drawn,
    not a cap -- lowering SAMPLE_SIZE cannot delete a processed paper.
    """
    kept = list(keep or [])
    seen = {r["base_id"] for r in kept}
    fresh = sorted((p for p in pool if p["base_id"] not in seen),
                   key=lambda p: _draw_key(p["base_id"], seed))
    return kept + fresh[:max(0, n - len(kept))]
