"""Download and unpack arXiv e-print LaTeX source.

Why LaTeX and not PDF: tables arrive as markup instead of layout guesswork, and
\\section boundaries let us delete the results section programmatically rather
than hoping a regex scrubber catches every leaked number. Papers with no source
available are dropped -- there is no shortage of candidates.
"""
from __future__ import annotations

import gzip
import io
import tarfile
import urllib.error
from pathlib import Path

import config
from arl.util import http_get, log


def _tex_files(outdir: Path) -> list[str]:
    return sorted(
        str(p.relative_to(outdir)).replace("\\", "/")
        for p in outdir.rglob("*.tex")
    )


def fetch_one(base_id: str, outdir: Path) -> dict:
    """Fetch + unpack one paper. Returns a status record."""
    rec: dict = {"base_id": base_id}

    if outdir.is_dir() and any(outdir.iterdir()):
        rec["tex_files"] = _tex_files(outdir)
        # A cached dir with no .tex (pdf-only, or a half-finished unpack)
        # must not launder into "source available" on re-runs.
        rec["status"] = "cached" if rec["tex_files"] else "cached_no_tex"
        return rec

    url = f"https://arxiv.org/e-print/{base_id}"
    try:
        raw = http_get(url, timeout=180, retries=3, delay=5.0)
    except urllib.error.HTTPError as e:
        rec["status"] = f"http_{e.code}"
        return rec
    except Exception as e:
        rec["status"] = f"err_{type(e).__name__}"
        return rec

    rec["bytes"] = len(raw)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if raw[:5] == b"%PDF-":
            (outdir / "paper.pdf").write_bytes(raw)
            rec["status"] = "pdf_only"
        else:
            try:
                with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                    # filter="data" refuses absolute paths and ../ escapes.
                    tf.extractall(outdir, filter="data")
                rec["status"] = "tar"
            except tarfile.ReadError:
                data = gzip.decompress(raw)
                if data[:5] == b"%PDF-":
                    (outdir / "paper.pdf").write_bytes(data)
                    rec["status"] = "pdf_only"
                else:
                    (outdir / "main.tex").write_bytes(data)
                    rec["status"] = "single_tex"
    except Exception as e:
        rec["status"] = f"unpack_err_{type(e).__name__}"
        rec["error"] = str(e)[:300]
        return rec

    rec["tex_files"] = _tex_files(outdir)
    if rec["status"] in ("tar", "single_tex") and not rec["tex_files"]:
        rec["status"] = "no_tex"
    return rec
