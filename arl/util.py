"""Small shared helpers: JSON/JSONL io, logging, HTTP with retry."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import config

# Paper text is full of arrows, Greek, and CJK. On Windows the console is
# often cp1252, and one such character in a progress line kills a stage.
# Make both streams UTF-8 with replacement; output content is unchanged on
# consoles that already handle UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, AttributeError):
            pass
del _stream


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """Atomic write: temp file + rename, so a crash mid-write never leaves a
    truncated JSON behind (checkpoints and stage outputs must survive kills)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    # On Windows os.replace raises PermissionError (WinError 5/32) if an
    # antivirus scanner or indexer has the target open for a moment. Retry
    # with backoff before giving up -- one transient lock must not cost a
    # paper its stage output.
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (2 ** attempt))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def is_network_error(e: Exception) -> bool:
    """A failure that means 'the internet is gone', not 'the server said no'.

    HTTPError is a server response -- not a network error. Timeouts are a
    SLOW server (or an over-long generation), not a dead network: they must
    consume the normal retry budget instead of looping in the wait-for-
    connectivity holding pattern. Everything else in the URLError /
    ConnectionError family counts as the network being down.
    """
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, TimeoutError):          # socket.timeout is an alias
        return False
    if isinstance(e, urllib.error.URLError):
        reason = getattr(e, "reason", None)
        return not isinstance(reason, TimeoutError)
    return isinstance(e, (ConnectionError, OSError))


def http_get(url: str, timeout: int = 60, retries: int = 3, delay: float = 3.0) -> bytes:
    """GET with retry on transient failures. Raises on 4xx other than 429.

    Network outages do not count against `retries`: while the connection is
    down the call waits (up to config.NET_WAIT_S) and resumes when the
    internet returns, so an outage mid-pipeline pauses progress instead of
    destroying it.
    """
    last = None
    attempt = 0
    net_deadline = time.monotonic() + config.NET_WAIT_S
    while attempt < retries:
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            last = e
            # 404 / 403 are permanent for a given arXiv id -- don't hammer.
            if e.code in (400, 403, 404):
                raise
            log(f"  HTTP {e.code} on {url}; retry {attempt + 1}/{retries}")
        except Exception as e:
            if is_network_error(e) and time.monotonic() < net_deadline:
                log(f"  network down ({type(e).__name__}) on {url}; waiting "
                    f"30s for connectivity (does not count as a retry)")
                time.sleep(30.0)
                continue
            last = e
            log(f"  {type(e).__name__} on {url}; retry {attempt + 1}/{retries}")
        attempt += 1
        time.sleep(delay * attempt)
    raise last  # type: ignore[misc]


class SchemaError(ValueError):
    """A model response did not match the requested schema."""


_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "number": (int, float),
    "integer": int,
}


def validate_against_schema(obj: Any, schema: dict, path: str = "$") -> None:
    """Check a model response against the JSON Schema we asked for.

    Raises SchemaError on the first violation. Exists because `strict: true` on
    the provider side is not reliably enforced: a screen run returned an
    out-of-enum value and two responses missing four required keys each, and
    without this check those got written to disk as if they were verdicts.
    A missing key is a failed call, not a decision.

    Deliberately narrow -- covers the constructs our schemas actually use
    (object/array/enum/scalar types, required, additionalProperties=false).
    """
    want = schema.get("type")

    # A union type, e.g. ["object", "null"] for a field that may be absent.
    if isinstance(want, list):
        if obj is None:
            if "null" not in want:
                raise SchemaError(f"{path}: null not allowed, expected {want}")
            return
        want = next((t for t in want if t != "null"), None)

    if want and want in _JSON_TYPES:
        # bool is a subclass of int in Python; don't let True satisfy "integer"
        if want in ("number", "integer") and isinstance(obj, bool):
            raise SchemaError(f"{path}: expected {want}, got boolean")
        if not isinstance(obj, _JSON_TYPES[want]):
            raise SchemaError(
                f"{path}: expected {want}, got {type(obj).__name__}")

    if "enum" in schema and obj not in schema["enum"]:
        raise SchemaError(f"{path}: {obj!r} not in enum {schema['enum']}")

    if want == "object":
        props = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in obj]
        if missing:
            raise SchemaError(f"{path}: missing required keys {missing}")
        if schema.get("additionalProperties") is False:
            extra = [k for k in obj if k not in props and not k.startswith("_")]
            if extra:
                raise SchemaError(f"{path}: unexpected keys {extra}")
        for k, sub in props.items():
            if k in obj:
                validate_against_schema(obj[k], sub, f"{path}.{k}")

    elif want == "array" and "items" in schema:
        for i, item in enumerate(obj):
            validate_against_schema(item, schema["items"], f"{path}[{i}]")


def extract_json_block(text: str) -> Any:
    """Best-effort JSON recovery from a model response.

    Structured outputs should make this unnecessary, but not every model on
    OpenRouter honours json_schema, so we keep a fallback rather than losing
    an extraction to a stray ```json fence.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Last resort: first balanced {...} span.
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from model output: {text[:400]!r}")
