"""Minimal OpenRouter client (stdlib only).

OpenRouter speaks the OpenAI chat-completions shape, so this is a thin POST
wrapper: retries on transient failures, structured output via json_schema, and
per-call token/cost accounting so the dataset build has a known price.

If you later move to the Anthropic API directly, this is the only file that
changes -- callers just need `chat_json()` to keep returning (obj, usage).
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import urllib.error
import urllib.request

import config
from arl.util import extract_json_block, is_network_error, log

_lock = threading.Lock()
_TOTALS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0,
           "errors": 0}

# ------------------------------------------------------- trajectory logging
#
# One JSON file per LLM interaction: the complete message list exactly as the
# model saw it (system, user, its own tool calls, every tool result) plus the
# final outcome and usage. This is the audit trail for "what did my agent
# see, and what did it do" -- written even when the interaction fails.

_traj_seq = 0


def _write_traj(label: str, payload: dict) -> None:
    import config as _cfg
    if not _cfg.TRAJECTORIES:
        return
    global _traj_seq
    with _lock:
        _traj_seq += 1
        seq = _traj_seq
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label or "call")[:80]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    from arl.util import write_json
    write_json(_cfg.TRAJ_DIR / f"{stamp}_{seq:05d}_{safe}.json", payload)


def totals() -> dict:
    with _lock:
        return dict(_TOTALS)


def _record(usage: dict) -> None:
    with _lock:
        _TOTALS["calls"] += 1
        _TOTALS["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        _TOTALS["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        try:
            _TOTALS["cost"] += float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ruhzi/autoresearchrl",
        "X-Title": "autoresearchrl dataset build",
    }


def list_models() -> list[dict]:
    """Fetch the OpenRouter model catalogue (no auth strictly required)."""
    req = urllib.request.Request(
        config.OPENROUTER_MODELS_URL,
        headers={"User-Agent": config.USER_AGENT},
    )
    raw = urllib.request.urlopen(req, timeout=60).read()
    return json.loads(raw).get("data", [])


class PermanentError(RuntimeError):
    """A response that will not improve on retry (e.g. a deterministic
    truncation at temperature 0). _post raises it immediately instead of
    burning the retry budget."""


def _post(body: dict, validate=None) -> dict:
    """POST one chat-completion body with retries; return the parsed response.

    `validate(data)` may raise to force a retry (e.g. empty completion).
    Usage is recorded before validation, matching the old behavior where a
    retried empty completion still counted toward totals.
    """
    model = body.get("model", "?")
    payload = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None

    attempt = 0
    net_deadline = time.monotonic() + config.NET_WAIT_S
    while attempt < config.MAX_RETRIES:
        req = urllib.request.Request(
            config.OPENROUTER_URL, data=payload, headers=_headers(), method="POST"
        )
        try:
            raw = urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT_S).read()
            data = json.loads(raw)

            if "error" in data and not data.get("choices"):
                msg = json.dumps(data["error"])[:300]
                raise RuntimeError(f"OpenRouter error: {msg}")

            _record(data.get("usage") or {})
            if validate is not None:
                validate(data)
            return data

        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            if e.code == 402:
                # Out of credits. Retrying cannot fix an empty wallet; fail
                # fast so the run stops cleanly (everything done so far is
                # on disk and resumes after a top-up).
                with _lock:
                    _TOTALS["errors"] += 1
                raise last_err
            if e.code in (400, 401, 403, 404) and "rate" not in detail.lower():
                # Permanent: bad key, bad model slug, malformed schema.
                with _lock:
                    _TOTALS["errors"] += 1
                raise last_err
            log(f"  {model}: HTTP {e.code}, retry {attempt + 1}/{config.MAX_RETRIES}")
        except PermanentError:
            with _lock:
                _TOTALS["errors"] += 1
            raise
        except Exception as e:
            # A dead connection pauses the pipeline instead of failing it:
            # outages don't consume retries, they wait for the net to return
            # (bounded by NET_WAIT_S). Progress already on disk is safe either
            # way; this keeps in-flight work alive too.
            if is_network_error(e) and time.monotonic() < net_deadline:
                log(f"  {model}: network down ({type(e).__name__}); waiting "
                    f"30s for connectivity (does not count as a retry)")
                time.sleep(30.0)
                continue
            last_err = e
            log(f"  {model}: {type(e).__name__}: {str(e)[:160]}; "
                f"retry {attempt + 1}/{config.MAX_RETRIES}")

        attempt += 1
        time.sleep(min(2 ** attempt * 3, 45))

    with _lock:
        _TOTALS["errors"] += 1
    raise last_err  # type: ignore[misc]


def chat_raw(
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float | None = None,
    reasoning: dict | None = None,
    extra_body: dict | None = None,
    traj: str = "",
) -> dict:
    """One completion, returning the WHOLE response object.

    `chat()` throws away everything but the text and usage. The eval harness
    needs the rest -- finish_reason, the provider that actually served the
    call, any reasoning trace -- and it must keep a truncated completion as
    data rather than raising on it, so no `validate` is passed here.

    `extra_body` is merged last, which is how the harness pins `plugins: []`
    to keep web search off.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "usage": {"include": True},
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    if extra_body:
        body.update(extra_body)

    try:
        data = _post(body)
    except Exception as e:
        _write_traj(traj or model, {
            "kind": "chat_raw", "model": model, "messages": messages,
            "error": f"{type(e).__name__}: {str(e)[:300]}"})
        raise
    _write_traj(traj or model, {
        "kind": "chat_raw", "model": model, "messages": messages,
        "response": data})
    return data


def chat(
    model: str,
    messages: list[dict],
    schema: dict | None = None,
    schema_name: str = "result",
    max_tokens: int = 8192,
    temperature: float | None = None,
    reasoning: dict | None = None,
    traj: str = "",
) -> tuple[str, dict]:
    """Single chat completion. Returns (text, usage)."""
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "usage": {"include": True},
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        # Keep the request away from providers that accept `response_format`
        # and do not enforce it. OpenRouter routes to any provider advertising
        # the parameter and lets it ignore what it has not implemented: of the
        # nine endpoints for anthropic/claude-sonnet-5, google-vertex/global,
        # /europe and /us advertise `response_format` WITHOUT
        # `structured_outputs`. Landing there on 2026-08-18 turned every schema
        # call unparseable -- the stage-8 judge answered in prose, stage 5 in
        # YAML, ~$2.80 discarded -- while deepseek (a different endpoint set)
        # kept working. No prompt states the output format, so this parameter
        # is the only thing holding it.
        #
        # A base slug covers every region of that provider, which is why one
        # entry is enough. `require_parameters: True` is the tidier switch and
        # CANNOT be used: no endpoint for this slug lists `temperature`, which
        # is always sent below, so requiring every parameter matches nothing
        # and the API answers `404 No endpoints found that can handle the
        # requested parameters`.
        body["provider"] = {"ignore": ["google-vertex"]}

    def validate(data: dict) -> None:
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "length":
            # Budget exhausted -- whether mid-output or entirely on hidden
            # reasoning (empty content). At temperature 0 a retry does the
            # same thing, so fail fast; callers (e.g. the RLM root) react
            # by splitting the work instead.
            raise PermanentError(
                f"completion budget exhausted at max_tokens "
                f"({len(text)} chars of content)")
        if not text.strip():
            raise RuntimeError(
                f"empty completion (finish_reason={choice.get('finish_reason')})"
            )

    try:
        data = _post(body, validate=validate)
    except Exception as e:
        _write_traj(traj or model, {
            "kind": "chat", "model": model, "messages": messages,
            "error": f"{type(e).__name__}: {str(e)[:300]}"})
        raise
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    _write_traj(traj or model, {
        "kind": "chat", "model": model, "messages": messages,
        "response": text, "usage": usage})
    return text, usage


# ------------------------------------------------------------------ tool loop


class ToolError(Exception):
    """A tool-level refusal or bad call. Not an exception in the Python sense
    of something going wrong -- it is fed back to the model as the tool's
    result so the model can correct itself (fix arguments, cover the gap,
    then retry). Transport failures still raise normally."""


def chat_tools(
    model: str,
    system: str,
    user: str,
    tools: list[dict],
    dispatch,
    terminal_tools: frozenset | set = frozenset(),
    max_turns: int = 24,
    max_tokens: int = 4096,
    temperature: float | None = None,
    max_result_chars: int | None = None,
    reasoning: dict | None = None,
    traj: str = "",
) -> dict:
    """Agentic tool loop: call tools via `dispatch(name, args) -> str` until
    the model successfully calls a tool named in `terminal_tools`.

    Returns {"name", "args", "result"} of the terminal call. A ToolError from
    dispatch -- including a refused terminal call, e.g. a coverage gate --
    becomes the tool result and the loop continues, so the model always gets a
    chance to repair. Raises RuntimeError when max_turns runs out.

    The complete conversation -- system, user, every assistant tool call,
    every tool result -- is written as a trajectory file at the end, success
    or failure.
    """
    if temperature is None:
        temperature = config.LLM_TEMPERATURE
    if max_result_chars is None:
        max_result_chars = config.TOOL_RESULT_MAX_CHARS

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Summed over every turn of the loop so the trajectory file carries the
    # loop's real price. Without this the agentic stages (5 root, 6) are
    # invisible in any cost audit of data/trajectories/.
    loop_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "cost": 0.0}

    def finish(final=None, error: str = ""):
        _write_traj(traj or model, {
            "kind": "tool_loop", "model": model, "tools": tools,
            "messages": messages, "usage": loop_usage,
            "final": final, "error": error})

    for _turn in range(max_turns):
        body: dict = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
        }
        if reasoning is not None:
            body["reasoning"] = reasoning
        try:
            data = _post(body)
        except Exception as e:
            finish(error=f"{type(e).__name__}: {str(e)[:300]}")
            raise
        u = data.get("usage") or {}
        loop_usage["calls"] += 1
        loop_usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        loop_usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        try:
            loop_usage["cost"] += float(u.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        assistant: dict = {"role": "assistant",
                           "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)

        if not tool_calls:
            names = ", ".join(sorted(terminal_tools)) or "a terminal tool"
            messages.append({
                "role": "user",
                "content": f"Respond only with tool calls, never plain text. "
                           f"Continue with your tools and finish by calling "
                           f"{names}.",
            })
            continue

        def run_call(tc: dict) -> tuple[dict, str, dict | None, bool]:
            """-> (tool_call, result_text, parsed_args, dispatch_succeeded)"""
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("top-level JSON value is not an object")
            except Exception as e:
                return tc, (f"ERROR: could not parse tool arguments as a "
                            f"JSON object: {e}"), None, False
            try:
                return tc, str(dispatch(name, args)), args, True
            except ToolError as e:
                return tc, f"ERROR: {e}", args, False

        # One assistant turn may fan out several calls (e.g. multiple
        # enumerator batches); those are LLM-bound, so run them concurrently.
        # Results are appended in call order regardless of completion order.
        if len(tool_calls) > 1:
            with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                outcomes = list(pool.map(run_call, tool_calls))
        else:
            outcomes = [run_call(tool_calls[0])]

        final: dict | None = None
        for tc, result, args, ok in outcomes:
            name = (tc.get("function") or {}).get("name") or ""
            if ok and name in terminal_tools and final is None:
                final = {"name": name, "args": args, "result": result}
            if max_result_chars and len(result) > max_result_chars:
                result = (result[:max_result_chars]
                          + f"\n[... truncated at {max_result_chars} chars]")
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": result})
        if final is not None:
            finish(final=final)
            return final

    finish(error=f"max_turns={max_turns} exhausted")
    raise RuntimeError(
        f"chat_tools: max_turns={max_turns} exhausted without a successful "
        f"terminal tool call ({', '.join(sorted(terminal_tools))})")


def chat_json(
    model: str,
    system: str,
    user: str,
    schema: dict,
    schema_name: str = "result",
    max_tokens: int = 8192,
    reasoning: dict | None = None,
    traj: str = "",
) -> tuple[dict, dict]:
    """Chat call that must return an object matching `schema`."""
    text, usage = chat(
        model,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        schema=schema,
        schema_name=schema_name,
        max_tokens=max_tokens,
        reasoning=reasoning,
        traj=traj,
    )
    return extract_json_block(text), usage
