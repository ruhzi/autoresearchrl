"""chat_tools(): the generic tool loop under the stage-5 root agent.

Transport is mocked at arl.openrouter._post -- these tests script the model's
side of the conversation and assert on loop mechanics: dispatch, error
feedback, termination, turn caps. No network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from arl import openrouter
from arl.openrouter import ToolError, chat_tools

TOOLS = [{"type": "function",
          "function": {"name": n, "parameters": {"type": "object"}}}
         for n in ("lookup", "submit")]


def _reply(tool_calls=None, content=""):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return {"choices": [{"message": msg}], "usage": {}}


def scripted(monkeypatch, replies):
    it = iter(replies)
    sent = []

    def fake_post(body):
        sent.append(body)
        return next(it)

    monkeypatch.setattr(openrouter, "_post", fake_post)
    return sent


def run(dispatch, **kw):
    return chat_tools("test/model", "sys", "user", TOOLS, dispatch,
                      terminal_tools={"submit"}, max_turns=kw.pop("max_turns", 5),
                      **kw)


def test_dispatch_results_are_fed_back_and_terminal_ends_loop(monkeypatch):
    sent = scripted(monkeypatch, [
        _reply([("lookup", {"q": "alpha"})]),
        _reply([("submit", {"ok": True})]),
    ])
    calls = []

    def dispatch(name, args):
        calls.append((name, args))
        return f"result-for-{args.get('q', '')}"

    final = run(dispatch)
    assert final["name"] == "submit" and final["args"] == {"ok": True}
    assert calls == [("lookup", {"q": "alpha"}), ("submit", {"ok": True})]
    # second request must carry the tool result back to the model
    msgs = sent[1]["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert tool_msgs and "result-for-alpha" in tool_msgs[0]["content"]


def test_tool_error_goes_back_to_model_not_raised(monkeypatch):
    sent = scripted(monkeypatch, [
        _reply([("submit", {})]),          # premature: dispatch refuses
        _reply([("submit", {"ok": 1})]),   # second attempt accepted
    ])
    state = {"first": True}

    def dispatch(name, args):
        if state.pop("first", False):
            raise ToolError("coverage incomplete: s3 uncovered")
        return "accepted"

    final = run(dispatch)
    assert final["args"] == {"ok": 1}
    err_msgs = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert "coverage incomplete" in err_msgs[0]["content"]


def test_malformed_arguments_become_tool_error(monkeypatch):
    bad = {"choices": [{"message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c0", "type": "function",
                        "function": {"name": "lookup",
                                     "arguments": "{not json"}}]}}],
        "usage": {}}
    sent = scripted(monkeypatch, [bad, _reply([("submit", {})])])
    final = run(lambda n, a: "ok")
    assert final["name"] == "submit"
    err = [m for m in sent[1]["messages"] if m["role"] == "tool"][0]
    assert "arguments" in err["content"].lower()


def test_text_only_reply_is_nudged(monkeypatch):
    sent = scripted(monkeypatch, [
        _reply(content="thinking out loud, no tool call"),
        _reply([("submit", {})]),
    ])
    final = run(lambda n, a: "ok")
    assert final["name"] == "submit"
    nudges = [m for m in sent[1]["messages"] if m["role"] == "user"]
    assert any("tool" in m["content"].lower() for m in nudges[1:])


def test_max_turns_exhausted_raises(monkeypatch):
    scripted(monkeypatch, [_reply([("lookup", {})])] * 3)
    with pytest.raises(RuntimeError, match="max_turns"):
        run(lambda n, a: "more", max_turns=3)


def test_parallel_tool_calls_run_concurrently_and_keep_order(monkeypatch):
    import time

    sent = scripted(monkeypatch, [
        _reply([("lookup", {"q": "a"}), ("lookup", {"q": "b"}),
                ("lookup", {"q": "c"})]),
        _reply([("submit", {})]),
    ])

    def dispatch(name, args):
        time.sleep(0.2)
        return f"r-{args.get('q', 'submit')}"

    t0 = time.monotonic()
    final = run(dispatch)
    elapsed = time.monotonic() - t0
    assert final["name"] == "submit"
    # three 0.2s dispatches sequentially would be >= 0.6s
    assert elapsed < 0.55
    # sent[1] holds a live reference to the message list, so only compare the
    # first turn's tool results (later appends are the submit's result).
    tool_msgs = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs][:3] == ["r-a", "r-b", "r-c"]


def test_trajectory_file_captures_full_conversation(monkeypatch, tmp_path):
    import config as cfg
    import json as _json
    monkeypatch.setattr(cfg, "TRAJECTORIES", True)
    monkeypatch.setattr(cfg, "TRAJ_DIR", tmp_path)
    scripted(monkeypatch, [
        _reply([("lookup", {"q": "alpha"})]),
        _reply([("submit", {"ok": True})]),
    ])
    chat_tools("test/model", "sysP", "userP", TOOLS, lambda n, a: f"res-{n}",
               terminal_tools={"submit"}, max_turns=5, traj="unit_traj")
    files = list(tmp_path.glob("*unit_traj*.json"))
    assert len(files) == 1
    d = _json.loads(files[0].read_text(encoding="utf-8"))
    roles = [m["role"] for m in d["messages"]]
    assert roles[0] == "system" and d["messages"][0]["content"] == "sysP"
    assert "tool" in roles                        # tool results captured
    tool_contents = [m["content"] for m in d["messages"] if m["role"] == "tool"]
    assert "res-lookup" in tool_contents
    assert d["final"]["name"] == "submit"


def test_result_truncation(monkeypatch):
    sent = scripted(monkeypatch, [
        _reply([("lookup", {})]),
        _reply([("submit", {})]),
    ])
    chat_tools("test/model", "sys", "user", TOOLS,
               lambda n, a: "y" * 50_000, terminal_tools={"submit"},
               max_turns=5, max_result_chars=1000)
    tool_msg = [m for m in sent[1]["messages"] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) < 1200
    assert "truncated" in tool_msg["content"]
