"""A json_schema request must only reach providers that enforce it.

On 2026-08-18 every schema-constrained call to anthropic/claude-sonnet-5 came
back unparseable -- the stage-8 judge answered in prose, the stage-5 enumerator
in YAML -- while the same code had returned clean JSON four days earlier and
deepseek kept working the whole time. Cause: of the nine provider endpoints
OpenRouter lists for that slug, three (google-vertex/global, /europe, /us)
advertise `response_format` but NOT `structured_outputs`. A request landing
there has the parameter accepted and never enforced, so the model free-forms.

`provider.require_parameters` is OpenRouter's documented switch for "only use
providers that support all parameters in my request", but it cannot be used
here: NO endpoint for this slug lists `temperature`, and chat() always sends it,
so requiring every parameter matches nothing and the API returns
`HTTP 404: No endpoints found that can handle the requested parameters`.

Excluding the offender by name is the mechanism that works. A base slug matches
all of a provider's regions, so one entry covers all three Vertex endpoints.

Transport is mocked at arl.openrouter._post. These tests assert on the request
body only: no stage prompt, schema, threshold, or control flow is involved.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl import openrouter  # noqa: E402
from arl.openrouter import chat, chat_json  # noqa: E402

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


def capture(monkeypatch, content='{"ok": true}'):
    sent = []

    def fake_post(body, validate=None):
        sent.append(body)
        data = {"choices": [{"message": {"content": content},
                             "finish_reason": "stop"}], "usage": {}}
        if validate:
            validate(data)
        return data

    monkeypatch.setattr(openrouter, "_post", fake_post)
    return sent


def test_a_schema_request_excludes_providers_that_do_not_enforce_it(monkeypatch):
    sent = capture(monkeypatch)
    chat_json("anthropic/claude-sonnet-5", "sys", "user", SCHEMA)
    assert sent[0]["provider"] == {"ignore": ["google-vertex"]}


def test_the_exclusion_does_not_filter_on_parameter_support(monkeypatch):
    """require_parameters would 404: no endpoint for this slug supports the
    `temperature` that chat() always sends."""
    sent = capture(monkeypatch)
    chat_json("anthropic/claude-sonnet-5", "sys", "user", SCHEMA)
    assert "require_parameters" not in sent[0]["provider"]
    assert "temperature" in sent[0]


def test_the_schema_itself_is_still_sent_unchanged(monkeypatch):
    sent = capture(monkeypatch)
    chat_json("anthropic/claude-sonnet-5", "sys", "user", SCHEMA,
              schema_name="thing")
    assert sent[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "thing", "strict": True, "schema": SCHEMA},
    }


def test_a_request_with_no_schema_is_left_alone(monkeypatch):
    """Schemaless calls must route exactly as before -- nothing to enforce."""
    sent = capture(monkeypatch, content="plain text is fine here")
    chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])
    assert "provider" not in sent[0]
    assert "response_format" not in sent[0]


def test_nothing_else_about_the_body_changes(monkeypatch):
    sent = capture(monkeypatch)
    chat_json("anthropic/claude-sonnet-5", "sys", "user", SCHEMA,
              max_tokens=1234)
    body = sent[0]
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["max_tokens"] == 1234
    assert body["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "user"}]
    assert body["usage"] == {"include": True}
    assert set(body) == {"model", "messages", "max_tokens", "temperature",
                         "usage", "response_format", "provider"}


def test_the_parsed_result_is_returned_as_before(monkeypatch):
    capture(monkeypatch)
    res, _ = chat_json("anthropic/claude-sonnet-5", "sys", "user", SCHEMA)
    assert res == {"ok": True}
