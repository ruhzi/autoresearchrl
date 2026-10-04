"""PaperStore: the paper as an environment variable, never as prompt filler.

These tests run against a synthetic parsed-JSON fixture shaped exactly like
stage 3's output (sections[{level,title,kind,chars,body}], tables[{env,
caption,label,body}]), so they need no network and no real paper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from arl.paperstore import PaperStore

PARSED = {
    "ok": True,
    "abstract": "We study widgets.",
    "sections": [
        {"level": 1, "title": "Introduction", "kind": "intro",
         "chars": 30, "body": "Widgets are Important.  Very\nimportant, even."},
        {"level": 1, "title": "Method", "kind": "method",
         "chars": 40, "body": "We propose GADGET, a widget with a gating "
                              "mechanism g(x) = \\sigma(Wx)."},
        {"level": 1, "title": "Experiments", "kind": "results",
         "chars": 20, "body": "See Table~\\ref{tab:main} for results."},
    ],
    "n_tables": 2,
    "tables": [
        {"env": "table", "caption": "Main results on WidgetBench.",
         "label": "tab:main", "body": "baseline & 1.0 \\\\ GADGET & 2.0 \\\\"},
        {"env": "table*", "caption": "Ablating the gating mechanism.",
         "label": "tab:ablate", "body": "w/o gating & 1.5 \\\\ full & 2.0 \\\\"},
    ],
}


@pytest.fixture()
def store():
    return PaperStore(PARSED)


# ------------------------------------------------------------------ identity

def test_ids_are_stable_and_ordered(store):
    assert store.ids() == ["s0", "s1", "s2", "t0", "t1"]


def test_body_returns_exact_source(store):
    assert store.body("s1") == PARSED["sections"][1]["body"]
    assert store.body("t0") == PARSED["tables"][0]["body"]


def test_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.body("s99")
    with pytest.raises(KeyError):
        store.peek("x1")


def test_total_chars_counts_sections_and_tables(store):
    expected = sum(len(s["body"]) for s in PARSED["sections"]) + \
        sum(len(t["body"]) for t in PARSED["tables"])
    assert store.total_chars == expected


def test_chars_per_target(store):
    assert store.chars("s0") == len(PARSED["sections"][0]["body"])
    assert store.chars("t1") == len(PARSED["tables"][1]["body"])


# ------------------------------------------------------------------- outline

def test_outline_lists_every_target_with_kind_title_and_size(store):
    out = store.outline()
    for tid in store.ids():
        assert tid in out
    assert "intro" in out and "method" in out and "results" in out
    assert "Introduction" in out and "Method" in out
    assert "tab:main" in out and "tab:ablate" in out
    # captions appear (possibly truncated)
    assert "Main results on WidgetBench." in out
    # sizes appear so the root can budget batches
    assert str(len(PARSED["sections"][1]["body"])) in out


def test_outline_truncates_long_captions():
    parsed = dict(PARSED)
    parsed["tables"] = [dict(PARSED["tables"][0], caption="C" * 500)]
    out = PaperStore(parsed).outline()
    assert "C" * 121 not in out


# ---------------------------------------------------------------------- peek

def test_peek_returns_prefix_with_position_header(store):
    got = store.peek("s1")
    assert PARSED["sections"][1]["body"] in got     # small body: all of it
    assert "of " + str(len(PARSED["sections"][1]["body"])) in got


def test_peek_pages_with_offset():
    body = "x" * 5000
    parsed = dict(PARSED)
    parsed["sections"] = [dict(PARSED["sections"][0], body=body)]
    st = PaperStore(parsed, peek_chars=2000)
    first = st.peek("s0")
    assert "x" * 2000 in first and "x" * 2001 not in first
    assert "0..2000 of 5000" in first
    mid = st.peek("s0", offset=4500)
    assert "x" * 500 in mid and "x" * 501 not in mid
    assert "4500..5000 of 5000" in mid


def test_peek_offset_past_end_is_empty_not_error():
    st = PaperStore(PARSED)
    got = st.peek("s0", offset=10_000)
    assert "of " in got  # header still present, no exception


# ---------------------------------------------------------------------- grep

def test_grep_finds_literal_matches_with_target_and_context(store):
    hits = store.grep("gating mechanism")
    targets = {h["target"] for h in hits}
    assert "s1" in targets           # method body
    assert "t1" in targets           # ablation caption
    for h in hits:
        assert "gating mechanism" in h["context"].lower()


def test_grep_is_case_insensitive(store):
    assert store.grep("GATING MECHANISM")


def test_grep_normalizes_whitespace(store):
    # body has "Very\nimportant"; a single-space query must still hit
    hits = store.grep("very important")
    assert any(h["target"] == "s0" for h in hits)


def test_grep_no_hits_returns_empty_list(store):
    assert store.grep("no such phrase anywhere") == []


def test_grep_caps_hits():
    parsed = dict(PARSED)
    parsed["sections"] = [
        {"level": 1, "title": f"S{i}", "kind": "other", "chars": 10,
         "body": "needle haystack " * 3}
        for i in range(30)
    ]
    st = PaperStore(parsed, grep_max_hits=5)
    assert len(st.grep("needle")) == 5


def test_grep_regex_metachars_are_literal(store):
    # LaTeX is full of regex metachars; grep must treat them literally
    hits = store.grep("g(x) = \\sigma(Wx)")
    assert any(h["target"] == "s1" for h in hits)


def test_grep_body_offsets_index_the_body_itself():
    # captioned tables used to shift body offsets by len(caption)+1,
    # making every anchor copied from a grep hit unresolvable
    st = PaperStore(PARSED, grep_context_chars=10)
    hits = [h for h in st.grep("w/o gating") if not h.get("in_caption")]
    assert hits
    h = hits[0]
    body = st.body(h["target"])
    assert body[h["offset"]:h["offset"] + 3] == "w/o"


def test_grep_caption_hits_are_marked():
    st = PaperStore(PARSED)
    hits = st.grep("Ablating the gating")
    assert hits and all(h.get("in_caption") for h in hits)


# --------------------------------------------------------------------- slice

from arl.paperstore import AnchorError  # noqa: E402


def test_slice_returns_exact_source_bytes(store):
    body = PARSED["sections"][1]["body"]
    got = store.slice({"target": "s1", "start_anchor": "We propose GADGET",
                       "end_anchor": "\\sigma(Wx)."})
    assert got == body                    # anchors at both extremes: whole body
    assert got is not body or got == body  # exact equality, not normalization


def test_slice_midspan(store):
    got = store.slice({"target": "s0", "start_anchor": "Very",
                       "end_anchor": "even."})
    assert got == "Very\nimportant, even."   # newline preserved verbatim


def test_slice_whitespace_tolerant_anchor(store):
    # anchor written with single spaces still matches the line-wrapped source
    got = store.slice({"target": "s0", "start_anchor": "Very important,",
                       "end_anchor": "even."})
    assert got == "Very\nimportant, even."


def test_slice_anchor_not_found(store):
    with pytest.raises(AnchorError, match="not found"):
        store.slice({"target": "s0", "start_anchor": "no such text",
                     "end_anchor": "even."})


def test_slice_anchor_ambiguous():
    parsed = dict(PARSED)
    parsed["sections"] = [{"level": 1, "title": "S", "kind": "other",
                           "chars": 0, "body": "alpha beta alpha beta"}]
    st = PaperStore(parsed)
    with pytest.raises(AnchorError, match="2 times"):
        st.slice({"target": "s0", "start_anchor": "alpha",
                  "end_anchor": "beta"})


def test_slice_end_before_start(store):
    with pytest.raises(AnchorError, match="before"):
        store.slice({"target": "s0", "start_anchor": "even.",
                     "end_anchor": "Very"})


def test_slice_table_row(store):
    got = store.slice({"target": "t1", "start_anchor": "w/o gating",
                       "end_anchor": "1.5 \\\\"})
    assert got == "w/o gating & 1.5 \\\\"


def test_slice_unknown_target(store):
    with pytest.raises(KeyError):
        store.slice({"target": "s9", "start_anchor": "a", "end_anchor": "b"})
