"""Stage 1: the draw must grow the corpus, never redraw it.

The old draw shuffled the whole pool and took the first n. That is stable
across SAMPLE_SIZE but NOT across the pool: inserting one paper re-randomises
a Fisher-Yates shuffle, so a refreshed pool kept 3 of 25 already-processed
papers. Every paper it drops has already been fetched, screened, enumerated,
extracted, composed and verified -- the expensive stages -- so a redraw
throws money away and orphans the corpus.

Two properties fix that, and both are tested here:

  - the order depends on each paper's own id, not on the list, so pool growth
    inserts instead of reordering;
  - papers already drawn are carried in a manifest and always kept, which
    makes growth monotone whatever happens to the query window.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arl.arxiv_api import draw_sample  # noqa: E402

SEED = 20260812


def pool(n, prefix="2608"):
    """n minimal pool records with distinct, sortable base_ids."""
    return [{"base_id": f"{prefix}.{10000 + i:05d}", "title": f"paper {i}"}
            for i in range(n)]


def ids(records):
    return [r["base_id"] for r in records]


# ------------------------------------------------- stability across n

def test_raising_sample_size_keeps_every_paper_already_drawn():
    p = pool(500)
    assert set(ids(draw_sample(p, 25, SEED))) <= set(ids(draw_sample(p, 33, SEED)))


def test_draw_returns_exactly_n_papers():
    assert len(draw_sample(pool(500), 33, SEED)) == 33


def test_draw_returns_the_whole_pool_when_n_exceeds_it():
    assert len(draw_sample(pool(10), 33, SEED)) == 10


# ------------------------------------------------- stability across the pool

def test_pool_growth_keeps_every_paper_already_drawn():
    """The old shuffle kept 3 of 25 here. Insertion must not reorder."""
    p = pool(1200)
    before = set(ids(draw_sample(p, 25, SEED)))
    grown = p + pool(120, prefix="2609")
    assert before <= set(ids(draw_sample(grown, 33, SEED)))


def test_one_inserted_paper_does_not_redraw():
    p = pool(1200)
    before = set(ids(draw_sample(p, 25, SEED)))
    grown = p + [{"base_id": "2609.00001", "title": "brand new"}]
    assert before <= set(ids(draw_sample(grown, 25, SEED)))


def test_draw_ignores_the_order_the_pool_arrives_in():
    p = pool(300)
    assert ids(draw_sample(p, 20, SEED)) == ids(draw_sample(p[::-1], 20, SEED))


def test_a_different_seed_draws_a_different_sample():
    p = pool(300)
    assert ids(draw_sample(p, 20, SEED)) != ids(draw_sample(p, 20, SEED + 1))


def test_the_draw_is_not_just_the_first_n_by_id():
    p = pool(300)
    assert ids(draw_sample(p, 20, SEED)) != ids(p[:20])


# ------------------------------------------------- the manifest

def test_a_kept_paper_missing_from_the_pool_is_still_drawn():
    """The window moved and an already-processed paper left the pool. It has
    been paid for; dropping it would orphan its samples."""
    done = [{"base_id": "2607.99999", "title": "already processed"}]
    out = draw_sample(pool(500), 25, SEED, keep=done)
    assert "2607.99999" in ids(out)
    assert len(out) == 25


def test_a_kept_paper_still_in_the_pool_is_not_duplicated():
    p = pool(500)
    out = draw_sample(p, 25, SEED, keep=draw_sample(p, 10, SEED))
    assert len(ids(out)) == len(set(ids(out))) == 25


def test_the_draw_never_shrinks_below_what_was_already_drawn():
    """Lowering SAMPLE_SIZE must not delete processed papers."""
    done = pool(30, prefix="2607")
    out = draw_sample(pool(500), 25, SEED, keep=done)
    assert set(ids(done)) <= set(ids(out))
    assert len(out) == 30


def test_kept_papers_keep_their_stored_record():
    """A kept paper absent from the pool can only come from the manifest, so
    its metadata must be carried through, not reconstructed."""
    done = [{"base_id": "2607.99999", "title": "the real title",
             "primary_cat": "cs.LG"}]
    out = draw_sample(pool(500), 25, SEED, keep=done)
    got = next(r for r in out if r["base_id"] == "2607.99999")
    assert got["title"] == "the real title"
    assert got["primary_cat"] == "cs.LG"


def test_kept_papers_come_first_so_new_work_is_visible():
    done = [{"base_id": "2607.99999", "title": "already processed"}]
    out = draw_sample(pool(500), 25, SEED, keep=done)
    assert ids(out)[0] == "2607.99999"


def test_an_empty_manifest_behaves_like_no_manifest():
    p = pool(500)
    assert ids(draw_sample(p, 25, SEED, keep=[])) == ids(draw_sample(p, 25, SEED))
