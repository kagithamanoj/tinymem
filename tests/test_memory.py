"""Tests for tinymem. Everything runs offline against :memory: stores."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinymem import Memory
from tinymem import salience
from tinymem.consolidate import jaccard, rule_extract
from tinymem.providers import StubProvider
from tinymem.recall import BM25, tokenize

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mem():
    return Memory(path=":memory:", provider=StubProvider())


# -- salience ------------------------------------------------------------


def test_preferences_outscore_smalltalk():
    assert salience.score("I prefer aisle seats") > salience.score(
        "traffic was bad today"
    )


def test_explicit_remember_boosts():
    assert salience.score("remember that my passport expires in May") > 0.5


def test_questions_score_lower():
    assert salience.score("what time is it?") < salience.score("my flight is at 6")


def test_score_is_bounded():
    for text in ["hi", "I LOVE EVERYTHING " * 50, "remember " * 20]:
        assert 0.05 <= salience.score(text) <= 1.0


# -- tokenize / BM25 -----------------------------------------------------


def test_tokenize_drops_stopwords():
    toks = tokenize("The quick brown fox")
    assert "the" not in toks and "quick" in toks


def test_bm25_ranks_relevant_first():
    bm = BM25([("a", "I prefer aisle seats"), ("b", "weather was nice")])
    scores = bm.score("aisle seats")
    assert scores["a"] > scores["b"]


# -- extraction ----------------------------------------------------------


def test_rule_extract_preference():
    assert rule_extract("I prefer aisle seats on flights") == ["prefers aisle seats on flights"]


def test_rule_extract_allergy():
    assert rule_extract("I am allergic to peanuts") == ["allergic to peanuts"]


def test_rule_extract_my_x_is_y():
    assert rule_extract("My home airport is DFW") == ["home airport is dfw"]


def test_rule_extract_nothing_durable():
    assert rule_extract("traffic was bad today") == []


# -- memory facade -------------------------------------------------------


def test_remember_and_recall_roundtrip():
    m = mem()
    m.remember("I prefer aisle seats")
    results = m.recall("aisle or window?")
    assert results and "aisle" in results[0]["text"].lower()


def test_consolidation_creates_facts():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    stats = m.consolidate(now=BASE)
    assert stats["facts_added"] == 1
    assert any("aisle" in f["text"] for f in m.store.active_facts())


def test_contradiction_supersedes():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.remember(
        "Actually I prefer window seats now", ts=(BASE + timedelta(days=1)).isoformat()
    )
    m.consolidate(now=BASE + timedelta(days=2))
    facts = [f["text"] for f in m.store.active_facts()]
    assert any("window" in f for f in facts)
    assert not any("aisle" in f for f in facts)


def test_reinforcement_does_not_duplicate():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.remember("I really prefer aisle seats", ts=(BASE + timedelta(days=1)).isoformat())
    m.consolidate(now=BASE + timedelta(days=2))
    facts = [f["text"] for f in m.store.active_facts() if "aisle" in f["text"]]
    assert len(facts) == 1


def test_recall_prefers_facts_over_noise():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.remember("the weather was nice today", ts=BASE.isoformat())
    m.consolidate(now=BASE)
    results = m.recall("what seat do I prefer?", now=BASE, k=3)
    assert results[0]["kind"] == "fact"


def test_forget_removes_records():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.consolidate(now=BASE)
    n = m.forget("aisle")
    assert n >= 2  # episode + fact
    assert m.recall("aisle") == [] or all(
        "aisle" not in r["text"].lower() for r in m.recall("aisle")
    )


def test_decay_fades_old_episodes():
    m = mem()
    old = (BASE - timedelta(days=200)).isoformat()
    eid, _ = m.remember("traffic was bad that day", ts=old)
    before = m.store.conn.execute(
        "SELECT salience FROM episodes WHERE id=?", (eid,)
    ).fetchone()["salience"]
    m.decay(now=BASE)
    after = m.store.conn.execute(
        "SELECT salience FROM episodes WHERE id=?", (eid,)
    ).fetchone()["salience"]
    assert after < before


def test_empty_remember_rejected():
    with pytest.raises(ValueError):
        mem().remember("   ")


def test_jaccard():
    assert jaccard("prefers aisle seats", "prefers window seats") > 0.3
    assert jaccard("prefers aisle seats", "weather was nice") == 0.0


def test_cli_roundtrip(tmp_path, capsys):
    from tinymem.cli import main

    db = str(tmp_path / "t.db")
    main(["--db", db, "remember", "I prefer aisle seats"])
    main(["--db", db, "recall", "aisle or window?"])
    out = capsys.readouterr().out
    assert "aisle" in out.lower()


# -- v2: negation, identity, history, hybrid, confidence ------------------


def test_negation_extraction():
    from tinymem.consolidate import rule_extract

    assert rule_extract("I don't like middle seats") == ["dislikes middle seats"]
    assert rule_extract("I do not prefer red-eyes") == ["dislikes red-eyes"]


def test_reported_speech_not_extracted():
    from tinymem.consolidate import rule_extract

    assert rule_extract("My friend Tom said he loves red-eye flights") == []
    assert rule_extract("Priya loves window seats") == []


def test_identity_salience_reaches_threshold():
    assert salience.score("My home airport is DFW") >= 0.45
    assert salience.score("My manager's name is Dana") >= 0.45


def test_history_shows_superseded():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.remember(
        "Actually I prefer window seats now",
        ts=(BASE + timedelta(days=1)).isoformat(),
    )
    m.consolidate(now=BASE + timedelta(days=2))
    rows = m.history("seat")
    statuses = {r["status"] for r in rows}
    assert "superseded" in statuses and "active" in statuses
    assert any("aisle" in r["text"] for r in rows if r["status"] == "superseded")


def test_reinforcement_links_sources():
    m = mem()
    m.remember("My home airport is DFW", ts=BASE.isoformat())
    m.remember(
        "My home airport is DFW", ts=(BASE + timedelta(days=1)).isoformat()
    )
    m.consolidate(now=BASE + timedelta(days=2))
    rows = m.store.facts_matching("home airport is dfw", include_superseded=False)
    assert len(rows) == 1
    assert len(m.store.episodes_for_fact(rows[0]["id"])) == 2


def test_hybrid_mode_agrees_with_bm25():
    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.remember("My home airport is DFW", ts=BASE.isoformat())
    m.consolidate(now=BASE)
    bm25 = m.recall("which airport?", k=1, now=BASE, mode="bm25")
    hybrid = m.recall("which airport?", k=1, now=BASE, mode="hybrid")
    assert bm25[0]["text"] == hybrid[0]["text"]


def test_hashed_embeddings_deterministic_and_normalized():
    from tinymem.providers import HashedEmbeddingProvider
    import math

    ep = HashedEmbeddingProvider()
    a = ep.embed(["I prefer aisle seats"])
    b = ep.embed(["I prefer aisle seats"])
    assert a == b
    assert abs(math.sqrt(sum(x * x for x in a[0])) - 1.0) < 1e-9


def test_not_confident_on_irrelevant():
    from tinymem.recall import is_confident

    m = mem()
    m.remember("I prefer aisle seats", ts=BASE.isoformat())
    m.consolidate(now=BASE)
    results = m.recall("what is my mother's maiden name?", now=BASE)
    assert not m.confident(results)
    assert not is_confident(results)


def test_confident_on_relevant():
    m = mem()
    m.remember("My home airport is DFW", ts=BASE.isoformat())
    m.consolidate(now=BASE)
    results = m.recall("which airport do I fly from?", now=BASE)
    assert m.confident(results)


def test_stemmer_unifies_inflections():
    from tinymem.recall import tokenize

    assert tokenize("allergies") == tokenize("allergic")
    assert tokenize("seats") == tokenize("seat")
    assert tokenize("prefers") == tokenize("prefer")
