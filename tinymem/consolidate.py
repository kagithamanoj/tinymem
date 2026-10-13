"""Consolidation: the sleep cycle. Turn salient episodes into durable facts.

Two distillation paths. When a provider is configured, the model PROPOSES
facts and deterministic code validates them. Otherwise a rule-based
extractor runs. Either way, contradiction handling is deterministic: a new
fact that overlaps an old one supersedes it, and the old fact is kept for
provenance instead of being deleted.

The split is deliberate: the model describes, code decides.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .recall import tokenize

MIN_SALIENCE = 0.45
CONFLICT_JACCARD = 0.4

# Phrases that signal the user is revising something said before. Combined
# with topical overlap, they mark a candidate as an update, not a new fact.
UPDATE_MARKERS = ("actually", "changed my mind", "instead", "correction", "update:")

EXTRACTION_PATTERNS = [
    # Negation first: "I don't like X" must not match the "I like" rule.
    (re.compile(r"\bi\s+(?:do\s+not|don'?t)\s+(?:like|prefer|love)\s+(.+)", re.I),
     "dislikes {}"),
    (re.compile(r"\bi\s+don'?t\s+want\s+(.+)", re.I),
     "does not want {}"),
    (re.compile(r"\bi\s+(?:really\s+)?(?:prefer|love|like)\s+(.+)", re.I),
     "prefers {}"),
    (re.compile(r"\bi\s+(?:hate|dislike|can'?t stand)\s+(.+)", re.I),
     "dislikes {}"),
    (re.compile(r"\bi\s+am\s+allergic\s+to\s+(.+)", re.I),
     "allergic to {}"),
    (re.compile(r"\bmy\s+([a-z][a-z' ]{1,30}?)\s+is\s+(.+)", re.I),
     "{} is {}"),
    (re.compile(r"\bremember(?:\s+that)?\s+(.+)", re.I),
     "{}"),
]


def normalize(text):
    text = text.strip().rstrip(".").lower()
    text = re.sub(r"\s+", " ", text)
    return text


def rule_extract(text):
    """Rule-based distillation. Returns a list of candidate fact strings."""
    t = text.strip().rstrip(".")
    for pattern, template in EXTRACTION_PATTERNS:
        m = pattern.search(t)
        if m:
            groups = [g.strip().rstrip(".") for g in m.groups()]
            fact = normalize(template.format(*groups))
            if len(fact) > 3:
                return [fact]
    return []


def llm_extract(text, provider):
    """Ask the model to propose durable facts, one per line."""
    prompt = (
        "Extract durable facts about the user from this note. "
        "Output one fact per line, lowercase, no punctuation at the end. "
        "Only include things likely to matter in the future. "
        "If nothing is durable, output NOTHING.\n\nNote: " + text
    )
    try:
        out = provider.complete(prompt)
    except Exception:
        return []
    facts = []
    for line in out.splitlines():
        line = line.strip().lstrip("-*0123456789. ")
        if not line or line.upper() == "NOTHING":
            continue
        fact = normalize(line)
        if 3 < len(fact) < 200:
            facts.append(fact)
    return facts


def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_conflict(store, candidate, episode_text=""):
    """An active fact that says something different about the same thing.

    Two signals: high token overlap (same topic, different words), or
    explicit revision language plus topical overlap (the user said
    'actually...' about something on record).
    """
    marked = any(m in episode_text.lower() for m in UPDATE_MARKERS)
    cand_toks = set(tokenize(candidate))
    best, best_score = None, 0.0
    for fact in store.active_facts():
        if fact["text"] == candidate:
            return None
        overlap = len(set(tokenize(fact["text"])) & cand_toks)
        score = jaccard(fact["text"], candidate)
        if score >= CONFLICT_JACCARD or (marked and overlap >= 2):
            if score >= best_score:
                best, best_score = fact, score
    return best


def upsert_fact(store, candidate, episode, now_ts):
    """Insert, reinforce, or supersede. Returns (fact_id, action)."""
    existing = store.conn.execute(
        "SELECT id FROM facts WHERE text = ?", (candidate,)
    ).fetchone()
    if existing:
        fid, _ = store.add_fact(
            candidate, 0.5 + 0.5 * episode["salience"], now_ts
        )
        store.link_source(fid, episode["id"])
        return fid, "reinforced"

    conflict = find_conflict(store, candidate, episode["text"])
    if conflict is not None:
        store.supersede(conflict["id"], now_ts)
        fid, _ = store.add_fact(
            candidate, 0.5 + 0.5 * episode["salience"], now_ts
        )
        store.link_source(fid, episode["id"])
        return fid, "superseded #%d" % conflict["id"]

    fid, _ = store.add_fact(candidate, 0.5 + 0.5 * episode["salience"], now_ts)
    store.link_source(fid, episode["id"])
    return fid, "added"


def distill(text, provider):
    if provider is not None and provider.name != "stub":
        facts = llm_extract(text, provider)
        if facts:
            return facts
    return rule_extract(text)


def consolidate(store, provider=None, now=None, min_salience=MIN_SALIENCE):
    """Run one consolidation pass. Returns a stats dict."""
    now = now or datetime.now(timezone.utc)
    now_ts = now.isoformat()
    stats = {
        "episodes_seen": 0,
        "facts_added": 0,
        "facts_reinforced": 0,
        "superseded": 0,
    }
    for ep in store.unconsolidated():
        stats["episodes_seen"] += 1
        if ep["salience"] >= min_salience:
            for candidate in distill(ep["text"], provider):
                _, action = upsert_fact(store, candidate, ep, now_ts)
                if action == "added":
                    stats["facts_added"] += 1
                elif action == "reinforced":
                    stats["facts_reinforced"] += 1
                else:
                    stats["superseded"] += 1
        store.mark_consolidated(ep["id"])
    return stats
