"""Memory: the facade. Ten lines to give any agent a memory."""

from __future__ import annotations

from datetime import datetime, timezone

from . import salience
from .consolidate import consolidate
from .decay import apply_decay
from .providers import default_provider
from .recall import recall as hybrid_recall
from .store import Store

DEFAULT_PATH = "~/.tinymem/memory.db"


class Memory:
    def __init__(self, path=DEFAULT_PATH, provider=None):
        self.store = Store(path)
        self.provider = provider if provider is not None else default_provider()

    def remember(self, text, kind="note", ts=None):
        """Store an episode. Returns (episode_id, salience)."""
        text = text.strip()
        if not text:
            raise ValueError("cannot remember empty text")
        ts = ts or datetime.now(timezone.utc).isoformat()
        s = salience.score(text)
        eid = self.store.add_episode(ts, kind, text, s)
        return eid, round(s, 3)

    def recall(self, query, k=5, now=None):
        """Hybrid recall. Returns ranked dicts with text, score, provenance."""
        return hybrid_recall(self.store, query, k=k, now=now)

    def consolidate(self, now=None):
        """Run the sleep cycle: episodes become facts."""
        return consolidate(self.store, self.provider, now=now)

    def decay(self, now=None):
        """Run forgetting: decay salience, prune stale facts."""
        return apply_decay(self.store, now=now)

    def forget(self, phrase):
        """Immediately delete every episode and fact matching phrase."""
        return self.store.delete_matching(phrase)

    def explain(self, text):
        """Why would this score this salience?"""
        return salience.explain(text)

    def stats(self):
        return self.store.stats()
