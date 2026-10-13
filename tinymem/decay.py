"""Forgetting: the other half of memory.

Salience decays exponentially (Ebbinghaus-style) so old, unreinforced
episodes fade from recall. Facts that were never reinforced and have not
been touched in a long time get pruned. Explicit forgetting is separate
and immediate: see Memory.forget().
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .recall import parse_ts


def apply_decay(store, now=None, half_life_days=45.0,
                prune_confidence=0.15, prune_after_days=120):
    """Decay episode salience; prune stale low-confidence facts."""
    now = now or datetime.now(timezone.utc)

    def factor(ts):
        days = max(0.0, (now - parse_ts(ts)).total_seconds() / 86400.0)
        return 0.5 ** (days / half_life_days)

    episodes_decayed = store.decay_episodes(factor)
    cutoff = (now - timedelta(days=prune_after_days)).isoformat()
    facts_pruned = store.prune_facts(prune_confidence, cutoff)
    return {
        "episodes_decayed": episodes_decayed,
        "facts_pruned": facts_pruned,
    }
