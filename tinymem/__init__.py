"""tinymem: a from-scratch memory engine for AI agents.

Agents forget. This one does not. Three tiers, no frameworks:

  episodes  - everything that happened, timestamped, with salience
  facts     - distilled durable knowledge, with provenance
  recall    - hybrid retrieval: BM25 + recency + salience

Usage:
    from tinymem import Memory
    mem = Memory()
    mem.remember("I prefer aisle seats on flights")
    mem.recall("window or aisle?")
"""

from .memory import Memory

__version__ = "0.1.0"
__all__ = ["Memory"]
