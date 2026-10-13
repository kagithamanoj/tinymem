"""Salience: how much an episode deserves to be remembered.

Deterministic and explainable. Every rule is visible below, and recall()
can tell you exactly why an episode scored the way it did. The model never
assigns salience; code does.
"""

from __future__ import annotations

import re

PREFERENCE_VERBS = (
    "prefer", "prefers", "preferred",
    "love", "loves", "like", "likes",
    "hate", "hates", "dislike", "allergic",
    "always", "never",
)
EXPLICIT_MEMORY = ("remember", "don't forget", "do not forget", "note to self")
SELF_REF = re.compile(r"\b(my|mine|i'm|i am)\b", re.I)
CONCRETE = re.compile(r"\d")


def score(text):
    """Salience in [0.05, 1.0]. Higher means more worth consolidating."""
    t = text.lower()
    s = 0.25  # everything that happened is worth something

    if any(v in t for v in PREFERENCE_VERBS):
        s += 0.35  # preferences are the most durable knowledge
    if any(v in t for v in EXPLICIT_MEMORY):
        s += 0.25  # the user explicitly asked to remember
    if SELF_REF.search(t):
        s += 0.10  # about the user, likely to matter later
    if CONCRETE.search(t):
        s += 0.05  # concrete details beat vague ones
    if text.strip().endswith("?"):
        s -= 0.10  # questions are usually ephemeral

    return min(1.0, max(0.05, s))


def explain(text):
    """One-line explanation of a salience score. Used by the CLI."""
    reasons = ["base 0.25"]
    t = text.lower()
    if any(v in t for v in PREFERENCE_VERBS):
        reasons.append("preference +0.35")
    if any(v in t for v in EXPLICIT_MEMORY):
        reasons.append("explicit remember +0.25")
    if SELF_REF.search(t):
        reasons.append("self-reference +0.10")
    if CONCRETE.search(t):
        reasons.append("concrete detail +0.05")
    if text.strip().endswith("?"):
        reasons.append("question -0.10")
    return f"{score(text):.2f} ({', '.join(reasons)})"
