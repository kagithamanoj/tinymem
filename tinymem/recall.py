"""Recall: BM25 implemented from scratch, fused with recency and salience.

No vector database, no embeddings service. Tokenize, score, fuse. The full
ranking formula fits on one line and every term is inspectable, which is
exactly what you want when an agent's memory surprises you.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

STOP = frozenset(
    "a an the and or of to in on for with is are was were be been it its"
    " this that these those i you he she we they my your his her our their"
    " me him us them as at by from into over after before not no yes do"
    " does did will would can could should have has had what which who"
    " when where how why".split()
)


def stem(word):
    """Tiny stemmer. Not Porter, but honest: strips the inflections that
    break recall (prefers/prefer, seats/seat, allergies/allergic) and
    leaves the rest alone."""
    if len(word) <= 4:
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3]  # allergies -> allerg
    if word.endswith("ical") and len(word) > 7:
        return word[:-4]
    if word.endswith("ic") and len(word) > 5:
        return word[:-2]  # allergic -> allerg
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    if word.endswith("es") and len(word) > 5 and not word.endswith("ses"):
        return word[:-2]
    if (
        word.endswith("s")
        and not word.endswith(("ss", "is", "us"))
        and len(word) > 4
    ):
        return word[:-1]
    return word


def tokenize(text):
    return [
        stem(t)
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOP
    ]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# Below this top-1 score, recall() reports low confidence: the memory
# does not claim to know the answer. Calibrated on evals/benchmark.py;
# see docs/evals.md for the score distribution.
RECALL_CONFIDENCE = 0.45


def is_confident(results, threshold=RECALL_CONFIDENCE):
    """True when the top result scores above the know/don't-know line."""
    return bool(results) and results[0]["score"] >= threshold


def parse_ts(ts):
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def recency(ts, now):
    """Exponential recency in (0, 1]. 45-day time constant."""
    days = max(0.0, (now - parse_ts(ts)).total_seconds() / 86400.0)
    return math.exp(-days / 45.0)


class BM25:
    """ Textbook BM25. docs is a list of (doc_id, text). """

    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [did for did, _ in docs]
        self.N = len(docs)
        self.doc_len = {}
        self.tf = {}
        self.df = {}
        total = 0
        for did, text in docs:
            toks = tokenize(text)
            self.doc_len[did] = len(toks) or 1
            total += len(toks)
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf[did] = counts
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = total / max(1, self.N)

    def score(self, query):
        out = {}
        for did in self.doc_ids:
            s = 0.0
            dl = self.doc_len[did]
            for t in tokenize(query):
                df = self.df.get(t, 0)
                if not df:
                    continue
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                f = self.tf[did].get(t, 0)
                s += (
                    idf
                    * (f * (self.k1 + 1))
                    / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
            out[did] = s
        return out


def recall(store, query, k=5, now=None, mode="bm25", embed_provider=None):
    """Hybrid recall over episodes and facts.

    Relevance is the gate, recency and salience are amplifiers: the
    non-semantic terms multiply the semantic score instead of adding to
    it, so a recent but irrelevant memory can never outrank a relevant
    one. Facts get a small trust bonus, also gated on relevance.

    mode="bm25":   sem * (0.55 + 0.25*recency + 0.20*salience)
    mode="hybrid": blend * (0.60 + 0.25*recency + 0.15*salience)
                   where blend = 0.55*BM25 + 0.45*embedding

    Returns a list of dicts with text, score, kind, and provenance.
    """
    now = now or datetime.now(timezone.utc)
    episodes = store.all_episodes()
    facts = store.active_facts()

    docs = [("e%d" % e["id"], e["text"]) for e in episodes]
    docs += [("f%d" % f["id"], f["text"]) for f in facts]
    if not docs:
        return []

    raw = BM25(docs).score(query)
    peak = max(raw.values()) or 1.0
    sem = {did: raw[did] / peak for did, _ in docs}

    if mode == "hybrid":
        if embed_provider is None:
            from .providers import default_embedding_provider

            embed_provider = default_embedding_provider()
        vecs = embed_provider.embed([query] + [text for _, text in docs])
        qv = vecs[0]
        cos = {did: cosine(qv, v) for (did, _), v in zip(docs, vecs[1:])}
        peak_cos = max(cos.values()) or 1.0
        sem = {
            did: 0.55 * sem[did] + 0.45 * (cos[did] / peak_cos)
            for did, _ in docs
        }

    scored = []
    for e in episodes:
        key = "e%d" % e["id"]
        if mode == "hybrid":
            s = sem[key] * (0.60 + 0.25 * recency(e["ts"], now)
                            + 0.15 * e["salience"])
        else:
            s = sem[key] * (0.55 + 0.25 * recency(e["ts"], now)
                            + 0.20 * e["salience"])
        scored.append(
            {
                "kind": "episode",
                "text": e["text"],
                "score": round(s, 4),
                "ts": e["ts"],
                "provenance": "episode #%d" % e["id"],
            }
        )
    for f in facts:
        key = "f%d" % f["id"]
        if mode == "hybrid":
            s = sem[key] * (0.60 + 0.25 * recency(f["updated_ts"], now)
                            + 0.15 * f["confidence"])
        else:
            s = sem[key] * (0.55 + 0.25 * recency(f["updated_ts"], now)
                            + 0.20 * f["confidence"])
        s += 0.10 * sem[key]  # trust bonus, gated on relevance
        sources = store.episodes_for_fact(f["id"])
        scored.append(
            {
                "kind": "fact",
                "text": f["text"],
                "score": round(s, 4),
                "ts": f["updated_ts"],
                "provenance": "fact #%d from %d episode(s)"
                % (f["id"], len(sources)),
            }
        )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]
