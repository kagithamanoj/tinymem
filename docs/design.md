# Design: how tinymem remembers

## The problem

Every agent forgets. The context window ends, the session dies, and
everything the user taught it evaporates. The usual fixes are
unsatisfying: stuff everything into the prompt (expensive, noisy), or bolt
on a vector database (dumb storage that never forgets, never prioritizes,
never resolves contradictions).

Human memory does four things a vector store does not:

1. **It decides what is worth keeping.** You remember your passport expiry.
   You do not remember Tuesday's lunch. Salience is assigned at write time.
2. **It consolidates.** Sleep turns the day's episodes into durable facts.
   Nobody replays raw experience to answer "what seat do I prefer".
3. **It forgets on purpose.** Old, unreinforced traces fade. Forgetting is a
   feature: it is how signal survives noise.
4. **It revises.** "Actually, I prefer window seats now" overwrites the old
   preference. The old belief is not deleted from history, but it stops
   being the answer.

tinymem implements exactly those four, from scratch, with no frameworks.

## The three tiers

```
remember("I prefer aisle seats")
        |
        v
+------------------+      salience >= 0.45      +------------------+
|    EPISODES      | ---- (sleep cycle) ------> |      FACTS       |
| everything that  |      distill + resolve     | durable knowledge|
| happened, with   |                            | confidence,      |
| timestamp and    |                            | provenance       |
| salience         |                            +------------------+
+------------------+                                     |
        | recall: 0.55*BM25 + 0.25*recency + 0.20*salience |
        +------------------------------------------------+
                                   |
                                   v
                              ranked answers
                              with provenance
```

**Episodes** are append-only raw experience. Every `remember()` call stores
the text, an ISO timestamp, a kind, and a salience score. Nothing is
interpreted at write time.

**Facts** are the distilled layer. Consolidation reads salient,
unconsolidated episodes and extracts durable statements: preferences,
identity details, commitments. Each fact carries a confidence, a status
(active or superseded), and links back to the episodes it came from, so
every answer can show its work.

**Recall** searches both tiers with one hybrid score. Relevance is the
gate, recency and salience are amplifiers: the non-semantic terms
multiply the semantic score instead of adding to it, so a recent but
irrelevant memory can never outrank a relevant one.

```
bm25 mode:   semantic * (0.55 + 0.25*recency + 0.20*salience)
hybrid mode: blend   * (0.60 + 0.25*recency + 0.15*salience)
             where blend = 0.55*BM25 + 0.45*embedding
```

BM25 and a small stemmer are implemented from scratch (about 60 lines).
Recency is exponential with a 45-day time constant. Facts get a small
trust bonus, also gated on relevance: they survived consolidation.

Two honesty mechanisms sit on top of recall:

- **Confidence.** Below a calibrated top-1 score, the memory reports low
  confidence instead of claiming an answer. On the benchmark, answerable
  questions score 0.84+, unanswerable ones score 0.00. The line is drawn
  at 0.45.
- **History.** `history("seat")` returns the fact lineage including
  superseded facts, so the memory can answer what it used to believe and
  show exactly when it changed its mind.

## The model describes, code decides

The one architectural rule, enforced everywhere:

- The model may **propose** facts during consolidation. It never writes to
  the store directly.
- Salience, recall ranking, contradiction detection, decay, and pruning
  are **deterministic code**. Same inputs, same outputs, always testable.
- With no API key, a rule-based extractor handles the common durable
  patterns ("I prefer X", "my Y is Z", "remember that..."). The engine is
  fully functional offline; a provider only upgrades distillation quality.

This split is what makes the system debuggable. When recall surprises
you, every term of the score is inspectable, and `tinymem explain` tells
you why an episode scored its salience.

## Consolidation policy

A consolidation pass:

1. Takes unconsolidated episodes above the salience threshold.
2. Distills each into candidate facts (model or rules).
3. For each candidate: if the exact fact exists, reinforce it (confidence
   bump, new provenance link). If it conflicts with an active fact (token
   overlap, or revision language like "actually" plus topical overlap),
   supersede the old fact and add the new one. Otherwise, add it.
4. Marks every seen episode consolidated, whether or not it produced a
   fact. Low-salience episodes are seen and released, not kept forever.

## Forgetting policy

- Episode salience decays exponentially with a 45-day half-life. Old noise
  sinks in recall without being deleted.
- Facts below 0.15 confidence untouched for 120 days are pruned.
- `forget("phrase")` is immediate and total: every matching episode and
  fact is deleted. Memory you cannot delete is not yours.

## What it is not

Not a vector database. Not a framework. Not a demo wired to one provider.
It is the smallest thing that actually implements episodic memory,
consolidation, and forgetting, with tests and a benchmark to prove it.

## Embeddings, measured honestly

`recall()` has two modes. `bm25` is the default: pure lexical, deterministic,
offline. `hybrid` blends in embedding cosine similarity
(`0.55*BM25 + 0.45*embedding`) before the relevance gate is applied.

The embedding provider is a tiny interface with two implementations. When
`OPENAI_API_KEY` is set, hybrid mode uses OpenAI's text-embedding-3-small.
Otherwise it falls back to a deterministic token-hashed baseline that runs
fully offline. The hashed baseline is documented as a baseline, not a
replacement for real embeddings.

The benchmark runs every question in both modes and reports the comparison
side by side, so the embedding upgrade has to earn its place with numbers,
not promises. Swapping providers is one constructor argument; nothing else
in the engine changes.
