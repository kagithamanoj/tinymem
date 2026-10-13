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

**Recall** searches both tiers with one hybrid score:

```
score = 0.55 * BM25 + 0.25 * recency + 0.20 * salience
```

BM25 is implemented from scratch (about 40 lines) with a small stemmer.
Recency is exponential with a 45-day time constant. Facts get a small
trust bonus: they survived consolidation.

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
