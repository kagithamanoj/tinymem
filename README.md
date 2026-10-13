# tinymem

A memory engine for AI agents, built from scratch. No frameworks, no vector
database, no magic. Just the four things human memory does that software
memory usually does not: decide what is worth keeping, consolidate
experience into facts while it sleeps, forget on purpose, and revise
beliefs when corrected.

## Why

Every agent forgets. The context window ends, the session dies, and
everything the user taught it evaporates. The standard fixes are
unsatisfying: cram history into the prompt (expensive and noisy), or bolt
on a vector database (storage that never prioritizes, never consolidates,
never resolves contradictions, never forgets).

tinymem is the smallest thing that actually implements memory as a
cognitive process instead of a storage layer.

## How it works

Three tiers:

- **Episodes.** Everything that happened, timestamped, with a salience
  score. Append-only. Salience is assigned by deterministic rules at write
  time: preferences outrank smalltalk, explicit "remember this" outranks
  both.
- **Facts.** The sleep cycle. Consolidation distills salient episodes into
  durable statements with confidence scores and provenance links, so every
  answer can show its work. Contradictions are resolved, not duplicated:
  "actually, I prefer window seats now" supersedes the old preference and
  keeps it in history.
- **Recall.** One hybrid score over both tiers. Relevance is the gate,
  recency and salience are amplifiers: the non-semantic terms multiply the
  semantic score instead of adding to it, so a recent but irrelevant memory
  can never outrank a relevant one.
  - `bm25` (default): `semantic * (0.55 + 0.25*recency + 0.20*salience)`
  - `hybrid`: `blend * (0.60 + 0.25*recency + 0.15*salience)`,
    where `blend = 0.55*BM25 + 0.45*embedding`

  BM25 and a small stemmer are implemented from scratch. Facts get a trust
  bonus, also gated on relevance: they survived consolidation. Two honesty
  mechanisms sit on top: below a calibrated score the memory reports low
  confidence instead of guessing, and `history()` returns the fact lineage
  including superseded facts, so you can ask what it used to believe.

And one architectural rule enforced everywhere: **the model describes,
code decides.** The model may propose facts during consolidation. Salience,
ranking, contradiction detection, decay, and pruning are deterministic
code. Same inputs, same outputs, always testable. The engine runs fully
offline; a provider key only upgrades distillation quality.

Forgetting is a first-class operation: salience decays exponentially
(45-day half-life), stale low-confidence facts are pruned, and
`forget("phrase")` deletes immediately and totally. Memory you cannot
delete is not yours.

## Quickstart

```bash
pip install -r requirements.txt

python -m tinymem remember "I prefer aisle seats on flights"
python -m tinymem remember "My home airport is DFW"
python -m tinymem consolidate
python -m tinymem recall "which airport do I fly from?"
# [1.044] (fact) home airport is dfw  (fact #2 from 1 episode(s))
python -m tinymem recall "which airport do I fly from?" --mode hybrid
python -m tinymem history "seat"   # fact lineage, superseded facts included
python -m tinymem explain "I prefer aisle seats"  # why this salience score
```

Ask something it was never told and it says so instead of guessing:

```bash
python -m tinymem recall "what is my mother's maiden name?"
# (low confidence: the memory does not claim to know this)
```

As a library, ten lines:

```python
from tinymem import Memory

mem = Memory()  # ~/.tinymem/memory.db, created on first use
mem.remember("I am allergic to peanuts")
mem.consolidate()

for hit in mem.recall("any food allergies?"):
    print(hit["score"], hit["text"], "-", hit["provenance"])
```

With an OpenAI key set, consolidation uses the model to distill facts:

```bash
export OPENAI_API_KEY=...
python -m tinymem consolidate
```

## What it handles

| Situation | Behavior |
|---|---|
| "I prefer X" | Distilled into a durable fact at consolidation |
| "Actually, I prefer Y now" | Old fact superseded, new fact active, history kept |
| "Remember that my sister's birthday is June 4" | High salience, consolidated immediately |
| "Traffic was bad today" | Low salience, seen and released, fades with decay |
| "forget my old address" | Every matching episode and fact deleted |

## Evaluation

Two scripted benchmark suites, each run in both recall modes (`bm25` and
`hybrid`), fully deterministic with no API key needed:

- **standard**: a 10-day everyday scenario, 16 episodes, 8 questions.
- **adversarial**: reported-speech traps ("Tom said he loves red-eye
  flights" must never become your preference), negation ("I don't like
  middle seats"), contradictions with and without explicit revision
  language, unanswerable questions (must report low confidence, not
  guess), reinforcement (repeated episodes strengthen one fact instead of
  duplicating it), and temporal history (what did I used to believe?).
  16 episodes, 10 questions.

```bash
make eval
```

Current results: **36/36 across both suites and both modes**. Full report
in [docs/evals.md](docs/evals.md). The design rationale lives in
[docs/design.md](docs/design.md).

## Layout

- `tinymem/store.py` - SQLite schema: episodes, facts, provenance links
- `tinymem/salience.py` - deterministic salience scoring, with explanations
- `tinymem/recall.py` - BM25 from scratch, stemmer, hybrid ranking
- `tinymem/consolidate.py` - the sleep cycle: distill, reinforce, supersede
- `tinymem/decay.py` - exponential forgetting, stale-fact pruning
- `tinymem/providers.py` - tiny provider interface (OpenAI or offline stub)
- `tinymem/memory.py` - the facade: remember, recall, consolidate, forget,
  history, confidence
- `tinymem/cli.py` - command line interface
- `evals/` - benchmark scenarios, demo, and runner
- `tests/` - 30 tests, all offline

## License

MIT.
