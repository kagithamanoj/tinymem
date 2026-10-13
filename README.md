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
- **Recall.** One hybrid score over both tiers:
  `0.55 * BM25 + 0.25 * recency + 0.20 * salience`, with BM25 and a small
  stemmer implemented from scratch. Facts get a trust bonus: they survived
  consolidation.

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
# [0.731] (fact) home airport is dfw  (fact #2 from 1 episode(s))
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

A scripted 10-day benchmark: 16 episodes (preferences, identity details,
noise, one contradiction), then 8 quiz questions against recall. Fully
deterministic, no API key needed.

```bash
make eval
```

Current results: **8/8 recalled**, contradiction resolved correctly. Full
report in [docs/evals.md](docs/evals.md). The design rationale lives in
[docs/design.md](docs/design.md).

## Layout

- `tinymem/store.py` - SQLite schema: episodes, facts, provenance links
- `tinymem/salience.py` - deterministic salience scoring, with explanations
- `tinymem/recall.py` - BM25 from scratch, stemmer, hybrid ranking
- `tinymem/consolidate.py` - the sleep cycle: distill, reinforce, supersede
- `tinymem/decay.py` - exponential forgetting, stale-fact pruning
- `tinymem/providers.py` - tiny provider interface (OpenAI or offline stub)
- `tinymem/memory.py` - the facade: remember, recall, consolidate, forget
- `tinymem/cli.py` - command line interface
- `evals/` - benchmark scenario and runner
- `tests/` - 20 tests, all offline

## License

MIT.
