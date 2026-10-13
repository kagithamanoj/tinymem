# Benchmark results

Two suites, two recall modes. Deterministic: stub provider, fixed
timestamps, hashed-embedding baseline for hybrid mode.
Regenerate with `make eval`.

## Suites

- **standard**: 10-day everyday scenario, 16 episodes, 8 questions.
- **adversarial**: reported-speech traps, negation, unmarked
  contradictions, unanswerable questions, reinforcement, temporal
  history. 16 episodes, 10 questions.

## Results

| Suite | Mode | Score |
|---|---|---|
| standard | bm25 | 8/8 |
| standard | hybrid | 8/8 |
| adversarial | bm25 | 10/10 |
| adversarial | hybrid | 10/10 |

**Total: 36/36**

## What the adversarial suite proves

- Reported speech never becomes a user fact (`Tom said he loves red-eye flights`).
- Negation is extracted correctly (`I don't like middle seats`).
- Contradictions resolve with and without explicit revision language.
- Unanswerable questions report low confidence instead of guessing.
- Repeated episodes reinforce one fact instead of duplicating it.
- `history()` answers what the memory used to believe.

## Recall modes

- **bm25**: 0.55*BM25 + 0.25*recency + 0.20*salience (+0.10 fact bonus).
- **hybrid**: 0.40*BM25 + 0.25*embedding + 0.20*recency + 0.15*salience
  (+0.10 fact bonus). Embeddings are a hashed offline baseline unless
  `OPENAI_API_KEY` is set, in which case text-embedding-3-small is used.
