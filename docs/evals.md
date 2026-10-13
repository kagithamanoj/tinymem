# Benchmark results

Scripted 10-day scenario, 15 episodes, 8 quiz questions. Deterministic run (stub provider, fixed timestamps).

**Recall: 8/8**

Consolidation: 15 episodes seen, 6 facts added, 0 reinforced, 1 superseded.

| Question | Expected | Hit |
|---|---|---|
| window or aisle seat? | window | yes |
| which airport do I fly from? | DFW | yes |
| do I have any food allergies? | peanut | yes |
| when is my sister's birthday? | june 4 | yes |
| what coffee do I like? | filter coffee | yes |
| do I like red-eye flights? | hate | yes |
| who is my manager? | dana | yes |
| direct or connecting flights? | direct | yes |

Contradiction resolution: pass

Regenerate with `make eval`.
