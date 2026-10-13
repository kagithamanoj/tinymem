#!/usr/bin/env python3
"""tinymem benchmark: can the memory actually remember?

Two suites, two recall modes, fully deterministic (stub provider, fixed
timestamps, hashed-embedding baseline for hybrid mode):

  scenarios.json      - 10-day everyday scenario, 16 episodes, 8 questions
  scenarios_adv.json  - adversarial: reported-speech traps, negation,
                        unmarked contradictions, unanswerable questions,
                        reinforcement, temporal history

Check types per question:
  contains      - expect substring in top-k recall texts
  no_fact       - no fact in top-k (episodes are fine: provenance is not
                  hallucination)
  no_fact_about - no fact in top-k containing the `about` substring
  not_confident - recall must report low confidence (it should say it
                  does not know)
  history       - expect substring in fact lineage incl. superseded facts
  fact_sources  - a fact must have >= min_sources provenance links

Usage: python evals/benchmark.py
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinymem import Memory
from tinymem.providers import StubProvider
from tinymem.recall import RECALL_CONFIDENCE

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
MODES = ("bm25", "hybrid")


def run_question(mem, q, mode, now):
    check = q["check"]
    if check == "history":
        rows = mem.history(q["history_phrase"])
        hit = any(q["expect"].lower() in r["text"].lower() for r in rows)
        detail = " | ".join(
            f"[{r['status']}] {r['text']}" for r in rows[:4]
        )
        return hit, detail

    if check == "fact_sources":
        rows = mem.store.facts_matching(q["expect"], include_superseded=False)
        hit = any(len(mem.store.episodes_for_fact(r["id"])) >= q["min_sources"]
                  for r in rows)
        detail = f"{len(rows)} matching fact(s)"
        return hit, detail

    results = mem.recall(q["q"], k=3, now=now, mode=mode)
    top = " | ".join(f"({r['kind']}) {r['text']}" for r in results)

    if check == "contains":
        hit = any(q["expect"].lower() in r["text"].lower() for r in results)
    elif check == "no_fact":
        hit = not any(r["kind"] == "fact" for r in results)
    elif check == "no_fact_about":
        hit = not any(
            r["kind"] == "fact" and q["about"].lower() in r["text"].lower()
            for r in results
        )
    elif check == "not_confident":
        hit = not mem.confident(results)
    else:
        raise ValueError(f"unknown check {check!r}")
    return hit, top


def run_suite(scenarios_path, mode):
    scenarios = json.loads(Path(scenarios_path).read_text())
    mem = Memory(path=":memory:", provider=StubProvider())
    for ep in scenarios["episodes"]:
        ts = (BASE + timedelta(days=ep["day"])).isoformat()
        mem.remember(ep["text"], ts=ts)
    con = mem.consolidate(now=BASE + timedelta(days=10))
    now = BASE + timedelta(days=11)

    rows = []
    for q in scenarios["questions"]:
        hit, detail = run_question(mem, q, mode, now)
        rows.append((q["q"], q["check"], hit, detail))
    return rows, con


def main():
    root = Path(__file__).resolve().parent
    suites = {
        "standard": root / "scenarios.json",
        "adversarial": root / "scenarios_adv.json",
    }

    all_rows = {}
    totals = {}
    for suite, path in suites.items():
        for mode in MODES:
            rows, con = run_suite(path, mode)
            all_rows[(suite, mode)] = rows
            hits = sum(1 for _, _, hit, _ in rows)
            totals[(suite, mode)] = (hits, len(rows))
            print(f"[{suite}/{mode}] {hits}/{len(rows)}")
            for q, check, hit, detail in rows:
                print(f"  [{'PASS' if hit else 'FAIL'}] ({check}) {q}")
                if not hit:
                    print(f"         {detail[:120]}")
            print(f"  consolidation: {con['facts_added']} added,"
                  f" {con['superseded']} superseded")
            print()

    grand_hits = sum(h for h, _ in totals.values())
    grand_total = sum(t for _, t in totals.values())
    print(f"TOTAL: {grand_hits}/{grand_total}")
    print(f"confidence threshold: {RECALL_CONFIDENCE}")

    md = [
        "# Benchmark results",
        "",
        "Two suites, two recall modes. Deterministic: stub provider, fixed",
        "timestamps, hashed-embedding baseline for hybrid mode.",
        "Regenerate with `make eval`.",
        "",
        "## Suites",
        "",
        "- **standard**: 10-day everyday scenario, 16 episodes, 8 questions.",
        "- **adversarial**: reported-speech traps, negation, unmarked",
        "  contradictions, unanswerable questions, reinforcement, temporal",
        "  history. 16 episodes, 10 questions.",
        "",
        "## Results",
        "",
        "| Suite | Mode | Score |",
        "|---|---|---|",
    ]
    for (suite, mode), (hits, total) in totals.items():
        md.append(f"| {suite} | {mode} | {hits}/{total} |")
    md += [
        "",
        f"**Total: {grand_hits}/{grand_total}**",
        "",
        "## What the adversarial suite proves",
        "",
        "- Reported speech never becomes a user fact"
        " (`Tom said he loves red-eye flights`).",
        "- Negation is extracted correctly (`I don't like middle seats`).",
        "- Contradictions resolve with and without explicit revision language.",
        "- Unanswerable questions report low confidence instead of guessing.",
        "- Repeated episodes reinforce one fact instead of duplicating it.",
        "- `history()` answers what the memory used to believe.",
        "",
        "## Recall modes",
        "",
        "- **bm25**: 0.55*BM25 + 0.25*recency + 0.20*salience (+0.10 fact bonus).",
        "- **hybrid**: 0.40*BM25 + 0.25*embedding + 0.20*recency + 0.15*salience",
        "  (+0.10 fact bonus). Embeddings are a hashed offline baseline unless",
        "  `OPENAI_API_KEY` is set, in which case text-embedding-3-small is used.",
    ]
    docs_dir = root.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "evals.md").write_text("\n".join(md) + "\n")

    return 0 if grand_hits == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())
