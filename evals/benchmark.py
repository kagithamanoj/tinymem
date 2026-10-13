#!/usr/bin/env python3
"""tinymem benchmark: can the memory actually remember?

Scripted 10-day scenario: stores episodes with day offsets, runs
consolidation and decay, then quizzes recall. Deterministic: stub
provider, fixed timestamps. Prints a report and writes docs/evals.md.

Usage: python evals/benchmark.py
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinymem import Memory
from tinymem.providers import StubProvider

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
QUIZ_DAY = 11


def main():
    root = Path(__file__).resolve().parent
    scenarios = json.loads((root / "scenarios.json").read_text())

    mem = Memory(path=":memory:", provider=StubProvider())
    for ep in scenarios["episodes"]:
        ts = (BASE + timedelta(days=ep["day"])).isoformat()
        mem.remember(ep["text"], ts=ts)

    con = mem.consolidate(now=BASE + timedelta(days=10))
    dec = mem.decay(now=BASE + timedelta(days=10))
    now = BASE + timedelta(days=QUIZ_DAY)

    rows = []
    hits = 0
    for q in scenarios["questions"]:
        results = mem.recall(q["q"], k=3, now=now)
        top = " | ".join(r["text"] for r in results)
        hit = any(q["expect"].lower() in r["text"].lower() for r in results)
        hits += hit
        rows.append((q["q"], q["expect"], top, hit))

    total = len(rows)
    print(f"tinymem benchmark: {hits}/{total} questions recalled")
    print(f"consolidation: {con}")
    print(f"decay: {dec}")
    print()
    for q, expect, top, hit in rows:
        mark = "PASS" if hit else "FAIL"
        print(f"[{mark}] {q}")
        print(f"       expected {expect!r}, top recall: {top[:100]}")

    # contradiction check: the old aisle fact must be superseded, the new
    # window fact must be active
    facts = [f["text"] for f in mem.store.active_facts()]
    contradiction_ok = (
        any("window" in f for f in facts)
        and "prefers aisle seats on flights" not in facts
    )
    print()
    print(
        "[%s] contradiction resolved (window kept, aisle superseded)"
        % ("PASS" if contradiction_ok else "FAIL")
    )

    md = [
        "# Benchmark results",
        "",
        f"Scripted 10-day scenario, {len(scenarios['episodes'])} episodes, "
        f"{total} quiz questions. Deterministic run (stub provider, fixed "
        "timestamps).",
        "",
        f"**Recall: {hits}/{total}**",
        "",
        f"Consolidation: {con['episodes_seen']} episodes seen, "
        f"{con['facts_added']} facts added, {con['facts_reinforced']} "
        f"reinforced, {con['superseded']} superseded.",
        "",
        "| Question | Expected | Hit |",
        "|---|---|---|",
    ]
    for q, expect, _, hit in rows:
        md.append(f"| {q} | {expect} | {'yes' if hit else 'no'} |")
    md += [
        "",
        f"Contradiction resolution: {'pass' if contradiction_ok else 'fail'}",
        "",
        "Regenerate with `make eval`.",
    ]
    docs_dir = root.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "evals.md").write_text("\n".join(md) + "\n")

    ok = hits == total and contradiction_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
