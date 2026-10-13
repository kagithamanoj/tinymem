#!/usr/bin/env python3
"""tinymem demo: three days of memory in ten seconds."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinymem import Memory

base = datetime(2026, 1, 1, tzinfo=timezone.utc)
mem = Memory(path=":memory:")

mem.remember("I prefer aisle seats on flights", ts=base.isoformat())
mem.remember("My home airport is DFW",
             ts=(base + timedelta(days=1)).isoformat())
mem.remember("Traffic was bad today",
             ts=(base + timedelta(days=1)).isoformat())
mem.remember("Actually I prefer window seats now, aisle feels cramped",
             ts=(base + timedelta(days=2)).isoformat())

print("consolidate:", mem.consolidate(now=base + timedelta(days=2)))
print()
for q in ["which airport do I fly from?", "window or aisle?"]:
    print("Q:", q)
    for r in mem.recall(q, now=base + timedelta(days=3)):
        print(f"  [{r['score']:.3f}] ({r['kind']}) {r['text']}")
        print(f"           {r['provenance']}")
