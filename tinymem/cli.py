"""Command line interface.

    python -m tinymem remember "I prefer aisle seats"
    python -m tinymem recall "window or aisle?" -k 5
    python -m tinymem consolidate
    python -m tinymem forget "aisle"
    python -m tinymem history "seat"
    python -m tinymem stats
    python -m tinymem decay
"""

from __future__ import annotations

import argparse

from .memory import DEFAULT_PATH, Memory


def cmd_remember(mem, args):
    eid, s = mem.remember(args.text)
    print(f"remembered #{eid} (salience {s})")


def cmd_recall(mem, args):
    results = mem.recall(args.query, k=args.k, mode=args.mode)
    if not results:
        print("nothing remembered yet")
        return
    for r in results:
        print(f"[{r['score']:.3f}] ({r['kind']}) {r['text']}")
        print(f"         {r['provenance']} | {r['ts'][:10]}")
    if not mem.confident(results):
        print("(low confidence: the memory does not claim to know this)")


def cmd_consolidate(mem, args):
    stats = mem.consolidate()
    print(
        "consolidated {episodes_seen} episodes: "
        "{facts_added} facts added, {facts_reinforced} reinforced, "
        "{superseded} superseded".format(**stats)
    )


def cmd_forget(mem, args):
    n = mem.forget(args.phrase)
    print(f"forgot {n} record(s) matching {args.phrase!r}")


def cmd_stats(mem, args):
    for k, v in mem.stats().items():
        print(f"{k}: {v}")


def cmd_decay(mem, args):
    stats = mem.decay()
    print(
        "decay: {episodes_decayed} episodes faded,"
        " {facts_pruned} facts pruned".format(**stats)
    )


def cmd_explain(mem, args):
    print(mem.explain(args.text))


def cmd_history(mem, args):
    rows = mem.history(args.phrase)
    if not rows:
        print(f"no history for {args.phrase!r}")
        return
    for r in rows:
        print(f"[{r['status']}] {r['text']}")
        print(
            f"         confidence {r['confidence']}, updated {r['updated']},"
            f" {r['sources']} source(s)"
        )


def build_parser():
    p = argparse.ArgumentParser(prog="tinymem", description="a memory for agents")
    p.add_argument("--db", default=DEFAULT_PATH, help="sqlite path")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("remember", help="store an episode")
    r.add_argument("text")
    r.set_defaults(fn=cmd_remember)

    r = sub.add_parser("recall", help="recall relevant memories")
    r.add_argument("query")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--mode", choices=["bm25", "hybrid"], default="bm25")
    r.set_defaults(fn=cmd_recall)

    r = sub.add_parser("consolidate", help="distill episodes into facts")
    r.set_defaults(fn=cmd_consolidate)

    r = sub.add_parser("forget", help="delete records matching a phrase")
    r.add_argument("phrase")
    r.set_defaults(fn=cmd_forget)

    r = sub.add_parser("stats", help="show store stats")
    r.set_defaults(fn=cmd_stats)

    r = sub.add_parser("decay", help="run the forgetting curve")
    r.set_defaults(fn=cmd_decay)

    r = sub.add_parser("explain", help="explain a salience score")
    r.add_argument("text")
    r.set_defaults(fn=cmd_explain)

    r = sub.add_parser("history", help="fact lineage incl. superseded facts")
    r.add_argument("phrase")
    r.set_defaults(fn=cmd_history)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    mem = Memory(path=args.db)
    args.fn(mem, args)


if __name__ == "__main__":
    main()
