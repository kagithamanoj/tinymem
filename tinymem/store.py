"""SQLite persistence. Three tables, zero magic.

episodes      - raw experience: what happened, when, how salient, whether
                it has been consolidated into facts yet.
facts         - distilled durable knowledge. UNIQUE on text; consolidation
                reinforces instead of duplicating. Superseded facts are kept
                for provenance, never deleted by consolidation.
fact_sources  - provenance: which episodes back each fact.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    salience     REAL NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL UNIQUE,
    confidence   REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'active',
    created_ts   TEXT NOT NULL,
    updated_ts   TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id    INTEGER NOT NULL REFERENCES facts(id),
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    PRIMARY KEY (fact_id, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
"""


class Store:
    def __init__(self, path=":memory:"):
        if path != ":memory:":
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path = str(p)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- episodes --------------------------------------------------------

    def add_episode(self, ts, kind, text, salience):
        cur = self.conn.execute(
            "INSERT INTO episodes (ts, kind, text, salience) VALUES (?,?,?,?)",
            (ts, kind, text, salience),
        )
        self.conn.commit()
        return cur.lastrowid

    def unconsolidated(self, min_salience=0.0):
        return self.conn.execute(
            "SELECT * FROM episodes WHERE consolidated = 0 AND salience >= ?"
            " ORDER BY id",
            (min_salience,),
        ).fetchall()

    def mark_consolidated(self, episode_id):
        self.conn.execute(
            "UPDATE episodes SET consolidated = 1 WHERE id = ?", (episode_id,)
        )
        self.conn.commit()

    def all_episodes(self):
        return self.conn.execute("SELECT * FROM episodes ORDER BY id").fetchall()

    def decay_episodes(self, factor_fn):
        """Multiply every episode's salience by factor_fn(ts). Returns count."""
        rows = self.conn.execute("SELECT id, ts FROM episodes").fetchall()
        n = 0
        for row in rows:
            factor = factor_fn(row["ts"])
            if factor < 1.0:
                self.conn.execute(
                    "UPDATE episodes SET salience = salience * ? WHERE id = ?",
                    (factor, row["id"]),
                )
                n += 1
        self.conn.commit()
        return n

    # -- facts -----------------------------------------------------------

    def add_fact(self, text, confidence, ts):
        """Insert or reinforce. Returns (fact_id, is_new)."""
        row = self.conn.execute(
            "SELECT id, confidence FROM facts WHERE text = ?", (text,)
        ).fetchone()
        if row:
            new_conf = min(1.0, max(row["confidence"], confidence) + 0.05)
            self.conn.execute(
                "UPDATE facts SET confidence = ?, updated_ts = ?,"
                " source_count = source_count + 1, status = 'active'"
                " WHERE id = ?",
                (new_conf, ts, row["id"]),
            )
            self.conn.commit()
            return row["id"], False
        cur = self.conn.execute(
            "INSERT INTO facts (text, confidence, created_ts, updated_ts,"
            " source_count) VALUES (?,?,?,?,1)",
            (text, confidence, ts, ts),
        )
        self.conn.commit()
        return cur.lastrowid, True

    def link_source(self, fact_id, episode_id):
        self.conn.execute(
            "INSERT OR IGNORE INTO fact_sources VALUES (?,?)",
            (fact_id, episode_id),
        )
        self.conn.commit()

    def supersede(self, fact_id, ts):
        self.conn.execute(
            "UPDATE facts SET status = 'superseded', updated_ts = ? WHERE id = ?",
            (ts, fact_id),
        )
        self.conn.commit()

    def active_facts(self):
        return self.conn.execute(
            "SELECT * FROM facts WHERE status = 'active' ORDER BY confidence DESC"
        ).fetchall()

    def episodes_for_fact(self, fact_id):
        return self.conn.execute(
            "SELECT e.* FROM episodes e"
            " JOIN fact_sources fs ON fs.episode_id = e.id"
            " WHERE fs.fact_id = ?",
            (fact_id,),
        ).fetchall()

    def prune_facts(self, min_confidence, older_than_ts):
        cur = self.conn.execute(
            "DELETE FROM facts WHERE confidence < ? AND updated_ts < ?"
            " AND status = 'active'",
            (min_confidence, older_than_ts),
        )
        self.conn.commit()
        return cur.rowcount

    # -- maintenance -----------------------------------------------------

    def delete_matching(self, phrase):
        like = f"%{phrase.lower()}%"
        cur = self.conn.execute(
            "DELETE FROM episodes WHERE lower(text) LIKE ?", (like,)
        )
        n_episodes = cur.rowcount
        cur = self.conn.execute(
            "DELETE FROM facts WHERE lower(text) LIKE ?", (like,)
        )
        n_facts = cur.rowcount
        self.conn.commit()
        return n_episodes + n_facts

    def stats(self):
        ep = self.conn.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(consolidated),0) AS k"
            " FROM episodes"
        ).fetchone()
        fa = self.conn.execute(
            "SELECT COUNT(*) AS c FROM facts WHERE status = 'active'"
        ).fetchone()
        return {
            "episodes": ep["c"],
            "consolidated": ep["k"],
            "facts": fa["c"],
        }
