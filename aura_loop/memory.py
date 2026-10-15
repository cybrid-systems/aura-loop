"""Memory: SQLite authority table of past transitions. Vector search is a stub (column only)."""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass

VERDICTS = ("KEEP", "ROLLBACK", "REJECT")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transitions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  repo       TEXT NOT NULL,
  symbol     TEXT NOT NULL,
  before_src TEXT NOT NULL,
  after_src  TEXT NOT NULL,
  intent     TEXT NOT NULL DEFAULT '',
  verdict    TEXT NOT NULL CHECK (verdict IN ('KEEP','ROLLBACK','REJECT')),
  reason     TEXT NOT NULL DEFAULT '',
  sha        TEXT NOT NULL DEFAULT '',
  ts         REAL NOT NULL,
  embedding  BLOB            -- stub: reserved for a later vector index (pgvector/LanceDB/...)
);
CREATE INDEX IF NOT EXISTS idx_transitions_symbol ON transitions(symbol, verdict);

-- aura-kernel-sink decisions (aura_loop/sink_ingest.py). Same store, second
-- table: it is both the retrieval memory of the sink policy and its replay
-- buffer (state, action, reward) for a later RL learner.
CREATE TABLE IF NOT EXISTS sink_transitions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  repo       TEXT NOT NULL,
  kernel     TEXT NOT NULL,
  shape      TEXT NOT NULL,             -- shape bucket key (sink_ingest.shape_key)
  state      TEXT NOT NULL,             -- JSON: kernel shape features + Soft cost
  action     TEXT NOT NULL CHECK (action IN ('sink','skip')),
  threshold  REAL NOT NULL,
  floor_us   REAL NOT NULL,
  decision   TEXT NOT NULL CHECK (decision IN ('KEEP','ROLLBACK','SKIP','ABORT')),
  stage      TEXT NOT NULL DEFAULT '',
  speedup    REAL NOT NULL DEFAULT 0,
  soft_us    REAL NOT NULL DEFAULT 0,
  reward     REAL NOT NULL,
  round      INTEGER NOT NULL DEFAULT 0,
  arm        TEXT NOT NULL DEFAULT '',  -- which policy proposed it (fixed:<t>, knn, bandit, ...)
  proposal   TEXT NOT NULL DEFAULT '',
  raw        TEXT NOT NULL,             -- the trajectory line as written by aura-kernel-sink
  digest     TEXT NOT NULL UNIQUE,      -- sha256 of raw: re-ingesting a file is a no-op
  ts         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sink_shape ON sink_transitions(shape, decision);
CREATE INDEX IF NOT EXISTS idx_sink_kernel ON sink_transitions(kernel, round);
"""


@dataclass(frozen=True)
class Precedent:
    id: int
    repo: str
    symbol: str
    before_src: str
    after_src: str
    intent: str
    verdict: str
    reason: str
    sha: str
    ts: float
    score: float = 0.0


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[^\s()]+", s))


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


class Memory:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def writeback(self, *, repo: str, symbol: str, before_src: str, after_src: str, intent: str,
                  verdict: str, reason: str = "", sha: str = "", ts: float | None = None) -> int:
        if verdict not in VERDICTS:
            raise ValueError(verdict)
        cur = self.db.execute(
            "INSERT INTO transitions(repo,symbol,before_src,after_src,intent,verdict,reason,sha,ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (repo, symbol, before_src, after_src, intent, verdict, reason, sha, ts or time.time()))
        self.db.commit()
        return int(cur.lastrowid)

    def _rows(self, sql: str, args=()) -> list[Precedent]:
        cols = "id,repo,symbol,before_src,after_src,intent,verdict,reason,sha,ts"
        return [Precedent(*r) for r in self.db.execute(f"SELECT {cols} FROM transitions {sql}", args)]

    def retrieve(self, symbol: str, text: str = "", k: int = 5, verdicts=VERDICTS) -> list[Precedent]:
        """MVP retrieval: exact symbol filter + token-overlap similarity on before_src.

        Later: replace/augment with an embedding search over `embedding`.
        """
        q = ",".join("?" * len(verdicts))
        rows = self._rows(f"WHERE symbol=? AND verdict IN ({q})", (symbol, *verdicts))
        if text:
            rows += [r for r in self._rows("WHERE symbol<>? AND before_src LIKE ?", (symbol, f"%{text[:40]}%"))
                     if r.verdict in verdicts]
        scored = [Precedent(**{**r.__dict__, "score": similarity(text, r.before_src) if text else 0.0}) for r in rows]
        scored.sort(key=lambda r: (r.score, r.ts), reverse=True)
        return scored[:k]

    def failed_afters(self, symbol: str) -> set[str]:
        return {r.after_src for r in self._rows("WHERE symbol=? AND verdict IN ('ROLLBACK','REJECT')", (symbol,))}

    def history(self, symbol: str) -> dict:
        return {v: n for v, n in self.db.execute(
            "SELECT verdict, COUNT(*) FROM transitions WHERE symbol=? GROUP BY verdict", (symbol,))}

    def all(self) -> list[Precedent]:
        return self._rows("ORDER BY id")


    # ---- aura-kernel-sink transitions (see sink_ingest.py)
    SINK_COLS = ("id,repo,kernel,shape,state,action,threshold,floor_us,decision,stage,speedup,soft_us,"
                 "reward,round,arm,proposal,raw,digest,ts")

    def sink_writeback(self, row: dict) -> int | None:
        """Insert one sink transition; returns its id, or None if this exact line was ingested before."""
        cols = [c for c in self.SINK_COLS.split(",") if c != "id"]
        try:
            cur = self.db.execute(f"INSERT INTO sink_transitions({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                  tuple(row[c] for c in cols))
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                return None
            raise
        self.db.commit()
        return int(cur.lastrowid)

    def sink_rows(self, where: str = "", args=()) -> list[dict]:
        cols = self.SINK_COLS.split(",")
        return [dict(zip(cols, r)) for r in
                self.db.execute(f"SELECT {self.SINK_COLS} FROM sink_transitions {where}", args)]
