"""Ingest aura-rocks structure-switch trajectories and learn its thresholds.

aura-rocks (lib/adapt.aura) switches its memtable representation (skiplist /
vector / hash) and compaction style (leveled / tiered) at runtime by hot
rebinding workspace defines. Every decision is one JSON line:

  {"knob": "memtable_rep", "from": "skiplist", "to": "vector",
   "decision": "SWAP" | "KEEP" | "ROLLBACK" | "REJECT" | "ABORT",
   "stage": "gate" | "policy" | "post" | "post-shifted",
   "speedup": ..., "reward": ..., "label": ..., "features": {observer features}}

A SWAP (the action, taken in `features` = state) is followed by its post check
for the same knob: KEEP with the realized speedup (reward log(speedup)) or
ROLLBACK (reward -1). `transitions` pairs them into (state, action, reward)
tuples; REJECT / ABORT are gate outcomes (the action was not taken: -0.1 / -2).

`AdaptPolicy.propose` learns the rule thresholds aura-rocks reads from
ADAPT_PROPOSAL: for each (knob, target) the feature that triggers it, the
threshold t maximizing the summed realized reward of the swaps the rule would
still take (feature >= t). Too few realized rows -> the default stays. The
engine never learns; it follows the proposal it is given.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DECISIONS = ("SWAP", "KEEP", "ROLLBACK", "REJECT", "ABORT")

# threshold name -> (knob, target value, feature(s) the rule compares, default)
RULES = {
    "mem_write_hi": ("memtable_rep", "vector", ("write_ratio",), 0.9),
    "mem_scan_hi": ("memtable_rep", "skiplist", ("scan_share",), 0.15),
    "mem_read_hi": ("memtable_rep", "hash", ("read_ratio",), 0.5),
    "cmp_write_hi": ("compaction_style", "tiered", ("write_ratio",), 0.7),
    "cmp_read_hi": ("compaction_style", "leveled", ("read_ratio", "scan_share"), 0.5),
}
GRID = [round(0.3 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95


def to_row(line: str, *, arm: str = "") -> dict | None:
    line = line.strip()
    if not line:
        return None
    r = json.loads(line)
    if r.get("decision") not in DECISIONS or "knob" not in r:
        return None
    return {
        "repo": r.get("repo", "aura-rocks"), "knob": r["knob"], "from_v": str(r.get("from", "")), "to_v": str(r.get("to", "")),
        "decision": r["decision"], "stage": r.get("stage", ""), "speedup": float(r.get("speedup") or 0.0),
        "reward": float(r.get("reward") or 0.0), "label": str(r.get("label", "")),
        "state": json.dumps(r.get("features") or {}, sort_keys=True), "arm": arm,
        "raw": line, "digest": hashlib.sha256(f"{arm}\n{line}".encode()).hexdigest(),
        "ts": float(r.get("ts", 0)) / 1000.0,
    }


def ingest(mem, path: str | Path, *, arm: str = "") -> list[int]:
    ids = []
    for line in Path(path).read_text().splitlines():
        row = to_row(line, arm=arm)
        if row is not None:
            i = mem.adapt_writeback(row)
            if i is not None:
                ids.append(i)
    return ids


def transitions(rows: list[dict]) -> list[dict]:
    """(state, action, reward) tuples: SWAP + its post outcome; REJECT/ABORT as not-taken actions."""
    out, open_swaps = [], {}
    for r in sorted(rows, key=lambda r: (r["arm"], r["label"], r["ts"], r["id"] if "id" in r else 0)):
        key = (r["arm"], r["label"], r["knob"])
        if r["decision"] == "SWAP":
            open_swaps[key] = r
        elif r["decision"] in ("KEEP", "ROLLBACK") and key in open_swaps:
            s = open_swaps.pop(key)
            realized = r["stage"] == "post"
            out.append({"knob": r["knob"], "from": s["from_v"], "to": s["to_v"], "state": json.loads(s["state"]),
                        "gate_speedup": s["speedup"], "outcome": r["decision"], "realized": realized,
                        "speedup": r["speedup"] if r["decision"] == "KEEP" else 0.0,
                        "reward": (r["reward"] if realized else 0.0)})
        elif r["decision"] in ("REJECT", "ABORT"):
            out.append({"knob": r["knob"], "from": r["from_v"], "to": r["to_v"], "state": json.loads(r["state"]),
                        "gate_speedup": r["speedup"], "outcome": r["decision"], "realized": False,
                        "speedup": 0.0, "reward": r["reward"]})
    return out


def _feature(state: dict, names) -> float:
    return sum(float(state.get(n, 0.0)) for n in names)


class AdaptPolicy:
    """Threshold learner over realized swap outcomes (per rule)."""

    def __init__(self, min_rows: int = 3):
        self.min_rows = min_rows

    def propose(self, mem_or_rows, base: dict | None = None) -> dict:
        rows = mem_or_rows if isinstance(mem_or_rows, list) else mem_or_rows.adapt_rows()
        ts = transitions(rows)
        prop = dict(base or {})
        why = {}
        for name, (knob, target, feats, default) in RULES.items():
            pts = [(_feature(t["state"], feats), t["reward"]) for t in ts
                   if t["knob"] == knob and t["to"] == target and t["realized"]]
            if len(pts) < self.min_rows:
                prop.setdefault(name, default)
                why[name] = f"default ({len(pts)} realized swaps)"
                continue
            def score(th):
                return sum(rw for f, rw in pts if f >= th)
            best = max(GRID, key=lambda th: (round(score(th), 9), -abs(th - default)))
            if score(best) <= 0 and all(rw <= 0 for _, rw in pts):
                best = min(0.99, max(f for f, _ in pts) + 0.01)   # every realized swap lost: rule off below all seen states
            prop[name] = best
            why[name] = f"{len(pts)} realized swaps, reward at threshold {round(score(best), 3)}"
        # gate threshold: smallest gate speedup whose swaps did not lose on average
        gates = [(t["gate_speedup"], t["reward"]) for t in ts if t["knob"] == "memtable_rep" and t["realized"]]
        if len(gates) >= self.min_rows:
            cands = sorted({round(g, 2) for g, _ in gates} | {1.15})
            ok = [c for c in cands if (lambda xs: xs and sum(xs) / len(xs) >= 0)([rw for g, rw in gates if g >= c])]
            prop["mem_min_speedup"] = min(ok) if ok else round(max(g for g, _ in gates) + 0.01, 2)
            why["mem_min_speedup"] = f"{len(gates)} realized memtable swaps"
        else:
            prop.setdefault("mem_min_speedup", 1.15)
        return {"proposal": prop, "why": why, "transitions": len(ts),
                "realized": sum(1 for t in ts if t["realized"])}


def summary(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        k = f'{r["knob"]}:{r["from_v"]}->{r["to_v"]}:{r["decision"]}'
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
