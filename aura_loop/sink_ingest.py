"""Ingest aura-kernel-sink trajectories (build/trajectory.jsonl) into aura-loop memory.

One JSON line per sink decision (aura-kernel-sink lib/policy.aura) becomes one
(state, action, reward) row in `sink_transitions`:

  state  = kernel shape: arity, number of vector params, vector length, op mix
           (from the row's `features` = [arity, nvec, vec_len, ["op:count", ...]])
           + kinds + measured Soft cost (us/call)
  action = sink | skip, with the speedup threshold and Soft floor that were in force
  reward = reward(decision, speedup)  (see `reward`)

The shape bucket (`shape_key`) groups kernels that should behave alike: same
vector count, same math ops, same vector-length band. Retrieval and the
bandit both work per bucket.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DECISIONS = ("KEEP", "ROLLBACK", "SKIP", "ABORT")
MATH_OPS = ("*", "/", "sqrt", "abs", "max", "min", "quotient", "remainder", "modulo", "exact->inexact")
CMP_OPS = ("<", ">", "<=", ">=", "=")

# Reward (documented in README):
#   KEEP      log(speedup)        what the kept kernel buys (> 0 once speedup > 1)
#   ROLLBACK  -ROLLBACK_COST      compile + bench + two mutations + heal, for nothing
#   ABORT     -ABORT_COST         gate mismatch: a wrong kernel was built
#   SKIP      0                   nothing spent (profile / policy skips) ...
#   SKIP at stage "compile" costs a little (vkgen ran, maybe cc): -COMPILE_SKIP_COST
ROLLBACK_COST = 1.0
ABORT_COST = 2.0
COMPILE_SKIP_COST = 0.1


def reward(decision: str, speedup: float, stage: str = "") -> float:
    if decision == "KEEP":
        return math.log(max(speedup, 1e-9))
    if decision == "ROLLBACK":
        return -ROLLBACK_COST
    if decision == "ABORT":
        return -ABORT_COST
    return -COMPILE_SKIP_COST if stage == "compile" else 0.0


def parse_features(feat) -> dict:
    """[arity, nvec, vec_len, ["op:count", ...]] -> dict (missing parts -> 0 / {})."""
    arity, nvec, vlen, ops = 0, 0, 0, {}
    if isinstance(feat, list):
        if len(feat) > 0 and isinstance(feat[0], (int, float)):
            arity = int(feat[0])
        if len(feat) > 1 and isinstance(feat[1], (int, float)):
            nvec = int(feat[1])
        if len(feat) > 2 and isinstance(feat[2], (int, float)):
            vlen = int(feat[2])
        if len(feat) > 3 and isinstance(feat[3], list):
            for item in feat[3]:
                op, _, n = str(item).rpartition(":")
                if op:
                    ops[op] = ops.get(op, 0) + int(n or 0)
    return {"arity": arity, "nvec": nvec, "vec_len": vlen, "ops": ops}


def len_band(n: int) -> int:
    """Vector length band: floor(log4 n) (128 -> 3, 1000 -> 4, 5000 -> 6); 0 for no vector."""
    return int(math.log(n, 4)) if n > 0 else 0


def shape_key(state: dict) -> str:
    ops = state.get("ops", {})
    math_ops = sorted(o for o in ops if o in MATH_OPS)
    branches = ops.get("if", 0) - 1  # one `if` is the loop test
    tags = math_ops + (["cond"] if branches > 0 else [])
    return f"v{state.get('nvec', 0)}|{'+'.join(tags) or 'add'}|len4^{len_band(state.get('vec_len', 0))}"


def feature_vector(state: dict) -> list[float]:
    """Numeric embedding for kNN across buckets (unseen shapes)."""
    ops = state.get("ops", {})
    return [float(state.get("nvec", 0)), float(state.get("arity", 0)), float(len_band(state.get("vec_len", 0))),
            float(sum(ops.get(o, 0) for o in MATH_OPS)), float(ops.get("sqrt", 0) + ops.get("abs", 0)),
            float(max(ops.get("if", 0) - 1, 0)), float(ops.get("f64vector-ref", 0)),
            math.log10(max(state.get("soft_us", 0.0), 1.0))]


def to_transition(line: str, *, repo: str = "aura-kernel-sink", arm: str = "") -> dict | None:
    """One trajectory line -> a sink_transitions row (None for a blank or non-sink line)."""
    line = line.strip()
    if not line:
        return None
    r = json.loads(line)
    if r.get("decision") not in DECISIONS or "name" not in r:
        return None
    extra = r.get("extra") or []
    ex = {extra[i]: extra[i + 1] for i in range(0, len(extra) - 1, 2)} if isinstance(extra, list) else {}
    state = parse_features(r.get("features"))
    state["kinds"] = r.get("kinds") or []
    state["soft_us"] = float(r.get("soft_us_per_call") or 0.0)
    speedup = float(r.get("speedup") or 0.0)
    return {
        "repo": repo, "kernel": r["name"], "shape": shape_key(state), "state": json.dumps(state, sort_keys=True),
        "action": "skip" if r.get("action") == "skip" else "sink",
        "threshold": float(r.get("threshold") or 0.0), "floor_us": float(r.get("floor_us") or 0.0),
        "decision": r["decision"], "stage": r.get("stage", ""), "speedup": speedup, "soft_us": state["soft_us"],
        "reward": reward(r["decision"], speedup, r.get("stage", "")),
        "round": int(ex.get("round", 0) or 0), "arm": arm, "proposal": str(ex.get("proposal", "") or ""),
        "raw": line, "digest": hashlib.sha256(f"{arm}\n{line}".encode()).hexdigest(),
        "ts": float(r.get("ts", 0)) / 1000.0,
    }


def ingest(mem, path: str | Path, *, repo: str = "aura-kernel-sink", arm: str = "") -> list[int]:
    """Append every new line of a trajectory file to memory; returns the new row ids (re-ingest is a no-op)."""
    ids = []
    for line in Path(path).read_text().splitlines():
        t = to_transition(line, repo=repo, arm=arm)
        if t is not None:
            i = mem.sink_writeback(t)
            if i is not None:
                ids.append(i)
    return ids
