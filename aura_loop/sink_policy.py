"""Policy over aura-kernel-sink trajectories: per-shape speedup threshold + skip floor.

Simple on purpose. Two learners share the same memory (`sink_transitions`):

  retrieval (kNN)  — precedents of the kernel's shape bucket (vsum and
                     vsum-short differ: other length band) decide; with no
                     rows in the bucket, the nearest bucket by feature vector
                     does. Measured speedups set the threshold (margin x the
                     worst one) or, below min_worth, a skip floor; see
                     `_decide`.
  bandit           — arms are (action, threshold, floor) drawn from a fixed
                     grid. Per bucket the arm with the best mean reward is
                     greedy; epsilon of the time a random arm is tried, so the
                     grid keeps being sampled. RL hook: `replay()` returns the
                     (state, action, reward, next_state) tuples a later learner
                     would train on.

`propose(kernels, memory)` returns the JSON proposal `scripts/policy.aura`
reads (`SINK_PROPOSAL`): a default threshold (hot-swapped into the workspace
policy define) plus per-kernel action / threshold / floor.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .sink_ingest import feature_vector, len_band, parse_features, shape_key

# Candidate speedup thresholds and Soft floors (us/call).
THRESHOLD_GRID = (2.0, 4.0, 8.0, 16.0)
FLOOR_GRID = (50.0, 500.0, 2000.0)


@dataclass(frozen=True)
class Decision:
    kernel: str
    action: str          # "sink" | "skip"
    threshold: float
    floor_us: float
    source: str          # "knn:bucket:v1|+|len4^6" | "knn:near:..." | "bandit:greedy:..." | "default"
    precedent_ids: tuple = ()

    def as_dict(self) -> dict:
        return {"action": self.action, "threshold": self.threshold, "floor_us": self.floor_us,
                "source": self.source, "precedents": list(self.precedent_ids)}


def _state_of(row: dict) -> dict:
    return json.loads(row["state"])


def _dist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _measured(rows: list[dict]) -> list[float]:
    """Speedups that were actually measured (KEEP, or ROLLBACK at the score stage)."""
    return [r["speedup"] for r in rows if r["speedup"] > 0 and r["decision"] in ("KEEP", "ROLLBACK")
            and r["stage"] == "score"]


def _decide(rows: list[dict], kernel: str, source: str, *, default_threshold: float, default_floor: float,
            margin: float, min_threshold: float, min_worth: float) -> Decision:
    """Decision for a kernel from the precedents of its shape bucket.

    - measured speedups exist and the best one is below `min_worth`: the shape is
      not worth an FFI boundary -> learned skip floor: floor_us above the Soft
      cost seen for this shape, so kernel-sink's own profile stage SKIPs it
      (no compile, no mutation) and any cheaper kernel of this shape too;
    - measured speedups exist: threshold = margin * the worst measured speedup
      (clamped at min_threshold): the shape keeps its kernels, and a kernel
      that falls well below what the shape delivers is still rolled back;
    - only compile SKIPs: vkgen cannot compile this shape -> skip;
    - otherwise the default.
    """
    ids = tuple(r["id"] for r in rows[-5:])
    sp = _measured(rows)
    if sp:
        if max(sp) < min_worth:
            soft = max(r["soft_us"] for r in rows if r["soft_us"] > 0)
            return Decision(kernel, "sink", default_threshold, round(max(default_floor, soft * 1.5), 1),
                            source + ":floor", ids)
        return Decision(kernel, "sink", round(max(min_threshold, margin * min(sp)), 2), default_floor, source, ids)
    if rows and all(r["decision"] == "SKIP" and r["stage"] == "compile" for r in rows):
        return Decision(kernel, "skip", default_threshold, default_floor, source + ":uncompilable", ids)
    return Decision(kernel, "sink", default_threshold, default_floor, source + ":no-measurement", ids)


class SinkPolicy:
    def __init__(self, kind: str = "knn", *, default_threshold: float = 2.0, default_floor: float = 50.0,
                 epsilon: float = 0.3, rng: random.Random | None = None, margin: float = 0.7,
                 min_threshold: float = 1.2, min_worth: float = 1.5):
        if kind not in ("knn", "bandit"):
            raise ValueError(kind)
        self.kind = kind
        self.default_threshold = default_threshold
        self.default_floor = default_floor
        self.epsilon = epsilon
        self.rng = rng or random.Random(0)
        self.margin, self.min_threshold, self.min_worth = margin, min_threshold, min_worth

    def _kw(self) -> dict:
        return {"default_threshold": self.default_threshold, "default_floor": self.default_floor,
                "margin": self.margin, "min_threshold": self.min_threshold, "min_worth": self.min_worth}

    def decide(self, kernel: str, features, soft_us: float, memory, kinds=None) -> Decision:
        state = parse_features(features)
        state["kinds"] = list(kinds or [])
        state["soft_us"] = float(soft_us or 0.0)
        bucket = shape_key(state)
        rows = memory.sink_rows("WHERE shape=?", (bucket,))
        if self.kind == "bandit":
            return self._bandit(kernel, bucket, rows)
        own = memory.sink_rows("WHERE kernel=? ORDER BY id", (kernel,))
        if own and own[-1]["decision"] == "SKIP" and own[-1]["stage"] == "compile" and \
                _state_of(own[-1]).get("ops") == state.get("ops"):
            return Decision(kernel, "skip", self.default_threshold, self.default_floor,
                            "knn:kernel:uncompilable", (own[-1]["id"],))
        if rows:
            return _decide(rows, kernel, f"knn:bucket:{bucket}", **self._kw())
        near = self._nearest(state, memory)
        if near:
            return _decide(near, kernel, f"knn:near:{shape_key(_state_of(near[0]))}", **self._kw())
        return Decision(kernel, "sink", self.default_threshold, self.default_floor, "default")

    def _nearest(self, state: dict, memory, k: int = 5) -> list[dict]:
        """Rows from the nearest OTHER buckets by feature vector (no rows in this bucket)."""
        mine = shape_key(state)
        by_bucket: dict[str, list[dict]] = {}
        for r in memory.sink_rows():
            if r["shape"] != mine:
                by_bucket.setdefault(r["shape"], []).append(r)
        if not by_bucket:
            return []
        v = feature_vector(state)
        ranked = sorted(by_bucket, key=lambda b: _dist(v, feature_vector(_state_of(by_bucket[b][0]))))
        return by_bucket[ranked[0]][:k]

    def _bandit(self, kernel: str, bucket: str, rows: list[dict]) -> Decision:
        arms = [("sink", t, f) for t in THRESHOLD_GRID for f in FLOOR_GRID] + [("skip", self.default_threshold, f)
                                                                                 for f in FLOOR_GRID]
        stats: dict[tuple, list[float]] = {}
        for r in rows:
            stats.setdefault((r["action"], r["threshold"], r["floor_us"]), []).append(r["reward"])
        if not stats or self.rng.random() < self.epsilon:
            action, t, f = self.rng.choice(arms)
            return Decision(kernel, action, t, f, f"bandit:explore:{bucket}")
        (action, t, f), rewards = max(stats.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        return Decision(kernel, action, t, f, f"bandit:greedy:{bucket}")

    def propose(self, kernels: list[dict], memory, *, proposal_id: str = "") -> dict:
        """kernels: [{name, features, soft_us, kinds}] -> the SINK_PROPOSAL JSON."""
        out = {}
        for k in kernels:
            d = self.decide(k["name"], k.get("features"), k.get("soft_us", 0.0), memory, k.get("kinds"))
            out[k["name"]] = d.as_dict()
        return {"proposal": proposal_id, "policy": self.kind, "default_threshold": self.default_threshold,
                "default_floor_us": self.default_floor, "kernels": out}

    @staticmethod
    def replay(memory) -> list[dict]:
        """(state, action, reward, next_state) tuples, one per consecutive round of a kernel.

        Hook for a later RL learner; nothing here trains on them.
        """
        rows = memory.sink_rows("ORDER BY kernel, round, id")
        out = []
        for a, b in zip(rows, rows[1:]):
            if a["kernel"] == b["kernel"]:
                out.append({"kernel": a["kernel"], "state": json.loads(a["state"]),
                            "action": {"action": a["action"], "threshold": a["threshold"], "floor_us": a["floor_us"]},
                            "reward": a["reward"], "decision": a["decision"],
                            "next_state": json.loads(b["state"])})
        return out


def write_proposal(path: str, proposal: dict) -> None:
    with open(path, "w") as f:
        json.dump(proposal, f, indent=2, sort_keys=True)
