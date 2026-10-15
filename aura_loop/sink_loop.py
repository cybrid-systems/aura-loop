"""Close the loop with aura-kernel-sink.

    aura-loop proposes (threshold per kernel shape, skip floor)  -> SINK_PROPOSAL
    aura-kernel-sink runs scripts/policy.aura (real Aura, real cc) -> trajectory.jsonl
    aura-loop ingests the new lines into memory                    -> next proposal

`compare()` runs a fixed-threshold arm and a learning arm for the same number
of rounds (separate memories) and reports KEEP / ROLLBACK / SKIP / ABORT, the
speedup the kept kernels deliver, and the summed reward per round.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

from .memory import Memory
from .sink_ingest import ingest
from .sink_policy import SinkPolicy, write_proposal


class FixedPolicy:
    """Baseline: one threshold and one Soft floor for every kernel, never updated."""

    kind = "fixed"

    def __init__(self, threshold: float, floor_us: float = 50.0):
        self.threshold, self.floor_us = threshold, floor_us

    def propose(self, kernels, memory, *, proposal_id: str = "") -> dict:
        return {"proposal": proposal_id, "policy": f"fixed:{self.threshold:g}", "default_threshold": self.threshold,
                "default_floor_us": self.floor_us, "kernels": {}}


class SinkRunner:
    """Runs aura-kernel-sink's scripts/policy_all.sh (scripts/policy.aura, one process per kernel) on real Aura."""

    def __init__(self, sink_repo: str | Path, aura_bin: str | None = None, timeout: int = 1800):
        self.repo = Path(sink_repo).resolve()
        self.aura_bin = aura_bin or os.environ.get("AURA_BIN")
        self.timeout = timeout
        if not self.aura_bin:
            raise RuntimeError("SinkRunner needs AURA_BIN")

    def info(self) -> dict:
        def git(*a):
            try:
                return subprocess.run(["git", "-C", str(self.repo), *a], capture_output=True, text=True).stdout.strip()
            except OSError:
                return ""
        return {"kind": "aura-kernel-sink scripts/policy_all.sh", "sink_repo": str(self.repo),
                "sink_head": git("rev-parse", "--short", "HEAD"), "aura_bin": self.aura_bin}

    def run(self, proposal_path: str, round_no: int, traj_path: str) -> str:
        env = dict(os.environ, AURA_BIN=self.aura_bin, SINK_PROPOSAL=str(proposal_path), SINK_ROUND=str(round_no),
                   SINK_TRAJECTORY=str(traj_path))
        # one Aura process per kernel (aura-kernel-sink scripts/policy_all.sh)
        p = subprocess.run(["bash", "scripts/policy_all.sh"],
                           cwd=self.repo, env=env, capture_output=True, text=True, timeout=self.timeout)
        if p.returncode != 0 or not Path(traj_path).exists():
            raise RuntimeError(f"policy.aura failed rc={p.returncode}: {p.stdout[-2000:]}{p.stderr[-2000:]}")
        return p.stdout


class FakeSinkRunner:
    """In-process stand-in with the same proposal semantics as scripts/policy.aura (tests, no Aura).

    kernels: {name: (features, soft_us, true_speedup, compilable)}; measured
    speedup = true * (1 +- noise).
    """

    def __init__(self, kernels: dict, noise: float = 0.1, seed: int = 1, default_threshold: float = 2.0):
        self.kernels, self.noise, self.rng = kernels, noise, random.Random(seed)
        self.default_threshold = default_threshold

    def info(self) -> dict:
        return {"kind": "FakeSinkRunner"}

    def run(self, proposal_path: str, round_no: int, traj_path: str) -> str:
        prop = json.loads(Path(proposal_path).read_text())
        dthr = prop.get("default_threshold") or self.default_threshold
        lines = []
        for name, (feat, soft_us, true_sp, ok) in self.kernels.items():
            k = prop.get("kernels", {}).get(name, {})
            thr = k.get("threshold", dthr)
            fl = k.get("floor_us", prop.get("default_floor_us", 50.0))
            row = {"ts": int(time.time() * 1000) + len(lines), "name": name, "kinds": [], "threshold": thr,
                   "floor_us": fl, "action": k.get("action", "sink"), "features": feat, "speedup": 0.0,
                   "soft_us_per_call": soft_us, "extra": ["round", round_no, "proposal", prop.get("proposal", "")]}
            if k.get("action") == "skip":
                row.update(decision="SKIP", stage="policy", soft_us_per_call=0.0)
            elif soft_us < fl:
                row.update(decision="SKIP", stage="profile")
            elif not ok:
                row.update(decision="SKIP", stage="compile")
            else:
                sp = true_sp * (1 + self.rng.uniform(-self.noise, self.noise))
                row.update(speedup=sp, stage="score", decision="KEEP" if sp >= thr else "ROLLBACK")
            lines.append(json.dumps(row))
        with open(traj_path, "a") as f:
            f.write("\n".join(lines) + "\n")
        return ""


def latest_kernels(mem: Memory) -> list[dict]:
    """Every kernel seen so far, with its latest known shape and Soft cost."""
    out: dict[str, dict] = {}
    for r in mem.sink_rows("ORDER BY id"):
        st = json.loads(r["state"])
        raw = json.loads(r["raw"])
        prev = out.get(r["kernel"], {})
        out[r["kernel"]] = {"name": r["kernel"], "features": raw.get("features"), "kinds": st.get("kinds", []),
                            "soft_us": st["soft_us"] if st.get("soft_us") else prev.get("soft_us", 0.0)}
    return list(out.values())


def round_metrics(rows: list[dict]) -> dict:
    by = {d: sum(1 for r in rows if r["decision"] == d) for d in ("KEEP", "ROLLBACK", "SKIP", "ABORT")}
    kept = [r["speedup"] for r in rows if r["decision"] == "KEEP"]
    return {**{k.lower(): v for k, v in by.items()},
            "kept_speedup_sum": round(sum(kept), 2),
            "kept_speedup_geomean": round(math.exp(sum(math.log(s) for s in kept) / len(kept)), 2) if kept else 0.0,
            "reward": round(sum(r["reward"] for r in rows), 3),
            "kept": sorted(r["kernel"] for r in rows if r["decision"] == "KEEP"),
            "rolled_back": sorted(r["kernel"] for r in rows if r["decision"] == "ROLLBACK")}


def run_arm(name: str, policy, runner, rounds: int, mem: Memory, workdir: str | Path, log=None) -> list[dict]:
    workdir = Path(workdir).resolve()  # the runner's cwd is the sink repo
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in range(1, rounds + 1):
        pid = f"{name}-r{r}"
        prop = policy.propose(latest_kernels(mem), mem, proposal_id=pid)
        ppath, tpath = workdir / f"{pid}.proposal.json", workdir / f"{pid}.jsonl"
        write_proposal(str(ppath), prop)
        if tpath.exists():
            tpath.unlink()
        t0 = time.time()
        runner.run(str(ppath), r, str(tpath))
        ids = ingest(mem, tpath, arm=name)
        rows = mem.sink_rows(f"WHERE id IN ({','.join('?' * len(ids))})", tuple(ids)) if ids else []
        m = {"arm": name, "round": r, "proposal": prop, "seconds": round(time.time() - t0, 1), **round_metrics(rows)}
        out.append(m)
        if log:
            log(f"[{name}] round {r}: KEEP {m['keep']} ROLLBACK {m['rollback']} SKIP {m['skip']} ABORT {m['abort']} "
                f"kept-speedup-sum {m['kept_speedup_sum']} reward {m['reward']} ({m['seconds']}s)")
    return out


def compare(runner, rounds: int = 4, baseline_threshold: float = 10.0, learner: str = "knn",
            workdir: str | Path | None = None, log=None, db_dir: str | None = None) -> dict:
    workdir = Path(workdir or tempfile.mkdtemp(prefix="aura-loop-sink-"))
    dbs = {a: (str(Path(db_dir) / f"sink-{a}.db") if db_dir else ":memory:") for a in ("fixed", learner)}
    fixed_mem, learn_mem = Memory(dbs["fixed"]), Memory(dbs[learner])
    fixed = run_arm("fixed", FixedPolicy(baseline_threshold), runner, rounds, fixed_mem, workdir, log)
    # The learner starts from the same prior as the baseline (round 1 is identical).
    learned = run_arm(learner, SinkPolicy(learner, default_threshold=baseline_threshold), runner, rounds, learn_mem,
                      workdir, log)
    def total(arm, k):
        return round(sum(m[k] for m in arm), 3)
    summary = {a: {k: total(arm, k) for k in ("keep", "rollback", "skip", "abort", "kept_speedup_sum", "reward")}
               for a, arm in (("fixed", fixed), (learner, learned))}
    after1 = {a: {k: round(sum(m[k] for m in arm[1:]), 3) for k in ("keep", "rollback", "kept_speedup_sum", "reward")}
              for a, arm in (("fixed", fixed), (learner, learned))}
    replay = len(SinkPolicy.replay(learn_mem))
    fixed_mem.close(), learn_mem.close()
    return {"runner": runner.info(), "rounds": rounds, "baseline_threshold": baseline_threshold, "learner": learner,
            "reward_model": {"KEEP": "log(speedup)", "ROLLBACK": -1.0, "ABORT": -2.0, "SKIP": 0.0,
                             "SKIP@compile": -0.1},
            "per_round": {"fixed": fixed, learner: learned}, "totals": summary, "totals_after_round1": after1,
            "replay_tuples": replay, "workdir": str(workdir)}
