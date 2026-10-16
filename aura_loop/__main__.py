from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .controller import RetrievalController
from .loop import run
from .memory import Memory
from .plant import FakePlant, SoftPlant, tests_for
from .seeds import seed
from .state import WorkspaceState

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny.scm"


def e2e(a) -> int:
    from . import edsl_bridge
    from .edsl_controller import EdslPatchController

    plant_name = a.plant or "soft"
    if plant_name == "soft" and not (a.aura_bin or os.environ.get("AURA_BIN")):
        print("e2e: the default plant is real Aura (SoftPlant) and needs AURA_BIN=/path/to/aura "
              "(+ AURA_LIB=<aura>/lib). Pass --plant fake to run against the in-process FakePlant.", file=sys.stderr)
        return 2
    mem = Memory(a.db)
    if not a.no_seed and not mem.all():
        seed(mem)
    if plant_name == "soft":
        plant = SoftPlant(aura_bin=a.aura_bin, tests=a.tests or tests_for(a.fixture))
    else:
        plant = FakePlant()
    ctrl = EdslPatchController(mem, policy=a.policy)
    x0 = WorkspaceState.from_file(a.fixture, symbol=a.symbol, repo="fixtures")
    log = (lambda s: print(s, file=sys.stderr)) if a.verbose else None
    x, results = run(x0, ctrl, plant, mem, steps=a.steps, log=log, static=(plant_name == "fake"),
                     stop_on_noop=True)
    e = edsl_bridge.load()
    out = {
        "plant": plant_name,
        "plant_info": plant.info() if hasattr(plant, "info") else {"kind": "FakePlant (toy arity gate)"},
        "controller": {"kind": "edsl-patch", "policy": ctrl.policy, "edsl_patch_dir": str(e.root),
                       "edsl_patch_sha": e.sha},
        "transitions": [r.as_dict() for r in results],
        "final_state": x.as_dict(),
        "converged": bool(results) and results[-1].verdict in ("NOOP", "KEEP") and
                     any(r.verdict == "KEEP" for r in results),
        "memory_rows": len(mem.all()),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["converged"] else 1


def sink(a) -> int:
    from .memory import Memory
    from .sink_ingest import ingest
    from .sink_loop import SinkRunner, compare
    from .sink_policy import SinkPolicy

    if a.sink_ingest:
        mem = Memory(a.db)
        ids = []
        for f in a.sink_ingest:
            ids += ingest(mem, f, arm=a.sink_arm)
        rows = mem.sink_rows()
        print(json.dumps({"ingested": len(ids), "rows": len(rows),
                          "replay_tuples": len(SinkPolicy.replay(mem))}, indent=2))
        return 0
    repo = a.sink_repo or os.environ.get("SINK_REPO")
    if not repo:
        print("--sink needs --sink-repo (or SINK_REPO) pointing at an aura-kernel-sink checkout", file=sys.stderr)
        return 2
    runner = SinkRunner(repo, aura_bin=a.aura_bin)
    log = (lambda s: print(s, file=sys.stderr)) if a.verbose else None
    out = compare(runner, rounds=a.rounds, baseline_threshold=a.baseline_threshold, learner=a.learner,
                  workdir=a.sink_workdir, log=log, db_dir=a.sink_db_dir)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


def adapt(a) -> int:
    from .adapt_ingest import AdaptPolicy, ingest, summary
    from .memory import Memory

    mem = Memory(a.db)
    ids = []
    for f in a.adapt_ingest or []:
        ids += ingest(mem, f, arm=a.adapt_arm)
    rows = mem.adapt_rows()
    out = {"ingested": len(ids), "rows": len(rows), "decisions": summary(rows)}
    if a.adapt_propose:
        p = AdaptPolicy().propose(mem)
        Path(a.adapt_propose).write_text(json.dumps(p["proposal"], indent=2) + "\n")
        out["proposal"] = p
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aura_loop", description="Aura closed-loop control")
    ap.add_argument("--e2e", action="store_true",
                    help="closed loop x->y->u->x' with the edsl-patch controller; prints JSON transitions")
    ap.add_argument("--demo", action="store_true", help="legacy: retrieval controller + FakePlant, 3 steps")
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--tests", default=None, help="Aura test file (default: <fixture>.tests.scm)")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--plant", choices=["fake", "soft"], default=None,
                    help="e2e default: soft (real Aura, needs AURA_BIN); demo default: fake")
    ap.add_argument("--aura-bin", default=None)
    ap.add_argument("--policy", choices=["auto", "minimax", "retrieval"], default="auto",
                    help="edsl-patch controller policy (auto = minimax if configured, else retrieval)")
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="e2e: human trace on stderr")
    sk = ap.add_argument_group("aura-kernel-sink loop")
    sk.add_argument("--sink", action="store_true",
                    help="closed loop with aura-kernel-sink: fixed-threshold arm vs learning arm, JSON report")
    sk.add_argument("--sink-repo", default=None, help="aura-kernel-sink checkout (or $SINK_REPO)")
    sk.add_argument("--sink-ingest", nargs="*", default=None, help="only ingest trajectory files into --db")
    sk.add_argument("--sink-arm", default="ingest", help="arm label for --sink-ingest rows")
    sk.add_argument("--rounds", type=int, default=4)
    sk.add_argument("--baseline-threshold", type=float, default=10.0)
    sk.add_argument("--learner", choices=["knn", "bandit"], default="knn")
    sk.add_argument("--sink-workdir", default=None, help="proposals + trajectories per round (default: tmp)")
    sk.add_argument("--sink-db-dir", default=None, help="keep the per-arm SQLite memories here")
    sk.add_argument("--out", default=None, help="also write the JSON report here")
    ad = ap.add_argument_group("aura-rocks structure switching")
    ad.add_argument("--adapt-ingest", nargs="*", default=None,
                    help="ingest aura-rocks adapt trajectories (build/adapt-traj.jsonl) into --db")
    ad.add_argument("--adapt-arm", default="ingest", help="arm label for --adapt-ingest rows")
    ad.add_argument("--adapt-propose", default=None,
                    help="write learned thresholds (ADAPT_PROPOSAL json for aura-rocks) to this path")
    a = ap.parse_args(argv)
    if a.adapt_ingest is not None or a.adapt_propose:
        return adapt(a)
    if a.sink or a.sink_ingest is not None:
        return sink(a)
    if a.e2e:
        return e2e(a)
    if not a.demo:
        ap.print_help()
        return 0
    mem = Memory(a.db)
    if not a.no_seed and not mem.all():
        ids = seed(mem)
        print(f"seeded memory: {len(ids)} precedents")
    plant = SoftPlant(aura_bin=a.aura_bin, tests=a.tests or tests_for(a.fixture)) if a.plant == "soft" else FakePlant()
    x0 = WorkspaceState.from_file(a.fixture, symbol=a.symbol, repo="fixtures")
    x, results = run(x0, RetrievalController(mem), plant, mem, steps=a.steps)
    print(f"\nfinal define: {x.define_src}")
    print(f"final errors: {list(x.type_errors) or 'none'}")
    print(f"memory rows : {len(mem.all())}  history[{x.symbol}]={mem.history(x.symbol)}")
    return 0 if not x.type_errors else 1


if __name__ == "__main__":
    sys.exit(main())
