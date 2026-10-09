from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .controller import RetrievalController
from .loop import run
from .memory import Memory
from .plant import FakePlant, SoftPlant
from .seeds import seed
from .state import WorkspaceState

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny.scm"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aura_loop", description="Aura closed-loop control MVP")
    ap.add_argument("--demo", action="store_true", help="seed memory and run 3 steps on fixtures/tiny.scm")
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--plant", choices=["fake", "soft"], default="fake")
    ap.add_argument("--no-seed", action="store_true")
    a = ap.parse_args(argv)
    if not a.demo:
        ap.print_help()
        return 0
    mem = Memory(a.db)
    if not a.no_seed and not mem.all():
        ids = seed(mem)
        print(f"seeded memory: {len(ids)} precedents")
    plant = SoftPlant() if a.plant == "soft" else FakePlant()
    x0 = WorkspaceState.from_file(a.fixture, symbol=a.symbol, repo="fixtures")
    x, results = run(x0, RetrievalController(mem), plant, mem, steps=a.steps)
    print(f"\nfinal define: {x.define_src}")
    print(f"final errors: {list(x.type_errors) or 'none'}")
    print(f"memory rows : {len(mem.all())}  history[{x.symbol}]={mem.history(x.symbol)}")
    return 0 if not x.type_errors else 1


if __name__ == "__main__":
    sys.exit(main())
