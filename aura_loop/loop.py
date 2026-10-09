"""One control step: observe -> controller -> plant -> writeback."""
from __future__ import annotations

from dataclasses import dataclass

from .controller import Control, Controller
from .memory import Memory
from .observe import Observation, observe
from .plant import Plant
from .state import WorkspaceState


@dataclass(frozen=True)
class StepResult:
    y: Observation
    u: Control
    verdict: str          # KEEP | ROLLBACK | REJECT | NOOP
    reason: str
    memory_id: int | None
    x_next: WorkspaceState


def step(x: WorkspaceState, ctrl: Controller, plant: Plant, mem: Memory) -> StepResult:
    y = observe(x, mem)
    u = ctrl.propose(y)
    if u.is_identity:
        return StepResult(y, u, "NOOP", u.intent, None, x)
    r = plant.apply(x, u)
    mid = mem.writeback(repo=x.repo, symbol=x.symbol, before_src=x.define_src, after_src=u.new_src,
                        intent=u.intent, verdict=r.verdict, reason=r.reason, sha=x.sha)
    return StepResult(y, u, r.verdict, r.reason, mid, r.next_state.with_history(mid))


def run(x: WorkspaceState, ctrl: Controller, plant: Plant, mem: Memory, steps: int = 3, log=print):
    results = []
    for i in range(steps):
        r = step(x, ctrl, plant, mem)
        results.append(r)
        if log:
            log(f"\n── step {i + 1}  state={r.y.state_sha}  history={r.y.history}")
            log(f"   y.define : {r.y.define_src}")
            log(f"   y.errors : {list(r.y.type_errors) or 'none'}")
            log(f"   u        : [{r.u.source}] {r.u.intent}")
            if not r.u.is_identity:
                log(f"   u.src    : {r.u.new_src}")
            log(f"   verdict  : {r.verdict} ({r.reason})" + (f" -> memory #{r.memory_id}" if r.memory_id else ""))
        x = r.x_next
    return x, results
