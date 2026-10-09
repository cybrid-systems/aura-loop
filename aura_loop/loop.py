"""One control step: x -> y=observe(x) -> u=controller(y) -> plant(x,u) -> verdict, x' -> memory writeback."""
from __future__ import annotations

from dataclasses import dataclass

from .controller import Control, Controller
from .memory import Memory
from .observe import Observation, observe
from .plant import Plant
from .state import WorkspaceState


@dataclass(frozen=True)
class StepResult:
    x: WorkspaceState
    y: Observation
    u: Control
    verdict: str          # KEEP | ROLLBACK | REJECT | NOOP
    reason: str
    memory_id: int | None
    x_next: WorkspaceState

    def as_dict(self) -> dict:
        return {"state_before": self.x.as_dict(), "observation": self.y.as_dict(), "control": self.u.as_dict(),
                "verdict": self.verdict, "reason": self.reason, "memory_id": self.memory_id,
                "state_after": self.x_next.as_dict()}


def step(x: WorkspaceState, ctrl: Controller, plant: Plant, mem: Memory, static: bool = True) -> StepResult:
    y = observe(x, mem, plant, static=static)
    u = ctrl.propose(y)
    if u.is_identity:
        return StepResult(x, y, u, "NOOP", u.intent, None, x)
    r = plant.apply(x, u)
    mid = mem.writeback(repo=x.repo, symbol=x.symbol, before_src=x.define_src, after_src=u.new_src,
                        intent=u.intent, verdict=r.verdict, reason=r.reason, sha=x.sha)
    return StepResult(x, y, u, r.verdict, r.reason, mid, r.next_state.with_history(mid))


def run(x: WorkspaceState, ctrl: Controller, plant: Plant, mem: Memory, steps: int = 3, log=print,
        static: bool = True, stop_on_noop: bool = False):
    results = []
    for i in range(steps):
        r = step(x, ctrl, plant, mem, static=static)
        results.append(r)
        if log:
            log(f"\n── step {i + 1}  state={r.y.state_sha}  history={r.y.history}")
            log(f"   y.define : {r.y.define_src}")
            log(f"   y.errors : {list(r.y.errors) or 'none'}")
            log(f"   u        : [{r.u.source}] {r.u.intent}")
            if not r.u.is_identity:
                log(f"   u.src    : {r.u.new_src}")
            log(f"   verdict  : {r.verdict} ({r.reason})" + (f" -> memory #{r.memory_id}" if r.memory_id else ""))
        x = r.x_next
        if stop_on_noop and r.verdict == "NOOP":
            break
    return x, results
