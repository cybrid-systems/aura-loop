"""Observation y = phi(x): the projection of state the controller is allowed to see."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .state import WorkspaceState


@dataclass(frozen=True)
class Observation:
    repo: str
    symbol: str
    define_src: str
    params: tuple[str, ...]
    type_errors: tuple[str, ...]                 # static check on x (toy arity table)
    state_sha: str
    history: dict = field(default_factory=dict)  # verdict -> count for this symbol
    source: str = ""                             # file text the controller may read (edsl-patch input.source)
    gate_errors: tuple[str, ...] = ()            # plant probe on x: e.g. failing Aura tests

    @property
    def errors(self) -> tuple[str, ...]:
        return self.type_errors + self.gate_errors

    @property
    def healthy(self) -> bool:
        return not self.errors

    @property
    def intent(self) -> str:
        """Task sentence for the controller, derived only from observed errors."""
        if self.healthy:
            return ""
        return f"Fix {self.symbol} without changing its signature: " + "; ".join(self.errors)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("source")
        d["intent"] = self.intent
        d["healthy"] = self.healthy
        return d


def observe(x: WorkspaceState, memory=None, plant=None, static: bool = True) -> Observation:
    """static=False drops the toy arity check (the real plant's probe is then the only error source)."""
    hist = memory.history(x.symbol) if memory is not None else {}
    probe = getattr(plant, "probe", None)
    gate = tuple(probe(x)) if callable(probe) else ()
    return Observation(repo=x.repo, symbol=x.symbol, define_src=x.define_src, params=x.params,
                       type_errors=x.type_errors if static else (), state_sha=x.sha, history=hist, source=x.source,
                       gate_errors=gate)
