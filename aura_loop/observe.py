"""Observation y = phi(x): the projection of state the controller is allowed to see."""
from __future__ import annotations

from dataclasses import dataclass, field

from .state import WorkspaceState


@dataclass(frozen=True)
class Observation:
    repo: str
    symbol: str
    define_src: str
    params: tuple[str, ...]
    type_errors: tuple[str, ...]
    state_sha: str
    history: dict = field(default_factory=dict)   # verdict -> count for this symbol

    @property
    def healthy(self) -> bool:
        return not self.type_errors


def observe(x: WorkspaceState, memory=None) -> Observation:
    hist = memory.history(x.symbol) if memory is not None else {}
    return Observation(repo=x.repo, symbol=x.symbol, define_src=x.define_src, params=x.params,
                       type_errors=x.type_errors, state_sha=x.sha, history=hist)
