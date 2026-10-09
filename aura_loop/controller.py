"""Controller pi(y, memory) -> u. MVP: retrieval-only. An LLM controller plugs in later."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .memory import Memory
from .observe import Observation


@dataclass(frozen=True)
class Control:
    symbol: str
    new_src: str
    intent: str
    source: str            # "memory:<id>" | "identity" | "llm:<model>" (later)

    @property
    def is_identity(self) -> bool:
        return self.source == "identity"


class Controller(Protocol):
    def propose(self, y: Observation) -> Control: ...


class RetrievalController:
    """Propose the after_src of the most similar KEEP precedent for the same symbol.

    Candidates that previously ended in ROLLBACK/REJECT for this symbol are skipped,
    so failures written back by the loop change the next decision (closed loop).
    If the observation is healthy, or nothing usable is found, return identity (no-op).
    TODO: LLMController(y, precedents) -> Control, using retrieved precedents as few-shot context.
    """

    def __init__(self, memory: Memory, k: int = 5):
        self.memory, self.k = memory, k

    def propose(self, y: Observation) -> Control:
        if y.healthy:
            return Control(y.symbol, y.define_src, "healthy: no-op", "identity")
        bad = self.memory.failed_afters(y.symbol)
        for p in self.memory.retrieve(y.symbol, y.define_src, k=self.k, verdicts=("KEEP",)):
            if p.after_src in bad or p.after_src == y.define_src:
                continue
            return Control(y.symbol, p.after_src, p.intent or f"replay precedent #{p.id}", f"memory:{p.id}")
        return Control(y.symbol, y.define_src, "no usable precedent (LLM controller: TODO)", "identity")
