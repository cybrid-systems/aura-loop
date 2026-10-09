"""Seed precedents for fixtures/tiny.scm (symbol `clamp`)."""
from __future__ import annotations

from .memory import Memory

BUGGY = "(define (clamp x lo hi) (max lo (min x)))"

SEEDS = [
    # Most similar before_src, but it changes the signature -> the plant will REJECT it.
    dict(repo="example/other-repo", symbol="clamp", before_src=BUGGY,
         after_src="(define (clamp x range) (max (car range) (min x (cdr range))))",
         intent="refactor clamp to take a (lo . hi) pair", verdict="KEEP", reason="kept upstream", sha="seed0001"),
    # The real fix: pass the upper bound to min.
    dict(repo="example/aura-pad", symbol="clamp", before_src="(define (clamp x lo hi) (let ((m (min x))) (max lo m)))",
         after_src="(define (clamp x lo hi) (max lo (min x hi)))",
         intent="fix arity: min needs both x and hi", verdict="KEEP", reason="tests pass", sha="seed0002"),
    # A known-bad attempt (negative prior): never propose this again.
    dict(repo="example/aura-pad", symbol="clamp", before_src=BUGGY,
         after_src="(define (clamp x lo hi) (min lo (max x)))",
         intent="swap min/max", verdict="ROLLBACK", reason="arity: (max x)", sha="seed0003"),
]


def seed(mem: Memory) -> list[int]:
    return [mem.writeback(ts=1_700_000_000 + i, **s) for i, s in enumerate(SEEDS)]
