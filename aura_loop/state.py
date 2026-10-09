"""State x: a replayable snapshot of the code world (MVP: one define in one file)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .sexpr import ParseError, arity_errors, define_header, parse_all, to_src


@dataclass(frozen=True)
class WorkspaceState:
    repo: str
    path: str
    symbol: str
    source: str                      # full file text
    define_src: str                  # current source of the target define
    params: tuple[str, ...] = ()
    type_errors: tuple[str, ...] = ()
    history_ptr: int | None = None   # last memory row id written for this symbol
    step: int = 0

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]

    @staticmethod
    def check(define_src: str) -> tuple[str, ...]:
        try:
            forms = parse_all(define_src)
        except ParseError as e:
            return (f"syntax: {e}",)
        return tuple(e for f in forms for e in arity_errors(f))

    @classmethod
    def from_source(cls, source: str, symbol: str | None = None, repo: str = "local", path: str = "<mem>") -> "WorkspaceState":
        forms = parse_all(source)
        for f in forms:
            h = define_header(f)
            if h and (symbol is None or h[0] == symbol):
                dsrc = to_src(f)
                return cls(repo=repo, path=path, symbol=h[0], source=source, define_src=dsrc,
                           params=tuple(h[1]), type_errors=cls.check(dsrc))
        raise ValueError(f"define {symbol!r} not found")

    @classmethod
    def from_file(cls, path: str | Path, symbol: str | None = None, repo: str = "local") -> "WorkspaceState":
        p = Path(path)
        return cls.from_source(p.read_text(), symbol=symbol, repo=repo, path=str(p))

    def rebind(self, new_define_src: str) -> "WorkspaceState":
        """Return the successor state with the target define replaced (rebind)."""
        forms = parse_all(self.source)
        out = []
        for f in forms:
            h = define_header(f)
            out.append(new_define_src if h and h[0] == self.symbol else to_src(f))
        new_source = "\n".join(out) + "\n"
        return replace(self, source=new_source, define_src=to_src(parse_all(new_define_src)[0]),
                       type_errors=self.check(new_define_src), step=self.step + 1)

    def with_history(self, ptr: int | None) -> "WorkspaceState":
        return replace(self, history_ptr=ptr)
