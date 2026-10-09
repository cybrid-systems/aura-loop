"""Plant: applies a control u to state x and returns a verdict. FakePlant is real; SoftPlant is a stub."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Protocol

from .controller import Control
from .sexpr import ParseError, define_header, parse_all
from .state import WorkspaceState


@dataclass(frozen=True)
class PlantResult:
    verdict: str                 # KEEP | ROLLBACK | REJECT
    reason: str
    next_state: WorkspaceState   # == input state unless KEEP


class Plant(Protocol):
    def apply(self, x: WorkspaceState, u: Control) -> PlantResult: ...


def _admit(x: WorkspaceState, u: Control) -> str | None:
    """MutationBoundary-like admission: parseable, exactly one define, same name & params."""
    try:
        forms = parse_all(u.new_src)
    except ParseError as e:
        return f"syntax: {e}"
    if len(forms) != 1 or define_header(forms[0]) is None:
        return "boundary: control must be exactly one (define (name ...) ...)"
    name, params = define_header(forms[0])
    if name != x.symbol:
        return f"boundary: rebinds {name!r}, expected {x.symbol!r}"
    if tuple(params) != x.params:
        return f"boundary: signature changed {x.params} -> {tuple(params)}"
    return None


class FakePlant:
    """In-memory rebind with a toy gate.

    REJECT   = refused at the boundary (syntax / wrong symbol / signature change); state untouched.
    ROLLBACK = applied, but the gate (toy arity checker) still reports errors; rolled back.
    KEEP     = applied and gate clean; state advances.
    """

    def apply(self, x: WorkspaceState, u: Control) -> PlantResult:
        why = _admit(x, u)
        if why:
            return PlantResult("REJECT", why, x)
        x2 = x.rebind(u.new_src)
        if x2.type_errors:
            return PlantResult("ROLLBACK", "; ".join(x2.type_errors), x)
        return PlantResult("KEEP", "gate clean", x2)


class SoftPlant:
    """STUB: run the candidate file through a real Aura binary if AURA_BIN is set.

    It only checks that `$AURA_BIN <file>` exits 0. The intended real version drives an Aura
    workspace via mutate:rebind / MutationBoundary and reads the type checker + tests.
    """

    def __init__(self, aura_bin: str | None = None, timeout: float = 30.0):
        self.aura_bin = aura_bin or os.environ.get("AURA_BIN")
        self.timeout = timeout
        if not self.aura_bin:
            raise RuntimeError("SoftPlant needs AURA_BIN (path to an aura binary)")

    def apply(self, x: WorkspaceState, u: Control) -> PlantResult:
        why = _admit(x, u)
        if why:
            return PlantResult("REJECT", why, x)
        x2 = x.rebind(u.new_src)
        with tempfile.NamedTemporaryFile("w", suffix=".scm", delete=False) as f:
            f.write(x2.source)
            path = f.name
        try:
            p = subprocess.run([self.aura_bin, path], capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return PlantResult("ROLLBACK", "aura: timeout", x)
        finally:
            os.unlink(path)
        if p.returncode != 0:
            return PlantResult("ROLLBACK", f"aura exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}", x)
        return PlantResult("KEEP", "aura exit 0", x2)
