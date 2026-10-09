"""Plant: applies a control u to state x and returns a verdict.

FakePlant - in-process rebind + toy arity gate (unit tests, no Aura binary).
SoftPlant - the real plant: edsl-patch `apply_patch` drives an Aura binary (`mutate:rebind`, Aura's type
            checker / MutationBoundary, `eval-current`, snapshot/restore), then the fixture's tests are run
            on Aura against the post-mutation source. Needs AURA_BIN (+ AURA_LIB for the stdlib).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .controller import Control
from .sexpr import ParseError, define_header, define_to_lambda, parse_all
from .state import WorkspaceState


@dataclass(frozen=True)
class PlantResult:
    verdict: str                 # KEEP | ROLLBACK | REJECT
    reason: str
    next_state: WorkspaceState   # == input state unless KEEP


class Plant(Protocol):
    def apply(self, x: WorkspaceState, u: Control) -> PlantResult: ...


def _admit(x: WorkspaceState, u: Control) -> str | None:
    """MutationBoundary-like admission: parseable, exactly one define, same name & params, legal patch."""
    try:
        forms = parse_all(u.new_src)
    except ParseError as e:
        return f"syntax: {e}"
    if len(forms) != 1 or define_header(forms[0]) is None:
        return "boundary: control must be exactly one define"
    name, params = define_header(forms[0])
    if name != x.symbol:
        return f"boundary: rebinds {name!r}, expected {x.symbol!r}"
    if tuple(params) != x.params:
        return f"boundary: signature changed {x.params} -> {tuple(params)}"
    if u.patch is not None:
        from . import edsl_bridge
        e = edsl_bridge.load()
        try:
            e.edsl_patch.validate_patch(u.patch, observe={"hot": [x.symbol], "frozen": []})
        except e.edsl_patch.PatchError as exc:
            return f"edsl-patch: {exc}"
    return None


class FakePlant:
    """In-memory rebind with a toy gate.

    REJECT   = refused at the boundary (syntax / wrong symbol / signature change / illegal patch); state untouched.
    ROLLBACK = applied, but the gate (toy arity checker) still reports errors; rolled back.
    KEEP     = applied and gate clean; state advances.
    """

    name = "fake"

    def apply(self, x: WorkspaceState, u: Control) -> PlantResult:
        why = _admit(x, u)
        if why:
            return PlantResult("REJECT", why, x)
        x2 = x.rebind(u.new_src)
        if x2.type_errors:
            return PlantResult("ROLLBACK", "; ".join(x2.type_errors), x)
        return PlantResult("KEEP", "gate clean (toy arity)", x2)


TEST_MARK = "AURA_LOOP_TESTS "

TEST_PRELUDE = """(define *al-fails* (list))
(define (al-check name got want)
  (if (equal? got want) #t (set! *al-fails* (cons name *al-fails*))))
"""


def tests_for(path: str) -> Path | None:
    """fixtures/foo.scm -> fixtures/foo.tests.scm (lines of `(al-check "name" expr want)`)."""
    p = Path(path)
    t = p.with_name(p.name.rsplit(".", 1)[0] + ".tests.scm")
    return t if t.is_file() else None


class SoftPlant:
    """Real plant on an Aura binary.

    1. boundary admission (same as FakePlant)                          -> REJECT
    2. edsl-patch apply_patch(x.source, u.patch): query* + mutate:rebind on Aura; Aura's type checker /
       MutationBoundary rejects bad rebinds; eval-current failure restores the snapshot -> ROLLBACK
    3. x' = Aura's post-mutation source (`current-source :workspace`)
    4. fixture tests run on Aura against x'                            -> ROLLBACK if any fail
    5. otherwise KEEP, state advances to x'
    """

    name = "soft"

    def __init__(self, aura_bin: str | None = None, aura_lib: str | None = None, tests: str | Path | None = None,
                 timeout: int = 60):
        self.aura_bin = aura_bin or os.environ.get("AURA_BIN")
        if not self.aura_bin or not Path(self.aura_bin).is_file():
            raise RuntimeError("SoftPlant needs AURA_BIN (path to an aura binary); use --plant fake otherwise")
        self.aura_bin = str(Path(self.aura_bin).resolve())
        self.aura_lib = aura_lib or os.environ.get("AURA_LIB") or str(Path(self.aura_bin).parent.parent / "lib")
        os.environ["AURA_BIN"], os.environ["AURA_LIB"] = self.aura_bin, self.aura_lib   # edsl-patch apply reads these
        self.tests = Path(tests) if tests else None
        self.timeout = timeout
        from . import edsl_bridge
        self.e = edsl_bridge.load()

    def env(self) -> dict:
        env = os.environ.copy()
        env.update({"AURA_PATH": self.aura_lib, "AURA_SANDBOX": "off", "AURA_PIPELINE_STRICT": "0"})
        return env

    def info(self) -> dict:
        h = hashlib.sha256(Path(self.aura_bin).read_bytes()).hexdigest()
        src = Path(self.aura_bin).parent.parent
        try:
            head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            head = ""
        return {"aura_bin": self.aura_bin, "aura_bin_sha256": h, "aura_src_head": head, "aura_lib": self.aura_lib}

    # -- tests on Aura -----------------------------------------------------------------------------
    def run_tests(self, source: str) -> list[str]:
        if not self.tests:
            return []
        with tempfile.TemporaryDirectory(prefix="aura-loop-") as tmp:
            s = Path(tmp) / "source.aura"
            d = Path(tmp) / "tests.aura"
            s.write_text(source.strip() + "\n")
            d.write_text(f'(load {json.dumps(str(s))})\n' + TEST_PRELUDE + self.tests.read_text()
                         + f'\n(display "{TEST_MARK}")\n(display (json-encode (hash "fails" *al-fails*)))\n(newline)\n')
            try:
                p = subprocess.run([self.aura_bin, str(d)], cwd=tmp, env=self.env(), capture_output=True,
                                   text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return ["tests: aura timeout"]
        for line in (p.stdout or "").splitlines():
            if line.startswith(TEST_MARK):
                fails = json.loads(line[len(TEST_MARK):]).get("fails") or []
                return [f"test failed on Aura: {n}" for n in reversed(fails)]
        return [f"tests: no result (aura exit {p.returncode}): {(p.stderr or p.stdout)[-200:]}"]

    def probe(self, x: WorkspaceState) -> list[str]:
        """Observation hook: failing Aura tests on the current state."""
        return self.run_tests(x.source)

    # -- control ------------------------------------------------------------------------------------
    def apply(self, x: WorkspaceState, u: Control) -> PlantResult:
        why = _admit(x, u)
        if why:
            return PlantResult("REJECT", why, x)
        patch = u.patch
        if patch is None:   # plain Control (e.g. RetrievalController): wrap as an edsl-patch rebind
            _, body = define_to_lambda(u.new_src)
            patch = self.e.parse_aura.patch_for(x.symbol, body, u.intent[:120] or "rebind")
        ap = self.e.apply
        try:
            res = ap.apply_patch(x.source, patch, timeout=self.timeout)
        except self.e.edsl_patch.PatchError as exc:
            return PlantResult("ROLLBACK", f"aura apply: {str(exc)[-300:]}", x)
        if not ap.apply_succeeded(res):
            det = (res.get("synthesis") or {}).get("detail")
            return PlantResult("ROLLBACK", f"aura mutate:rebind rejected: {json.dumps(det)[:300]}", x)
        post = res.get("source") or ""
        try:
            x2 = WorkspaceState.from_source(post, symbol=x.symbol, repo=x.repo, path=x.path)
            x2 = replace(x2, step=x.step + 1, history_ptr=x.history_ptr)
        except (ParseError, ValueError) as exc:
            return PlantResult("ROLLBACK", f"aura post-source unreadable: {exc}", x)
        fails = self.run_tests(post)
        if fails:
            return PlantResult("ROLLBACK", "mutate:rebind ok, " + "; ".join(fails), x)
        return PlantResult("KEEP", "aura mutate:rebind #t" + (", tests pass on Aura" if self.tests else ""), x2)
