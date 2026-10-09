"""Tiny s-expression reader + toy arity checker (MVP stand-in for Aura's type checker)."""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r'\s*(;[^\n]*|"(?:\\.|[^"\\])*"|[()]|[^\s()";]+)')


class ParseError(ValueError):
    pass


def tokenize(src: str) -> list[str]:
    out, pos = [], 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            if src[pos:].strip() == "":
                break
            raise ParseError(f"bad token at {pos}")
        tok = m.group(1)
        pos = m.end()
        if tok and not tok.startswith(";"):
            out.append(tok)
    return out


def parse_all(src: str) -> list[Any]:
    toks = tokenize(src)
    forms, i = [], 0

    def read(i: int):
        if i >= len(toks):
            raise ParseError("unexpected EOF (unbalanced parens)")
        t = toks[i]
        if t == "(":
            lst, i = [], i + 1
            while True:
                if i >= len(toks):
                    raise ParseError("unexpected EOF (unbalanced parens)")
                if toks[i] == ")":
                    return lst, i + 1
                x, i = read(i)
                lst.append(x)
        if t == ")":
            raise ParseError("unexpected ')'")
        return t, i + 1

    while i < len(toks):
        f, i = read(i)
        forms.append(f)
    return forms


def to_src(x: Any) -> str:
    if isinstance(x, list):
        return "(" + " ".join(to_src(e) for e in x) + ")"
    return str(x)


def define_header(form: Any) -> tuple[str, list[str]] | None:
    """Return (name, params) for `(define (name . params) body...)`."""
    if isinstance(form, list) and len(form) >= 3 and form[0] == "define" and isinstance(form[1], list) and form[1]:
        return form[1][0], list(form[1][1:])
    return None


# name -> (min_args, max_args or None). Toy table; real gate is Aura's type checker.
ARITY: dict[str, tuple[int, int | None]] = {
    "min": (2, None),
    "max": (2, None),
    "car": (1, 1),
    "cdr": (1, 1),
    "cons": (2, 2),
    "not": (1, 1),
    "string-append": (0, None),
}


def arity_errors(form: Any) -> list[str]:
    errs: list[str] = []

    def walk(x: Any) -> None:
        if not isinstance(x, list) or not x:
            return
        head = x[0]
        if isinstance(head, str) and head in ARITY:
            lo, hi = ARITY[head]
            n = len(x) - 1
            if n < lo or (hi is not None and n > hi):
                want = f">={lo}" if hi is None else (f"{lo}" if lo == hi else f"{lo}..{hi}")
                errs.append(f"arity: {to_src(x)} — `{head}` expects {want} args, got {n}")
        for e in x:
            walk(e)

    walk(form)
    return errs
