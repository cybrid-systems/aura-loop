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
    """Return (name, params) for `(define (name . params) body...)` or `(define name (lambda (params) body...))`."""
    if not (isinstance(form, list) and len(form) >= 3 and form[0] == "define"):
        return None
    if isinstance(form[1], list) and form[1]:
        return form[1][0], list(form[1][1:])
    lam = form[2]
    if (isinstance(form[1], str) and len(form) == 3 and isinstance(lam, list) and len(lam) >= 3
            and lam[0] == "lambda" and isinstance(lam[1], list)):
        return form[1], list(lam[1])
    return None


def canonical_define(form: Any) -> list:
    """Normalize either define style to sugar form `(define (name params...) body...)`."""
    h = define_header(form)
    if h is None:
        raise ParseError("not a define")
    if isinstance(form[1], list):
        return form
    lam = form[2]
    return ["define", [form[1], *lam[1]], *lam[2:]]


def define_to_lambda(define_src: str) -> tuple[str, str]:
    """`(define (f a b) body)` -> ("f", "(lambda (a b) body)")  — the edsl-patch rebind.body shape."""
    form = canonical_define(parse_all(define_src)[0])
    name, params, body = form[1][0], form[1][1:], form[2:]
    return name, to_src(["lambda", list(params), *body])


def lambda_to_define(name: str, lambda_src: str) -> str:
    """("f", "(lambda (a b) body)") -> "(define (f a b) body)"."""
    forms = parse_all(lambda_src)
    if len(forms) != 1 or not isinstance(forms[0], list) or len(forms[0]) < 3 or forms[0][0] != "lambda" \
            or not isinstance(forms[0][1], list):
        raise ParseError("rebind.body must be one (lambda (params) body...) form")
    lam = forms[0]
    return to_src(["define", [name, *lam[1]], *lam[2:]])


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
