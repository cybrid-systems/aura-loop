"""Locate and import cybrid-systems/edsl-patch (the controller's patch contract + policy code).

Search order: $EDSL_PATCH_DIR, third_party/edsl-patch (git submodule), ../edsl-patch.
Nothing is copied: aura-loop imports edsl-patch's own modules from its scripts/ directory.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


class EdslPatchMissing(RuntimeError):
    pass


def find_edsl_patch() -> Path:
    cands = []
    if os.environ.get("EDSL_PATCH_DIR"):
        cands.append(Path(os.environ["EDSL_PATCH_DIR"]))
    cands += [ROOT / "third_party" / "edsl-patch", ROOT.parent / "edsl-patch"]
    for c in cands:
        if (c / "scripts" / "edsl_patch.py").is_file():
            return c.resolve()
    raise EdslPatchMissing(
        "edsl-patch not found. Run `git submodule update --init` (third_party/edsl-patch) "
        "or set EDSL_PATCH_DIR=/path/to/edsl-patch")


_CACHE: SimpleNamespace | None = None


def load() -> SimpleNamespace:
    """Import edsl-patch modules: edsl_patch, parse_aura, export_sft, apply, pipeline.baseline/intent."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = find_edsl_patch()
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    ns = SimpleNamespace(root=root)
    for mod in ("edsl_patch", "parse_aura", "export_sft", "apply"):
        setattr(ns, mod, importlib.import_module(mod))
    ns.baseline = importlib.import_module("pipeline.baseline")
    ns.intent = importlib.import_module("pipeline.intent")
    try:
        ns.sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        ns.sha = ""
    _CACHE = ns
    return ns
