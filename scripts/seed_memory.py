#!/usr/bin/env python3
"""Seed a SQLite memory DB with precedents for fixtures/tiny.scm.

    python scripts/seed_memory.py --db loop.db
    python -m aura_loop --demo --db loop.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_loop.memory import Memory  # noqa: E402
from aura_loop.seeds import seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="loop.db")
a = ap.parse_args()
m = Memory(a.db)
ids = seed(m)
print(f"seeded {len(ids)} precedents into {a.db}: ids={ids}")
