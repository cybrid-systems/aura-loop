"""aura-kernel-sink trajectories -> memory -> per-shape policy -> proposals (no Aura needed).

The real loop (Aura + cc via aura-kernel-sink) runs when AURA_LOOP_SINK_E2E=1,
AURA_BIN and SINK_REPO are set.
"""
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path

from aura_loop.memory import Memory
from aura_loop.sink_ingest import ingest, parse_features, reward, shape_key, to_transition
from aura_loop.sink_loop import FakeSinkRunner, FixedPolicy, compare, latest_kernels
from aura_loop.sink_policy import SinkPolicy

SAMPLE = Path(__file__).resolve().parent.parent / "fixtures" / "sink" / "trajectory-sample.jsonl"
F_SUM = [2, 1, 5000, ["f64vector-ref:1", "+:2", "loop:2", "<:1", "if:1", "lambda:1", "letrec:1"]]
F_SUM_SHORT = [2, 1, 128, ["f64vector-ref:1", "+:2", "loop:2", "<:1", "if:1", "lambda:1", "letrec:1"]]
F_DOT = [3, 2, 5000, ["f64vector-ref:2", "*:1", "+:2", "loop:2", "<:1", "if:1", "lambda:1", "letrec:1"]]
F_NORM = [1, 1, 5000, ["sqrt:1", "f64vector-ref:2", "*:1", "+:2", "loop:2", "<:1", "if:1", "lambda:1", "letrec:1"]]
F_MAXABS = [1, 1, 5000, ["f64vector-ref:1", "abs:1", "max:1", "+:1", "loop:2", "<:1", "if:1", "lambda:1", "letrec:1"]]


def by_kernel(rows):
    out = {}
    for r in rows:
        out.setdefault(r["kernel"], []).append(r)
    return out


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()

    def tearDown(self):
        self.mem.close()

    def test_sample_rows_become_state_action_reward(self):
        ids = ingest(self.mem, SAMPLE)
        self.assertEqual(len(ids), 18)
        rows = by_kernel(self.mem.sink_rows("ORDER BY id"))
        vsum = rows["vsum"][0]
        st = json.loads(vsum["state"])
        self.assertEqual((st["arity"], st["nvec"], st["vec_len"]), (2, 1, 5000))
        self.assertEqual(st["ops"]["f64vector-ref"], 1)
        self.assertEqual(vsum["decision"], "KEEP")
        self.assertAlmostEqual(vsum["reward"], math.log(vsum["speedup"]))
        short = rows["vsum-short"][0]
        self.assertEqual(short["decision"], "ROLLBACK")
        self.assertEqual(short["reward"], -1.0)
        self.assertNotEqual(short["shape"], vsum["shape"])            # other length band
        self.assertEqual(rows["vsum-short"][1]["action"], "skip")      # round 1 of the sample: proposal said skip
        self.assertEqual(rows["vsum-short"][1]["round"], 1)
        self.assertEqual(rows["vbad-mix"][0]["stage"], "compile")
        self.assertEqual(rows["vbad-mix"][0]["reward"], -0.1)

    def test_reingest_is_a_noop(self):
        ingest(self.mem, SAMPLE)
        self.assertEqual(ingest(self.mem, SAMPLE), [])
        self.assertEqual(len(self.mem.sink_rows()), 18)

    def test_shape_buckets(self):
        k = lambda f: shape_key(parse_features(f))
        self.assertEqual(k(F_SUM), "v1|add|len4^6")
        self.assertEqual(k(F_SUM_SHORT), "v1|add|len4^3")
        self.assertEqual(k(F_DOT), "v2|*|len4^6")
        self.assertEqual(k(F_NORM), "v1|*+sqrt|len4^6")
        self.assertEqual(k(F_MAXABS), "v1|abs+max|len4^6")

    def test_reward_model(self):
        self.assertAlmostEqual(reward("KEEP", 8.0), math.log(8.0))
        self.assertEqual(reward("ROLLBACK", 0.3), -1.0)
        self.assertEqual(reward("ABORT", 0.0), -2.0)
        self.assertEqual(reward("SKIP", 0.0, "profile"), 0.0)
        self.assertIsNone(to_transition("   "))

if __name__ == "__main__":
    unittest.main()
