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


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()
        ingest(self.mem, SAMPLE)
        self.pol = SinkPolicy("knn", default_threshold=10.0)

    def tearDown(self):
        self.mem.close()

    def test_threshold_follows_the_shape(self):
        d = self.pol.decide("vsum", F_SUM, 3200.0, self.mem)
        sp = [r["speedup"] for r in self.mem.sink_rows("WHERE kernel='vsum'")]
        self.assertEqual(d.action, "sink")
        self.assertAlmostEqual(d.threshold, round(0.7 * min(sp), 2))
        self.assertTrue(d.source.startswith("knn:bucket:v1|add|len4^6"))
        self.assertLess(d.threshold, min(sp))

    def test_unprofitable_shape_gets_a_skip_floor(self):
        d = self.pol.decide("vsum-short", F_SUM_SHORT, 160.0, self.mem)
        self.assertEqual(d.action, "sink")
        self.assertTrue(d.source.endswith(":floor"))
        self.assertGreater(d.floor_us, 160.0 * 1.4)

    def test_uncompilable_kernel_is_skipped(self):
        f = [1, 1, 5000, ["f64vector-ref:1", "+:2", "loop:2", "f64vector-length:1", "<:1", "if:1", "lambda:1",
                          "letrec:1"]]
        d = self.pol.decide("vbad-mix", f, 4000.0, self.mem)
        self.assertEqual((d.action, d.source), ("skip", "knn:kernel:uncompilable"))

    def test_unseen_shape_uses_the_nearest_bucket(self):
        f = [1, 1, 2000, ["sqrt:1", "f64vector-ref:2", "*:1", "+:2", "loop:2", "<:1", "if:1"]]   # length band 5
        d = self.pol.decide("vnorm2-mid", f, 2000.0, self.mem)
        self.assertTrue(d.source.startswith("knn:near:"), d.source)
        self.assertEqual(d.action, "sink")

    def test_empty_memory_uses_the_prior(self):
        d = SinkPolicy("knn", default_threshold=10.0).decide("x", F_SUM, 100.0, Memory())
        self.assertEqual((d.action, d.threshold, d.source), ("sink", 10.0, "default"))

    def test_bandit_goes_greedy_on_the_best_arm(self):
        pol = SinkPolicy("bandit", epsilon=0.0, rng=random.Random(3))
        d = pol.decide("vsum", F_SUM, 3200.0, self.mem)
        self.assertTrue(d.source.startswith("bandit:greedy:"))
        # the best mean reward in that bucket is a KEEP arm
        best = max(self.mem.sink_rows("WHERE shape=?", (shape_key(parse_features(F_SUM)),)), key=lambda r: r["reward"])
        self.assertEqual(d.action, "sink")
        self.assertEqual(d.threshold, best["threshold"])

    def test_proposal_json_and_replay(self):
        prop = self.pol.propose(latest_kernels(self.mem), self.mem, proposal_id="p")
        self.assertEqual(prop["default_threshold"], 10.0)
        self.assertEqual(set(prop["kernels"]), {"vsum", "vdot", "vnorm2", "vscaled", "vcount-above", "vmaxabs",
                                                 "vsum-short", "vdot-short", "vbad-mix"})
        self.assertEqual(prop["kernels"]["vbad-mix"]["action"], "skip")
        rp = SinkPolicy.replay(self.mem)
        self.assertEqual(len(rp), 9)
        self.assertEqual(set(rp[0]), {"kernel", "state", "action", "reward", "decision", "next_state"})

if __name__ == "__main__":
    unittest.main()
