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


class FakeLoopTest(unittest.TestCase):
    KERNELS = {  # name: (features, soft_us, true speedup, compilable) — shaped like the real run
        "vsum": (F_SUM, 3300.0, 7.0, True), "vdot": (F_DOT, 4900.0, 9.5, True), "vnorm2": (F_NORM, 5800.0, 12.0, True),
        "vmaxabs": (F_MAXABS, 6000.0, 13.0, True), "vsum-short": (F_SUM_SHORT, 170.0, 0.3, True),
        "vbad": ([1, 1, 5000, ["f64vector-ref:1", "+:2", "if:1"]], 4000.0, 0.0, False),
    }

    def test_learner_beats_fixed_threshold_after_round_one(self):
        with tempfile.TemporaryDirectory() as d:
            rep = compare(FakeSinkRunner(self.KERNELS, seed=5), rounds=4, baseline_threshold=10.0, workdir=d)
        fixed, knn = rep["per_round"]["fixed"], rep["per_round"]["knn"]
        self.assertEqual(fixed[0]["keep"], knn[0]["keep"])                 # same prior in round 1
        self.assertEqual(sum(m["rollback"] for m in knn[1:]), 0)
        self.assertGreater(sum(m["rollback"] for m in fixed[1:]), 0)
        self.assertGreater(rep["totals_after_round1"]["knn"]["kept_speedup_sum"],
                           rep["totals_after_round1"]["fixed"]["kept_speedup_sum"])
        self.assertGreater(rep["totals"]["knn"]["reward"], rep["totals"]["fixed"]["reward"])
        self.assertEqual(knn[-1]["kept"], ["vdot", "vmaxabs", "vnorm2", "vsum"])
        self.assertGreater(rep["replay_tuples"], 0)

    def test_regression_is_still_rolled_back(self):
        # learned threshold for vsum ~ 0.7 x 7; a kernel that suddenly delivers 2x is refused
        k = dict(self.KERNELS)
        mem = Memory()
        with tempfile.TemporaryDirectory() as d:
            from aura_loop.sink_loop import run_arm
            pol = SinkPolicy("knn", default_threshold=10.0)
            run_arm("knn", pol, FakeSinkRunner(k, seed=2), 2, mem, d)
            k["vsum"] = (F_SUM, 3300.0, 2.0, True)
            m = run_arm("knn2", pol, FakeSinkRunner(k, seed=3), 1, mem, d)[0]
        self.assertIn("vsum", m["rolled_back"])
        mem.close()

    def test_fixed_policy_has_no_per_kernel_entries(self):
        self.assertEqual(FixedPolicy(10.0).propose([], None)["kernels"], {})


@unittest.skipUnless(os.environ.get("AURA_LOOP_SINK_E2E") == "1" and os.environ.get("AURA_BIN")
                     and os.environ.get("SINK_REPO"), "real aura-kernel-sink loop: set AURA_LOOP_SINK_E2E=1, AURA_BIN, SINK_REPO")
class RealSinkLoopTest(unittest.TestCase):
    def test_two_rounds_on_real_aura(self):
        from aura_loop.sink_loop import SinkRunner
        with tempfile.TemporaryDirectory() as d:
            rep = compare(SinkRunner(os.environ["SINK_REPO"]), rounds=2, baseline_threshold=10.0, workdir=d)
        knn2, fixed2 = rep["per_round"]["knn"][1], rep["per_round"]["fixed"][1]
        self.assertLess(knn2["rollback"], fixed2["rollback"])
        self.assertGreaterEqual(knn2["kept_speedup_sum"], fixed2["kept_speedup_sum"])

if __name__ == "__main__":
    unittest.main()
