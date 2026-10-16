"""aura-rocks adapt trajectories -> memory -> (state, action, reward) -> learned thresholds."""
import json
import tempfile
import unittest
from pathlib import Path

from aura_loop.adapt_ingest import AdaptPolicy, ingest, to_row, transitions
from aura_loop.memory import Memory

SAMPLE = Path(__file__).resolve().parent.parent / "fixtures" / "adapt" / "trajectory-test.jsonl"


def line(knob, frm, to, decision, stage, speedup, reward, feats, label="t", ts=0):
    return json.dumps({"ts": ts, "repo": "aura-rocks", "knob": knob, "from": frm, "to": to, "decision": decision,
                       "stage": stage, "speedup": speedup, "reward": reward, "label": label, "features": feats})


class AdaptIngestTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()

    def tearDown(self):
        self.mem.close()

    def test_ingest_sample_is_idempotent(self):
        ids = ingest(self.mem, SAMPLE)
        self.assertEqual(len(ids), len(SAMPLE.read_text().splitlines()))
        self.assertEqual(ingest(self.mem, SAMPLE), [])
        self.assertTrue(all(r["repo"] == "aura-rocks" for r in self.mem.adapt_rows()))

    def test_pairs_swap_with_post_outcome(self):
        ingest(self.mem, SAMPLE)
        ts = transitions(self.mem.adapt_rows())
        swaps = [t for t in ts if t["outcome"] in ("KEEP", "ROLLBACK")]
        self.assertTrue(swaps)
        for t in swaps:
            self.assertIn("write_ratio", t["state"])
        rb = [t for t in ts if t["outcome"] == "ROLLBACK"]
        self.assertTrue(all(t["reward"] == -1.0 for t in rb if t["realized"]))
        self.assertTrue(any(t["outcome"] == "REJECT" for t in ts))

    def test_learns_write_threshold(self):
        rows, ts = [], 0
        # vector swaps pay off in write-only phases and lose in mixed ones
        for w, sp in [(1.0, 1.4), (0.98, 1.3), (0.95, 1.2), (0.7, 0.5), (0.65, 0.6), (0.6, 0.4)]:
            ts += 1
            rows.append(line("memtable_rep", "skiplist", "vector", "SWAP", "gate", 3.0, 0.0, {"write_ratio": w}, ts=ts))
            ts += 1
            if sp >= 1:
                rows.append(line("memtable_rep", "skiplist", "vector", "KEEP", "post", sp, __import__("math").log(sp), {"write_ratio": w}, ts=ts))
            else:
                rows.append(line("memtable_rep", "vector", "skiplist", "ROLLBACK", "post", sp, -1.0, {"write_ratio": w}, ts=ts))
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(rows) + "\n")
        ingest(self.mem, f.name)
        p = AdaptPolicy().propose(self.mem)
        th = p["proposal"]["mem_write_hi"]
        self.assertGreater(th, 0.7)
        self.assertLessEqual(th, 0.95)
        # untouched rules keep their defaults
        self.assertEqual(p["proposal"]["mem_scan_hi"], 0.15)

    def test_non_adapt_lines_ignored(self):
        self.assertIsNone(to_row(""))
        self.assertIsNone(to_row(json.dumps({"name": "rk-hash", "decision": "KEEP"})))


if __name__ == "__main__":
    unittest.main()
