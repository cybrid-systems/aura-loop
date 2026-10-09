import unittest
from pathlib import Path

from aura_loop.controller import Control, RetrievalController
from aura_loop.loop import run, step
from aura_loop.memory import Memory
from aura_loop.plant import FakePlant
from aura_loop.seeds import seed
from aura_loop.state import WorkspaceState

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "tiny.scm"


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()
        seed(self.mem)
        self.x0 = WorkspaceState.from_file(FIXTURE, repo="fixtures")

    def tearDown(self):
        self.mem.close()

    def test_fixture_has_arity_error(self):
        self.assertEqual(self.x0.symbol, "clamp")
        self.assertTrue(self.x0.type_errors)

    def test_demo_converges_with_one_keep_writeback(self):
        before = len(self.mem.all())
        x, rs = run(self.x0, RetrievalController(self.mem), FakePlant(), self.mem, steps=3, log=None)
        self.assertEqual([r.verdict for r in rs], ["REJECT", "KEEP", "NOOP"])
        self.assertEqual(x.type_errors, ())
        self.assertIn("(min x hi)", x.define_src)
        new = self.mem.all()[before:]
        keeps = [p for p in new if p.verdict == "KEEP"]
        self.assertEqual(len(keeps), 1)
        self.assertEqual(keeps[0].after_src, x.define_src)
        self.assertEqual(keeps[0].before_src, self.x0.define_src)
        self.assertEqual(x.history_ptr, keeps[0].id)

    def test_failure_writeback_changes_next_decision(self):
        ctrl = RetrievalController(self.mem)
        r1 = step(self.x0, ctrl, FakePlant(), self.mem)
        self.assertEqual(r1.verdict, "REJECT")
        r2 = step(r1.x_next, ctrl, FakePlant(), self.mem)
        self.assertNotEqual(r1.u.new_src, r2.u.new_src)

    def test_plant_rollback_on_gate_error(self):
        u = Control("clamp", "(define (clamp x lo hi) (max (min x)))", "bad", "test")
        r = FakePlant().apply(self.x0, u)
        self.assertEqual(r.verdict, "ROLLBACK")
        self.assertIs(r.next_state, self.x0)

    def test_plant_rejects_unbalanced(self):
        u = Control("clamp", "(define (clamp x lo hi) (max lo (min x hi))", "bad", "test")
        self.assertEqual(FakePlant().apply(self.x0, u).verdict, "REJECT")

    def test_empty_memory_is_identity(self):
        empty = Memory()
        x, rs = run(self.x0, RetrievalController(empty), FakePlant(), empty, steps=1, log=None)
        empty.close()
        self.assertEqual(rs[0].verdict, "NOOP")
        self.assertEqual(x, self.x0)


if __name__ == "__main__":
    unittest.main()
