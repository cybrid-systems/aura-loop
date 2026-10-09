"""E2E closed loop with the edsl-patch controller (no network, no Aura binary needed)."""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from aura_loop import edsl_bridge
from aura_loop.__main__ import main
from aura_loop.edsl_controller import EdslPatchController
from aura_loop.loop import run, step
from aura_loop.memory import Memory
from aura_loop.observe import observe
from aura_loop.plant import FakePlant, SoftPlant
from aura_loop.seeds import seed
from aura_loop.state import WorkspaceState

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "tiny.scm"
VERDICTS = {"KEEP", "ROLLBACK", "REJECT", "NOOP"}


class LicenseTest(unittest.TestCase):
    def test_apache_license(self):
        text = (ROOT / "LICENSE").read_text()
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", text)
        self.assertNotIn("MIT License", text)
        self.assertIn("Apache-2.0", (ROOT / "pyproject.toml").read_text())


class EdslBridgeTest(unittest.TestCase):
    def test_loads_edsl_patch_modules(self):
        e = edsl_bridge.load()
        self.assertTrue((e.root / "scripts" / "edsl_patch.py").is_file())
        for fn in (e.edsl_patch.validate_patch, e.parse_aura.patch_for, e.export_sft.to_sft,
                   e.apply.apply_patch, e.baseline._ask, e.baseline._parse, e.intent.load_minimax):
            self.assertTrue(callable(fn))


class E2ETest(unittest.TestCase):
    def setUp(self):
        self.mem = Memory()
        seed(self.mem)
        self.x0 = WorkspaceState.from_file(FIXTURE, repo="fixtures")

    def tearDown(self):
        self.mem.close()

    def test_retrieval_policy_emits_legal_edsl_patch(self):
        ctrl = EdslPatchController(self.mem, policy="retrieval")
        u = ctrl.propose(observe(self.x0, self.mem))
        e = edsl_bridge.load()
        e.edsl_patch.validate_patch(u.patch, observe={"hot": ["clamp"], "frozen": []})
        self.assertEqual(u.patch[-1]["op"], "rebind")
        self.assertTrue(u.patch[-1]["body"].startswith("(lambda"))
        self.assertTrue(u.source.startswith("edsl-patch:retrieval:memory:"))

    def test_one_transition_x_y_u_xprime(self):
        ctrl = EdslPatchController(self.mem, policy="retrieval")
        r = step(self.x0, ctrl, FakePlant(), self.mem)
        d = r.as_dict()
        self.assertEqual(set(d) >= {"state_before", "observation", "control", "state_after", "verdict"}, True)
        self.assertIn(r.verdict, VERDICTS)
        self.assertIsNotNone(d["control"]["patch"])
        self.assertIsNotNone(r.memory_id)
        json.dumps(d)

    def test_closed_loop_converges_via_edsl_patch_controller(self):
        ctrl = EdslPatchController(self.mem, policy="retrieval")
        x, rs = run(self.x0, ctrl, FakePlant(), self.mem, steps=4, log=None, stop_on_noop=True)
        self.assertEqual([r.verdict for r in rs], ["REJECT", "KEEP", "NOOP"])
        self.assertIn("(min x hi)", x.define_src)
        self.assertEqual(rs[1].x_next, rs[2].x)          # step 2 observes x'
        self.assertNotEqual(rs[0].u.new_src, rs[1].u.new_src)   # REJECT writeback changed the decision

    def test_minimax_policy_uses_edsl_patch_prompt_and_parser(self):
        seen = {}

        def fake_ask(cfg, msgs):
            seen["msgs"] = msgs
            return '```json\n[{"kind":"query","op":"find","name":"clamp"},{"kind":"synthesis","op":"rebind",' \
                   '"name":"clamp","body":"(lambda (x lo hi) (max lo (min x hi)))","summary":"pass hi to min"}]\n```'

        ctrl = EdslPatchController(self.mem, policy="minimax", ask=fake_ask)
        r = step(self.x0, ctrl, FakePlant(), self.mem)
        e = edsl_bridge.load()
        self.assertEqual(seen["msgs"][0]["content"], e.edsl_patch.SYSTEM_CONTRACT.strip())
        self.assertIn("intent: Fix clamp", seen["msgs"][1]["content"])
        self.assertIn("observe: ", seen["msgs"][1]["content"])
        self.assertTrue(r.u.source.startswith("edsl-patch:minimax:"))
        self.assertEqual(r.verdict, "KEEP")

    def test_illegal_llm_output_falls_back_and_plant_rejects_illegal_patch(self):
        ctrl = EdslPatchController(self.mem, policy="minimax", ask=lambda c, m: '[{"kind":"completion"}]')
        u = ctrl.propose(observe(self.x0, self.mem))
        self.assertIn("minimax-fallback-retrieval", u.source)
        from aura_loop.controller import Control
        bad = Control("clamp", "(define (clamp x lo hi) (max lo (min x hi)))", "x", "test",
                      patch=[{"kind": "synthesis", "op": "rebind", "name": "clamp",
                              "body": "(lambda (x lo hi) x)", "summary": "s"}])
        res = FakePlant().apply(self.x0, bad)
        self.assertEqual(res.verdict, "REJECT")
        self.assertIn("edsl-patch", res.reason)

    def test_llm_repair_turn(self):
        answers = iter(['[{"kind":"rebind"}]',
                        '[{"kind":"query","op":"find","name":"clamp"},{"kind":"synthesis","op":"rebind",'
                        '"name":"clamp","body":"(lambda (x lo hi) (max lo (min x hi)))","summary":"fix"}]'])
        calls = []

        def ask(cfg, msgs):
            calls.append(msgs)
            return next(answers)

        u = EdslPatchController(self.mem, policy="minimax", ask=ask).propose(observe(self.x0, self.mem))
        self.assertEqual(len(calls), 2)
        self.assertIn("Illegal patch", calls[1][-1]["content"])
        self.assertTrue(u.source.endswith(":repaired1"))

    def test_cli_e2e_json(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--e2e", "--plant", "fake", "--policy", "retrieval"])
        out = json.loads(buf.getvalue())
        self.assertEqual(rc, 0)
        self.assertTrue(out["converged"])
        t = out["transitions"][0]
        for k in ("state_before", "observation", "control", "state_after", "verdict"):
            self.assertIn(k, t)
        self.assertIn(t["verdict"], VERDICTS)
        self.assertEqual(out["controller"]["kind"], "edsl-patch")

    def test_cli_e2e_default_plant_requires_aura(self):
        with mock.patch.dict(os.environ, {"AURA_BIN": ""}):
            self.assertEqual(main(["--e2e"]), 2)


@unittest.skipUnless(os.environ.get("AURA_BIN") and Path(os.environ["AURA_BIN"]).is_file(),
                     "AURA_BIN not set: real-Aura SoftPlant test skipped")
class RealAuraTest(unittest.TestCase):
    def test_real_aura_closed_loop(self):
        mem = Memory()
        seed(mem)
        plant = SoftPlant(tests=ROOT / "fixtures" / "tiny.tests.scm")
        x0 = WorkspaceState.from_file(FIXTURE, repo="fixtures")
        self.assertTrue(plant.probe(x0))               # Aura tests fail on x0
        x, rs = run(x0, EdslPatchController(mem, policy="retrieval"), plant, mem, steps=4, log=None,
                    static=False, stop_on_noop=True)
        self.assertEqual([r.verdict for r in rs], ["REJECT", "KEEP", "NOOP"])
        self.assertEqual(plant.probe(x), [])

    def test_real_aura_rollbacks(self):
        from aura_loop.controller import Control
        plant = SoftPlant(tests=ROOT / "fixtures" / "tiny.tests.scm")
        x0 = WorkspaceState.from_file(FIXTURE, repo="fixtures")
        # Aura's type checker rejects the rebind (unbound function) -> ROLLBACK, state unchanged
        r = plant.apply(x0, Control("clamp", "(define (clamp x lo hi) (no-such-fn x))", "t", "test"))
        self.assertEqual(r.verdict, "ROLLBACK")
        self.assertIn("mutate:rebind rejected", r.reason)
        self.assertIs(r.next_state, x0)
        # rebind accepted by Aura, but the Aura tests fail -> ROLLBACK
        r = plant.apply(x0, Control("clamp", "(define (clamp x lo hi) (min x hi))", "t", "test"))
        self.assertEqual(r.verdict, "ROLLBACK")
        self.assertIn("test failed on Aura", r.reason)


if __name__ == "__main__":
    unittest.main()
