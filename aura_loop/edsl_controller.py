"""Controller u = pi(y, memory) built on cybrid-systems/edsl-patch.

The control signal is an edsl-patch *patch array* (schema/patch.md):
    [{"kind":"query","op":"find","name":f}, ..., {"kind":"synthesis","op":"rebind","name":f,"body":"(lambda ...)","summary":...}]
validated by edsl-patch's own `validate_patch` (hot/frozen observe rules included). The plant applies it
with edsl-patch's `apply_patch` (-> Aura `mutate:rebind`).

Policies (`policy=`):
  minimax   - edsl-patch's zero-shot policy: SFT prompt from `export_sft.to_sft` (SYSTEM_CONTRACT + source +
              intent + observe), sent with `pipeline.baseline._ask`, parsed with `baseline._parse`.
              Key/config via `pipeline.intent.load_minimax` (~/.config/aura-build/minimax.env).
  retrieval - no network: the best memory precedent (KEEP, never failed for this symbol) is turned into an
              edsl-patch patch with `parse_aura.patch_for` and validated. This is the CI/default-offline policy.
  auto      - minimax when a MiniMax config is found (and AURA_LOOP_NO_LLM is unset), else retrieval.
Memory is in the loop either way: precedents and known-bad bodies are passed in `observe` for minimax,
and they are the candidate set for retrieval.
"""
from __future__ import annotations

import json
import os

from . import edsl_bridge
from .controller import Control
from .memory import Memory
from .observe import Observation
from .sexpr import ParseError, define_to_lambda, lambda_to_define

POLICIES = ("auto", "minimax", "retrieval")


class EdslPatchController:
    def __init__(self, memory: Memory, policy: str = "auto", k: int = 5, minimax_env: str | None = None,
                 ask=None, llm_retries: int = 1):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.memory, self.k, self.llm_retries = memory, k, llm_retries
        self.e = edsl_bridge.load()
        self._ask = ask            # injectable for tests: (cfg, messages) -> str
        self.cfg = None
        if policy in ("auto", "minimax") and not os.environ.get("AURA_LOOP_NO_LLM"):
            from pathlib import Path
            self.cfg = self.e.intent.load_minimax(Path(minimax_env) if minimax_env else None)
            if self.cfg is None and ask is not None:
                self.cfg = {"model": "injected", "base": "", "_key": ""}
        if policy == "minimax" and self.cfg is None:
            raise RuntimeError("policy=minimax but no MiniMax config (~/.config/aura-build/minimax.env)")
        self.policy = "minimax" if self.cfg is not None else "retrieval"
        self.last_prompt: list[dict] | None = None

    # -- observation -> edsl-patch sample input -------------------------------------------------
    def sample_input(self, y: Observation) -> dict:
        bad = sorted(self.memory.failed_afters(y.symbol))
        keeps = [p for p in self.memory.retrieve(y.symbol, y.define_src, k=self.k, verdicts=("KEEP",))
                 if p.after_src not in bad and p.after_src != y.define_src]
        observe = {"hot": [y.symbol], "frozen": [], "errors": list(y.errors),
                   "precedents": [self._lam(p.after_src) for p in keeps[:2]],
                   "avoid": [self._lam(b) for b in bad][:4]}
        return {"source": y.source or y.define_src, "intent": y.intent, "observe": observe}

    @staticmethod
    def _lam(define_src: str) -> str:
        try:
            return define_to_lambda(define_src)[1]
        except (ParseError, IndexError):
            return define_src

    def prompt(self, y: Observation) -> list[dict]:
        """Exactly edsl-patch's SFT chat prompt (system + user), via export_sft.to_sft."""
        inp = self.sample_input(y)
        placeholder = self.e.parse_aura.patch_for(y.symbol, self._lam(y.define_src), "placeholder")
        return self.e.export_sft.to_sft({"input": inp, "target": placeholder})["messages"][:2]

    # -- policy ----------------------------------------------------------------------------------
    def propose(self, y: Observation) -> Control:
        if y.healthy:
            return Control(y.symbol, y.define_src, "healthy: no-op", "identity")
        inp = self.sample_input(y)
        if self.policy == "minimax":
            return self._propose_llm(y, inp)
        return self._propose_retrieval(y, inp)

    def _propose_retrieval(self, y: Observation, inp: dict) -> Control:
        bad = self.memory.failed_afters(y.symbol)
        for p in self.memory.retrieve(y.symbol, y.define_src, k=self.k, verdicts=("KEEP",)):
            if p.after_src in bad or p.after_src == y.define_src:
                continue
            try:
                name, body = define_to_lambda(p.after_src)
            except (ParseError, IndexError):
                continue
            # edsl-patch renames nothing: the rebind always targets the observed symbol.
            patch = self.e.parse_aura.patch_for(y.symbol, body, (p.intent or f"precedent #{p.id}")[:120])
            try:
                self.e.edsl_patch.validate_patch(patch, observe=inp["observe"])
            except self.e.edsl_patch.PatchError:
                continue
            return self._to_control(y, patch, f"edsl-patch:retrieval:memory:{p.id}", p.intent)
        return Control(y.symbol, y.define_src, "no usable precedent", "identity")

    def _propose_llm(self, y: Observation, inp: dict) -> Control:
        msgs = self.prompt(y)
        self.last_prompt = msgs
        ask = self._ask or self.e.baseline._ask
        src = f"edsl-patch:minimax:{self.cfg.get('model', '?')}"
        patch, exc, attempt = None, None, 0
        for attempt in range(self.llm_retries + 1):
            text = ""
            try:
                text = ask(self.cfg, msgs)
                patch = self.e.edsl_patch.validate_patch(self.e.baseline._parse(text), observe=inp["observe"])
                break
            except Exception as e:  # noqa: BLE001 - illegal output / API error
                exc = e
                # aura-loop addition (not in edsl-patch's zero-shot baseline): one repair turn with the
                # validator's error, still the same contract.
                msgs = msgs[:2] + [{"role": "assistant", "content": (text or "")[:2000]},
                                   {"role": "user", "content": f"Illegal patch: {e}. Reply with one legal JSON "
                                                               "array only, following the system contract."}]
        if patch is None:   # fall back to memory, labelled as such
            ctrl = self._propose_retrieval(y, inp)
            return Control(ctrl.symbol, ctrl.new_src, f"{ctrl.intent} (llm output rejected: {str(exc)[:120]})",
                           ctrl.source if ctrl.is_identity else ctrl.source.replace(
                               "edsl-patch:retrieval", "edsl-patch:minimax-fallback-retrieval"), ctrl.patch)
        if attempt:
            src += f":repaired{attempt}"
        return self._to_control(y, patch, src, None)

    def _to_control(self, y: Observation, patch: list[dict], source: str, intent: str | None) -> Control:
        last = patch[-1]
        if last["kind"] == "refuse":
            return Control(y.symbol, y.define_src, f"refuse/{last['op']}: {last['why']}", "identity", patch)
        if last["op"] != "rebind":
            return Control(y.symbol, y.define_src, f"unsupported synthesis op {last['op']}", "identity", patch)
        try:
            new_src = lambda_to_define(last["name"], last["body"])
        except ParseError:
            new_src = f"(define ({last['name']}) {last['body']})"   # plant will REJECT at the boundary
        return Control(y.symbol, new_src, intent or last.get("summary") or "", source, patch)


def dumps_patch(patch) -> str:
    return json.dumps(patch, ensure_ascii=False)
