# aura-loop

把 Aura 代码库当成一个**状态空间**，用一个控制器（先是检索，之后是 LLM）根据观测给出 EDSL 变换，由 Aura 执行并门禁，结果写回记忆库，再进入下一步。这是一个最小 MVP。

*Treat an Aura workspace as a state space: a controller (retrieval now, LLM later) observes the code, proposes an EDSL transform, Aura (the plant) applies and gates it, and the outcome is written back to a memory DB that informs the next step. This repo is a minimal MVP.*

## 闭环刻画 / Closed-loop framing

```
            ┌──────────── memory (SQLite: KEEP / ROLLBACK / REJECT) ◄──────────┐
            │ retrieve precedents                                  write back │
            ▼                                                                 │
 x_t ──φ──► y_t ──► controller π(y, memory) ──u_t──► plant (Aura) ──verdict──┤
  ▲                                                     │                     │
  └──────────────────────── x_{t+1} (only on KEEP) ◄────┘─────────────────────┘
```

| | 含义 / meaning | MVP |
|---|---|---|
| **状态 x** | 可回放的代码世界：源码/AST、目标 define、参数、类型错误、历史指针 | `WorkspaceState`（一个文件里的一个 define） |
| **观测 y = φ(x)** | 控制器能看到的投影：define 源码、签名、类型错误、该符号历史 KEEP/DROP 计数 | `Observation` |
| **控制 u** | 限定作用域的 EDSL 变换（rebind 一个 define），不是自然语言 | `Control(symbol, new_src, intent, source)` |
| **被控对象 plant** | Aura：经 MutationBoundary 准入、执行、门禁 | `FakePlant`（真实可跑）/ `SoftPlant`（桩） |
| **记忆 memory** | 控制器的先验：历史转移及其结果，含失败 | SQLite 权威表 + `embedding` 预留列 |

Verdicts: **REJECT** = refused at the boundary (syntax / wrong symbol / signature change), state unchanged; **ROLLBACK** = applied but the gate still fails, rolled back; **KEEP** = gate clean, state advances. Every non-identity step is written back, so failures change the next decision.

Not state: interpreter heap, fiber scheduling, JIT caches — those are noise/constraints, not decision variables.

## MVP 范围 / scope

- Real: state/observation projection, SQLite memory with writeback, retrieval controller (same symbol + token-overlap similarity, skips candidates that previously failed), boundary admission (one define, same name & signature), toy arity gate, 3-step demo, tests.
- Stubbed: `SoftPlant` only checks `$AURA_BIN <file>` exits 0 (real version should drive `mutate:rebind` / MutationBoundary and Aura's type checker + tests); the toy arity table stands in for Aura's type checker; `embedding` column is unused (no vector DB yet); no LLM controller yet.
- Not yet: OceanBase / pgvector / LanceDB, multi-define or multi-file states, real repos.

## 运行 / Run

```bash
python3 -m aura_loop --demo                # seed memory, run 3 steps on fixtures/tiny.scm
python3 -m unittest discover -s tests -v   # or: pytest
# persistent memory:
python3 scripts/seed_memory.py --db loop.db && python3 -m aura_loop --demo --db loop.db --no-seed
# real Aura binary (stub plant):
AURA_BIN=/path/to/aura python3 -m aura_loop --demo --plant soft
```

Demo trace: step 1 retrieves the most similar KEEP precedent, which changes the signature → **REJECT** (written back); step 2 skips it and replays the arity fix `(min x hi)` → **KEEP**, state advances; step 3 the observation is healthy → no-op.

## Layout

```
aura_loop/state.py       WorkspaceState (x)
aura_loop/observe.py     φ: x -> Observation (y)
aura_loop/memory.py      SQLite authority table, retrieve, writeback
aura_loop/controller.py  RetrievalController (LLM controller: TODO)
aura_loop/plant.py       FakePlant, SoftPlant (stub)
aura_loop/loop.py        step / run
aura_loop/sexpr.py       tiny reader + toy arity checker
fixtures/tiny.scm        `clamp` with a wrong-arity `(min x)`
scripts/seed_memory.py   seed precedents
```

## Next

1. `SoftPlant` → real Aura workspace (`mutate:rebind`, MutationBoundary, type checker, tests).
2. `LLMController`: retrieved precedents as context, output constrained to one define.
3. Embeddings + vector recall alongside the SQLite authority table.
4. Larger fixtures / real repos; per-symbol state hashing for replay.

License: MIT.
