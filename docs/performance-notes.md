# Performance notes — measured baseline + optimization plan

Measured on the GPU host (Ryzen 7 9800X3D, RTX 5070 Ti, torch 2.11.0+cu128, Python 3.12.13)
on 2026-06-29. Headless, seeded (`seed=42`, `PYTHONHASHSEED=0`), `load_checkpoint=False`.
Reproduce with [`scripts/perf_bench.py`](../scripts/perf_bench.py):
`venv\Scripts\python.exe scripts\perf_bench.py sim` (CPU) and `... perf_bench.py gpu`.

> **Headline:** the dominant cost is **not** the neural brains — it is the **per-cell world
> environment update** (pure-Python scalar loops). And for these tiny nets the **CPU is 7-11×
> faster than the GPU**. The repo's premise that the sim "benefits from a CUDA GPU" does **not**
> hold for the current architecture; GPU is actively slower.

## Measurements

### Macro (CPU, ms per tick, grid 60×40)

| population | ms/tick | ticks/s |
|-----------:|--------:|--------:|
| 8          | 96.0    | 10.4    |
| 36         | 148.4   | 6.7     |

Decomposition (from the two points): **world update ≈ 81 ms/tick (population-independent)**,
brains ≈ **1.86 ms/agent**. So the world is ~84 % of runtime at pop 8, ~55 % at pop 36.

### cProfile (CPU, pop 8, 140 ticks, 39.2 s total) — by cumulative time

| share | call site |
|------:|-----------|
| 84 %  | `systems/world_regrowth._tick` → `world.update_environment` (`world.py:208`) |
| 57 %  | ↳ `environment/resources.regrow_cell` (`resources.py:190`) |
| 38 %  | ↳ `world.set_cell` (38 %) + `resources.clamp` (38 %) + `world.get_cell` (36 %) — overlapping |
| 23 %  | ↳ `world.diffuse_fields` (`world.py:152`) |
| 8 %   | `agents/agent.update` (all 8 agents) |
| 5 %   | `agents/brain.act` |

Self-time (tottime) is dominated by **`resources.clamp` — 33.6 M calls / 7.2 s**, plus
`builtins.max` (37.8 M) and `builtins.min` (34.4 M) it calls, and `world.get_cell`/`set_cell`
dict lookups (~11 M each). Torch ops are negligible (`_nn.linear` 0.26 s of 39 s).

### GPU vs CPU forward microbench (µs per `Brain.forward`)

| batch | GPU | CPU | winner |
|------:|----:|----:|:------:|
| 1     | 594 | 55  | **CPU 11×** |
| 8     | 590 | 66  | **CPU 9×** |
| 36    | 607 | 90  | **CPU 7×** |
| 256   | 299 | 220 | **CPU** |

CPU forward also batches well: batch-1 = 52 µs, batch-36 = 87 µs (2.4 µs/item) → batching the
population into one call is ~21× more efficient than 36 separate batch-1 calls.

### Episodic-memory re-upload (`brain.py:262`)

`torch.stack(500×57).to(cuda)` = **83 µs each**, executed every planning step × every agent ×
every tick (≈ 72×/tick at pop 36 once buffers fill → ~6 ms/tick of pure overhead).

## Optimization plan (priority by measured impact)

> **Status 2026-07-02:** Tier 1 and Tier 2 are ✅ DONE (`core/perf-tier1-world-vectorize`,
> merged). World cell state is numpy struct-of-arrays (`cell_store.py`) behind dict-compatible
> views; regrowth/events/diffusion are grid-wide array math, **bit-identical** to the scalar
> reference (golden stayed green; guard: `tests/environment/test_vectorized_equivalence.py`).
> Brains default to CPU (`AS_BRAIN_DEVICE=cuda` to opt back in). Measured on the GPU-PC
> (`perf_bench.py scale`, before → after): 36 @ 60×40 **137.5 → 56.8 ms/tick**;
> 8 @ 200×200 **1421.6 → 50.0 (28×)**; 200 @ 200×200 **1725.3 → 352.3**;
> 500 @ 200×200 **2104.8 → 740.0 ms/tick** (100k ticks: 58.5 h → 20.6 h).
> The world term is gone (~40-50 ms at 200×200, mostly the per-cell herb-RNG loop);
> the remaining cost is per-agent brain planning (`imagine_rollout`) at ~1.4 ms/agent —
> that is Tier 4 / the GPU-resident engine, and it touches `brain.py`/`agent.py`
> (coordinate with the capability-slice lane before starting).

### Tier 1 — Vectorize the world update *(the #1 hotspot, 55-84 % of runtime)* ✅ DONE

Replace the per-cell Python scalar loops with NumPy array ops over the whole grid.
- `environment/resources.py` — `regrow_cell` / `clamp` / `diffuse_step` (33 M `min`/`max` calls).
  `resources.py` is **not** a hot file → editable in a normal lane.
- `world.py` — `get_cell` / `set_cell` / `diffuse_fields`. `world.py` **is** a hot file →
  route through `core-lead`. Storing fields as `numpy` arrays (instead of per-cell dicts) removes
  the ~11 M dict lookups/100 ticks and lets regrowth/diffusion run as vectorized array math.
- **Determinism:** vectorized float reductions can change rounding vs scalar Python → may shift the
  golden trajectory. Treat as a deliberate change: regenerate the baseline, don't "fix" the test.

### Tier 2 — Run the brains on CPU, not GPU *(data-backed reversal)* ✅ DONE

For the current per-agent, batch-1 architecture the GPU is 7-11× **slower**. Default
`brain.py` `device` to CPU. This also sidesteps the FP16 autocast crash (see `docs/remote-host.md`)
— the CPU path works today. GPU would only be worth reconsidering after Tier 4 *and* at very large
populations, and even batch-256 favored CPU here.

### Tier 3 — Determinism-safe brain micro-wins *(numerically identical → golden stays green)*

Apply regardless of device; they shrink the 8-45 % brain share with no behavior change:
- Wrap the acting path in `torch.inference_mode()` — `brain.act` (`brain.py:348-372`) currently
  builds an autograd graph it immediately discards.
- Remove redundant forwards: `forward(obs,hidden)` runs 3× with identical inputs per act
  (`brain.py:350`, `:316`, `:359`); `:359` reproduces `:350`'s `next_hidden`. Collapse to one.
- Cache the episodic stack on-device and rebuild only when the buffer changes (kills the 83 µs
  re-upload at `brain.py:262`); same for `knowledge.py:57` `novelty`.
- Drop per-agent device syncs (`.item()` `brain.py:333`, `.tolist()` `:371`).

### Tier 4 — Batch the population *(larger refactor, hot file → core-lead/PR)*

Run all agents' brains in one batched call instead of a Python loop of batch-1 calls
(`simulation.py:442`). Weights differ per agent → `torch.func.functional_call` + `torch.vmap`
over stacked per-agent params. ~21× fewer forward calls; cuts the 1.86 ms/agent term. Do this on
**CPU** (Tier 2). Only meaningful once Tier 1 has removed the world bottleneck.

### Tier 5 — GPU-resident big-world engine *(decided 2026-07-02, scheduled after capability slice 1)*

The owner's target scale (hundreds of agents, 200×200+ grid, 100k+ tick runs) is out of reach
even for Tiers 1-4 (~6-10×). Decision: a torch-resident hybrid engine — world fields as GPU
tensors (regrowth = elementwise ops, diffusion = convolution), agents as struct-of-arrays,
brains in one vmapped batched pass, event-driven social systems stay Python. Own golden
baseline (GPU numerics ≠ v1 bytes). Full rationale, baseline table and acceptance criteria:
[`docs/superpowers/specs/2026-07-02-gpu-resident-engine-design-note.md`](superpowers/specs/2026-07-02-gpu-resident-engine-design-note.md).

**Target-scale baseline (measured 2026-07-02, GPU-PC CPU path, main @ 7b69f8e,
`perf_bench.py scale`):** 36 @ 60×40 = 137.5 ms/tick; 8 @ 200×200 = 1421.6; 200 @ 200×200 =
1725.3; 500 @ 200×200 = **2104.8 ms/tick (100k ticks ≈ 58.5 h)**. World term ≈ 1.41 s/tick at
the target grid (linear in cell count), brains ≈ 1.4 ms/agent.

### Process

Profile before/after every change (`scripts/perf_bench.py`). Tier 1 and Tier 2 change numerics →
regenerate the golden/headless-digest baselines deliberately. Tier 3 is numerically identical and
golden-safe. PPO cost: `PPO_EPOCHS=20` (`brain.py:57`) per agent every 128 ticks is also high —
4-10 is usually equivalent (numerics-changing → Tier 1/2 bucket).
