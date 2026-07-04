# RSSM (Dreamer-v3-style) Brain — Design Spec

**Date:** 2026-07-04
**Branch:** `experiment/rssm-dreamer-brain`
**Status:** Design approved (pending user spec review) → to be handed to writing-plans
**Author:** Claude (brainstormed with user; hardened by a 4-lens expert critique panel)

---

## 1. Context & Goal

The current agent "brain" ([agents/brain.py](../../../artificial_society/agents/brain.py)) uses a **deterministic MLP** as its world model (`world_fc → next_obs_head + reward_head`), and in the default **v1** path selects actions by **candidate evaluation**: an MPC planner (`plan_action → imagine_rollout`) samples ~12 candidate actions, rolls the MLP forward 2–6 steps, and picks the argmax. The policy itself is trained by on-policy PPO on *real* transitions; imagination is used **only at decision time**, never as a training signal.

The project's own research concluded the bottleneck is the **learning machinery / coupling**, not the world mechanics. This experiment replaces the deterministic MLP with a **stochastic latent RSSM (Dreamer-v3-style)** and trains the policy **entirely from imagined rollouts** (actor-critic in latent imagination), targeting better **sample efficiency** and **long-horizon planning**.

This is an **experimental, flag-gated lane**. The v1 and v2 brains are left completely untouched and serve as the control.

---

## 2. Fixed Constraints (user decisions — do not relitigate)

1. **Learning topology:** ONE **shared** RSSM world model, fed by the whole population's pooled experience, plus **per-agent** actor-critics trained in imagination. Individuality lives in the per-agent actors (+ gene conditioning).
2. **Substrate:** the **v1** path — 57-dim continuous observation (`INPUT_SIZE=57`), 7-dim continuous action (`ACTION_SIZE=7`, tanh-squashed Gaussian).
3. **Success metric:** a **sample-efficiency A/B**, headless, run on the GPU-PC.
4. **A/B arms:** **3 arms** — A (v1 baseline), B (full RSSM), **C (ablation:** shared RSSM world model used *only* as an MPC evaluator with the v1 planner, no imagination-trained actor).
5. **Newborn actor init:** **cultural warm-start** from a slowly-updated shared *prototype* actor — NOT genetic weight-copy from a parent. (Consistent with the already-shared world model; defuses the newborn-incompetence risk. The prior "no Lamarckian genetic inheritance" decision stands: no parent→child weight copy.)
6. **Reward:** the existing scalar reward is kept as the RSSM's reward *target*. Determinism routes through `seed_all` (extended — see §6).

---

## 3. Non-Goals / Scope

- **Not** touching v1/v2 brains, their tests, or their default behavior.
- **Not** claiming this lane demonstrates *emergent social learning*. A shared world model is a non-diegetic knowledge conduit that short-circuits exactly that story; this lane tests **sample efficiency only**. (Explicitly scoped per the multi-agent critique.)
- **Not** a full GPU-resident engine (Tier 5). Acting stays a batch-1 CPU loop; only imagination + WM training are batched (see §8).
- **Not** production-hardening the GPU FP16 path — we run FP32 and never call the crashing `imagine_rollout` code path.

---

## 4. Architecture

New module tree `artificial_society/agents/rssm/` (isolated from the frozen hot-file `brain.py`):

```
agents/rssm/
  world_model.py   # RSSMWorldModel (shared): encoder, prior, sequence GRUCell, heads, losses
  actor_critic.py  # ActorCritic (per-agent): actor, critic, imagination-trained update
  replay.py        # SharedReplay: per-agent-life segmented sequence buffer
  learner.py       # SharedLearner: owns WM + replay + WM optimizer + prototype actor; drives training
  prototype.py     # PrototypeActor: slowly-updated population template for cultural warm-start
  config.py        # RSSMConfig dataclass — all hyperparameters in one place
```

### 4.1 Shared World Model (RSSMWorldModel — one instance, sim-owned)

Dreamer-v3 RSSM adapted to a 57-d vector observation:

- **Deterministic recurrent state** `h ∈ ℝ²⁵⁶`, updated by an `nn.GRUCell` (per-step, deterministic — **not** cudnn `nn.GRU`, whose CUDA backward is nondeterministic):
  `h_t = GRUCell( MLP([z_{t-1}, a_{t-1}]) , h_{t-1} )`
- **Stochastic latent** `z` = **32 categoricals × 32 classes** (1024-d one-hot), straight-through gradients, **1% unimix** (uniform mixed into each categorical).
- **Encoder / posterior** `q(z_t | h_t, x_t)`: `symlog(x_t)` (per-dim) → MLP(LayerNorm, SiLU) → posterior logits.
- **Prior** `p(z_t | h_t)`: MLP → prior logits (used in imagination, no observation).
- **Heads** from model state `s = [h, z]` (**genes concatenated into the head inputs** so gene-dependent payoffs are visible in imagination):
  - **Decoder** → reconstruct observation, `symlog` target + MSE. **Grouped, not flat:** obs dims are partitioned into *predictable* (proprioception, local cell percepts, hormones, structures) vs *irreducible-from-egocentric-view* (social-nearby, episodic-retrieval). The latter group is **encoder-only** — fed to the encoder but **excluded from / heavily down-weighted in the decoder target**, so latent capacity is not burned reconstructing noise. Exact partition is derived from the `brain.py:24-41` obs layout and confirmed by a review agent during implementation.
  - **Reward** → symlog-twohot (bin support set from the empirical reward distribution *including* the death spike).
  - **Continue** (`= 1 − death`) → Bernoulli, with **terminal transitions upweighted** (stratified sampling ensures a minimum number of death transitions per batch) to counter class imbalance.
- **Loss:** `L = β_pred·(recon + reward + continue) + β_dyn·max(1, KL[sg(q)‖p]) + β_rep·max(1, KL[q‖sg(p)])`, with `β_pred=1.0, β_dyn=0.5, β_rep=0.1`, free-bits = 1 nat.
- **Optimizer:** its own Adam @ `lr=1e-4`, LayerNorm throughout, grad-clip.
- **Diagnostics:** per-dim reconstruction NLL and per-dim prior-KL logged, to detect capacity spent on unpredictable dims.

### 4.2 Per-agent Actor-Critic (ActorCritic — small, one per agent)

- **Actor** `π(a | s, g)`: MLP([h, z, genes]) → **tanh-Normal** over 7 continuous actions.
- **Critic** `v(s, g)`: MLP → **symlog-twohot** value.
- **Trained purely in imagination — differentiable dynamics-backprop (the crux):**
  - Reparameterized tanh-Normal actions + straight-through categorical latents make the imagined rollout **end-to-end differentiable**; the actor is trained by **backprop of normalized λ-returns through the world model** (NOT REINFORCE/score-function). A unit test asserts non-zero end-to-end action gradients — if this path is broken we've merely rebuilt PPO-in-imagination and the A/B has no reason to exist.
  - **λ-returns:** `λ=0.95`, discount **γ=0.997** with the predicted **continue** folded into the per-step discount (`γ·continue`), horizon **H=15**.
  - **Return normalization:** `S = EMA(percentile(R,95) − percentile(R,5))`; advantages divided by `max(1, S)` (the `max(1,·)` clamp is required — a known Dreamer-v3 footgun otherwise).
  - **Critic stability:** Dreamer-v3 style — λ-returns computed with the *current* critic + a **slow-critic (EMA) regularizer** term added to the critic loss; symlog return target.
  - **Entropy:** fixed coefficient `η = 3e-4` on the normalized-return objective (no schedule — Dreamer-v3 deliberately does not schedule it).
  - **Optimizer:** actor + critic Adam @ `lr=3e-5`.
- **Imagination start-states are RE-ENCODED with the current WM** (see §4.5) — never the stale carried `(h,z)`.
- **Per-agent replay indexing:** actor *i* imagines mostly from *its own* recent posterior states, age-blended with pooled states for coverage (more pooled when young/data-poor, more own as its life accumulates).

### 4.3 Cultural warm-start (PrototypeActor)

- A single **prototype actor** (same architecture as a per-agent actor) is maintained as a slow EMA of the population's actors (or of the best-performing cohort).
- **Newborn init:** a new agent's actor-critic is initialized **from the prototype** (a copy), then fine-tuned per-agent in imagination. This is *cultural* transmission (born into accumulated population skill), not *genetic* weight-copy from a parent — the "no Lamarckian genetic inheritance" decision is preserved.
- Individuality still emerges from per-agent fine-tuning + gene conditioning + local experience.
- **Prototype update:** EMA after each WM/AC training round; decay is a config knob.

### 4.4 Shared replay buffer (SharedReplay)

- Stores **raw** transitions `(obs, action, reward, continue, agent_id, genes)` — never encoded latents.
- **Segmented strictly per agent-life episode:** a training subsequence never crosses a death→birth seam (which would teach the GRU a spurious "teleport"). Short lives are **padded + masked**; padded steps are excluded from all losses.
- **Sequence sampling:** batch `B=16`, length `L=64` (sized toward the empirical lifespan distribution — a shorter L with more full sequences beats a long L that is mostly padding). `h` reset to zero at each subsequence start (burn-in approximation).
- **Truncation vs death:** `continue=0` fires **only** on real death; window-boundary truncations bootstrap (do not set continue=0).
- Indexed by `agent_id` to support per-agent imagination seeding (§4.2).

### 4.5 Training cadence & data flow

Per tick (in `Simulation.step()`, gated by `brain_arch=="rssm"`):
1. Each alive agent **acts** (batch-1, CPU): posterior inference through the shared WM given its `symlog(obs)` + carried `(h,z)`; samples action from its actor; carries the new `(h,z)` in `agent.hidden_state`; stores the raw transition to `SharedReplay`. (Carried `(h,z)` is used **only** for online acting.)
2. `agent.maybe_train()` is a **no-op** under rssm (training is centrally driven, not per-agent).

Every **K** environment-steps, a **centralized hook** (new, in `Simulation.step()`):
1. **WM update:** sample `B×L` per-life sequences → train `RSSMWorldModel`.
2. **Imagination update (batched across agents — see §8):** sample start-states per §4.2 → **re-encode with the current WM** via a short burn-in → roll each agent's actor + WM prior H steps → predict reward/continue/value → λ-returns → update actor+critic by differentiable backprop.
3. **Prototype EMA update.**

**Warmup / prefill:** a **prefill** phase (random-action rollouts) fills replay and the WM trains alone until it passes a loss threshold; only then do actor imagination updates begin. A newborn's imagination updates are likewise gated until the WM is warm.

**Train-ratio** (gradient steps per env-step) is the sample-efficiency lever; young actors get a higher train-ratio to reach competence within a short life. It is a config knob and is **matched/reported** in the A/B (§10).

---

## 5. Integration seams (real code, all behind the flag)

Mirrors the existing `physics_v2` flag pattern. `brain_arch: str = "v1"` threads through `Simulation` / `Agent`; env override `AS_BRAIN_ARCH`.

| Seam | File | Change |
|---|---|---|
| Flag + gated construction | [simulation.py](../../../artificial_society/simulation.py) `__init__` | Build `SharedLearner` **only** when `brain_arch=="rssm"` (an off-run must draw **zero** extra torch RNG, or the v1 golden breaks). Mirror `physics_v2` gating at `simulation.py:184-187`. |
| Learner reaches agents | [agent.py](../../../artificial_society/agents/agent.py) `update()` | Pass `learner` as a **kwarg** into `Agent.update()` (like `tribes/economy/technology`, `agent.py:1279-1281`). **Never** store the learner on the agent (else the live WM + optimizer + CUDA tensors pickle into every agent). |
| Dispatch | `agent.py` `update()` ~1327 | `if brain_arch=="rssm": brain_step = learner.act(self, features, self.hidden_state)` returning the same `{action_list, next_hidden, ...}` contract. |
| Centralized training | `simulation.py` `step()` | New post-step "every-K-steps" WM+imagination hook (hot file → **flag to core-lead at merge**). `agent.maybe_train()` no-ops under rssm. |
| Initial state | `agent.py:1346` / `ensure_fields` | rssm initial `(h,z)` comes from the learner; branch `ensure_fields` so the `INPUT_SIZE` rebuild guard (`simulation.py:146-151`) does **not** clobber rssm agents. Wire learner at every spawn site: `spawn_initial_population` (`:190`), `spawn_child_from_parent` (`:198`), `emergency_respawn` (`:305`). |
| Checkpoint | `simulation.py` `_save/_load_checkpoint` | Add `brain_arch` key + mismatch guard (mirror `physics_v2` guard `:470-474`); bump `CHECKPOINT_FORMAT_VERSION`. Save WM as `.cpu()` `state_dict` + WM optimizer state + prototype actor. Per-agent actor pickles with the agent (small, clean). Replay not checkpointed (fine — A/B runs are single-process). |
| A/B runner | new `scripts/rssm_ab.py` | Constructs with `load_checkpoint=False` (stale root `checkpoint.pkl` gotcha). |

---

## 6. Determinism & RNG

- **`seed_all` extended** ([rng.py](../../../artificial_society/rng.py)): add `torch.cuda.manual_seed_all(seed)` and, if any CUDA nondeterministic op remains, `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`. Simplest robust path: **run the act path on CPU** (repo default `AS_BRAIN_DEVICE=cpu`, 7–11× faster batch-1 anyway) and only batched WM-train/imagination on FP32 CUDA.
- **Split RNG streams** (required for a fair, pairable A/B):
  - **env/world/spawn stream** — bit-identical across arms for a given seed (enables paired within-seed analysis; the environment is the *same* across arms).
  - **policy/latent stream** — RSSM categorical sampling + action rsample draw here, never from the env stream.
  - Additionally, per-agent sub-generators seeded from `(global_seed, agent_id)` so latent draws are not order-dependent on agent iteration order (which changes with births/deaths).
- **Golden test** ([tests/test_regression_golden.py](../../../tests/test_regression_golden.py)): the default arm (v1) is **unchanged**, so with gated construction (zero extra draws when off) the golden should **not** move. Only if it does (it must not) do we treat regeneration as a reviewed, intentional event. The rssm lane has its **own** determinism test that asserts reproducibility **across a run with births/deaths**, not just a static roster.
- **vmap + RNG:** do not rely on the global generator inside `vmap`; pass RNG explicitly.

---

## 7. Checkpointing

- Payload gains `brain_arch` + a shape/arch mismatch guard; `CHECKPOINT_FORMAT_VERSION` bumped.
- WM saved as `.cpu()` `state_dict()` + `optimizer.state_dict()`; prototype actor saved; per-agent actors pickle within `self.agents` as today.
- The shared WM referenced by many agents is memoized **once** inside the single `pickle.dump` — but only because agents **do not** hold a reference to it (§5); it is saved via the sim payload.

---

## 8. Compute / performance plan

- **Act:** batch-1 CPU loop, ~150–300 µs/agent → ~10–20 ms/tick at N≈64. Tolerable next to the ~50 ms world update. Batched acting is **deferred**.
- **Imagination:** batched **across agents** from day one — this is the only part that blows up (N sequential H=15 rollouts batch-1 = hundreds of ms). Because the WM is shared, stack all agents' (re-encoded) start-states into one `(N, ·)` batch and roll H=15 once, on FP32 CUDA.
- **vmap over a fluctuating population:** pre-allocate a **max-population slab** of stacked actor params + Adam moments indexed by slot, with an **alive-mask**; newborns fill free slots, dead slots are masked out of the loss. The stacked shape stays constant (avoids recompilation cliffs and per-birth gather/scatter of optimizer state).
- **GPU only where it wins:** batched WM-train (`B·L`) and batched imagination (`N·H`); never the batch-1 act path (GPU is 11× slower there per perf-notes).
- FP32 throughout; never import `brain.py`'s `USE_FP16`/`device` or call `imagine_rollout` (avoids the known FP16-autocast crash).

---

## 9. Hyperparameters (RSSMConfig defaults)

| Group | Param | Value |
|---|---|---|
| Latent | deter `h` | 256 |
| Latent | stoch `z` | 32 × 32 categoricals, ST, 1% unimix |
| WM loss | β_pred / β_dyn / β_rep | 1.0 / 0.5 / 0.1 |
| WM loss | free bits | 1 nat |
| WM opt | Adam lr | 1e-4 |
| Replay | B × L | 16 × 64 (L tuned to lifespan dist.) |
| Imagination | horizon H | 15 |
| Returns | γ / λ | 0.997 / 0.95 |
| Return norm | S | EMA(p95−p5), divide by max(1,S) |
| Actor | entropy η | 3e-4 (fixed) |
| Actor/Critic opt | Adam lr | 3e-5 |
| Critic | target | symlog-twohot + slow-critic EMA reg |
| Prototype | EMA decay | config knob |
| Cadence | train ratio K | config knob (matched/reported in A/B) |

All values are Dreamer-v3 defaults except where the sim forces adaptation (grouped decoder, gene-conditioned heads, per-agent seeding, cultural warm-start).

---

## 10. Verification plan

### 10.1 Unit tests (`tests/agents/rssm/`)
- RSSM tensor shapes; posterior/prior logit shapes; KL + free-bits math; symlog/twohot round-trips.
- **Differentiable imagination:** assert non-zero end-to-end action gradients through the rolled WM.
- λ-return recursion with γ·continue; return-norm `max(1,S)` clamp.
- Replay: per-life segmentation (no death/birth seam crossing), padding+masking, stratified death sampling.
- Continue head: predicted continue drops before death on held-out sequences.
- **Determinism:** same seed → identical draws **including** a run with births/deaths.
- Cultural warm-start: newborn actor equals prototype at birth; diverges after fine-tuning.
- **Gating:** an `brain_arch!="rssm"` run draws zero extra torch RNG (golden digest unchanged).

### 10.2 A/B experiment (`scripts/rssm_ab.py`, run on GPU-PC)
- **Arms** (share one **env-RNG seed set**, paired): **A** = v1 baseline (MLP-WM MPC + PPO, at least as tuned as B); **B** = full RSSM (imagination-trained actor + cultural warm-start); **C** = ablation — shared RSSM world model used **only** as an MPC evaluator with the v1 planner, **no** imagination-trained actor (isolates world-model quality from imagination learning).
- **Primary endpoint (pre-registered, frozen, arm-agnostic, from physical state):** **median agent survival-age at a fixed tick budget T** — chosen over population/births because it is per-individual and therefore more robust to the population-feedback (rich-get-richer) confound; it is **not** either arm's shaped reward.
- **Secondary (all pre-registered):** cumulative births and mean population at T; AUC of a single *identical* evaluation-reward function applied post-hoc to all arms; per-capita reward; WM recon/reward/KL curves; actor entropy; achieved train-ratio; **population behavioral-diversity** (inter-agent action-distribution entropy) as a homogenization guardrail; **actor-competence-vs-age** curve.
- **Axes:** log every ~500 ticks vs **sim-ticks**, **cumulative agent-transitions**, AND **gradient-steps/wallclock** — report the sample-vs-compute frontier so a train-ratio win is not mislabeled sample-efficiency.
- **Protocol:** subprocess isolation per arm/seed; 3-seed pilot → variance/power estimate → commit N (~8–10 paired seeds); paired within-seed deltas; report medians + persistence/fraction-collapsed (collapsed runs retained under a pre-specified score); run length set to baseline plateau past the founder-overshoot transient.
- **Reproducibility log:** git SHA + config hash + seed + torch/CUDA versions per run.

### 10.3 Correctness floor
- Full `venv/bin/python -m pytest -q` green (v1/v2 tests unchanged).
- 500-tick headless run (`--headless --seed 42 --ticks 500`) does **not** collapse — a separate correctness smoke gate, distinct from A/B length.

---

## 11. Risks & open questions

- **Newborn incompetence (make-or-break):** mitigated by cultural warm-start + higher young-actor train-ratio; **measured** via actor-competence-vs-age with a pre-declared stopping criterion. If actors don't cross competence within median lifespan, the lane fails regardless of WM quality.
- **Homogenization:** the shared WM + pooled imagination push per-agent actors toward one policy modulo genes; tracked by the diversity guardrail. This lane is **not** evidence of emergent social learning (scoped out, §3).
- **Warm-vs-cold WM comparison:** the shared WM accumulates across generations, so a late-born agent is not "from scratch." The hypothesis level (collective sample-efficiency) is stated explicitly; WM warm-state is logged as a covariate.
- **Population feedback (rich-get-richer):** better policy → bigger population → more data → compounding. Mitigated by per-capita metrics + a capped carrying-capacity control if needed.
- **Hot-file coordination:** the `Simulation.step()` training hook and `agent.py` dispatch touch hot files — flag to core-lead at merge (per project convention).

---

## 12. Implementation phasing (feeds writing-plans)

1. **Scaffold + config** — `agents/rssm/` module tree, `RSSMConfig`, `brain_arch` flag threaded (gated, zero-draw-when-off), no behavior yet.
2. **World model** — `RSSMWorldModel` (encoder/prior/GRUCell/heads/losses) + unit tests (shapes, KL, symlog/twohot, diagnostics).
3. **Replay** — `SharedReplay` per-life segmentation + masking + stratified death sampling + tests.
4. **Actor-critic in imagination** — differentiable rollout, λ-returns, return-norm, critic, entropy + the non-zero-grad test.
5. **SharedLearner + prototype** — act path, centralized WM+imagination training hook, cultural warm-start, RNG split, per-agent seeding.
6. **Sim integration** — flag dispatch, learner-via-kwarg, spawn-site wiring, `ensure_fields` branch, checkpoint arch key/guard; full pytest green + 500-tick no-collapse.
7. **A/B harness** — `scripts/rssm_ab.py` (arms A/B/C, frozen endpoint, dual+compute axes, paired seeds, diversity + competence-vs-age metrics); 3-seed pilot on GPU-PC.
8. **Run + analyze** — commit N, paired analysis, sample-vs-compute frontier, write results note.

Each phase is independently testable; phases 2–4 are pure nets unit-testable without the sim; phase 6 is the only one touching hot files.
