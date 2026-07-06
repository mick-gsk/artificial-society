# RSSM (Dreamer-v3-style) Brain — Design Spec

**Date:** 2026-07-04
**Branch:** `experiment/rssm-dreamer-brain`
**Status:** Reviewed 2026-07-06 by a 4-agent tailored panel (RL-correctness, integration/determinism, experiment design, compute); all findings incorporated → to be handed to writing-plans
**Author:** Claude (brainstormed with user; hardened by a 4-lens expert critique panel + 2026-07-06 review round)

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
  - **Reward** → symlog-twohot with a **fixed** support: 255 bins linearly spaced in symlog space over [−20, 20] (Dreamer-v3; an empirical bin support is nonstationary — a moving target as the reward distribution shifts — and symlog already absorbs the death spike). Final layer **zero-initialized** (v3's cheap, load-bearing stability trick; same for the critic head, §4.2).
  - **Continue** (`= 1 − death`) → Bernoulli, with **terminal transitions upweighted** (stratified sampling ensures a minimum number of death transitions per batch) to counter class imbalance.
- **Loss:** `L = β_pred·(recon + reward + continue) + β_dyn·max(1, KL[sg(q)‖p]) + β_rep·max(1, KL[q‖sg(p)])`, with `β_pred=1.0, β_dyn=0.5, β_rep=0.1`. The `max(1,·)` clamp **is** the 1-nat free bits — do not apply free bits a second time; clamp the **per-timestep** KL before batch-averaging (official-impl behavior).
- **Optimizer:** its own Adam @ `lr=1e-4`, LayerNorm throughout, grad-clip.
- **Diagnostics:** per-dim reconstruction NLL and per-dim prior-KL logged, to detect capacity spent on unpredictable dims.

### 4.2 Per-agent Actor-Critic (ActorCritic — small, one per agent)

- **Actor** `π(a | s, g)`: MLP([h, z, genes]) → **tanh-Normal** over 7 continuous actions.
- **Critic** `v(s, g)`: MLP → **symlog-twohot** value.
- **Trained purely in imagination (the crux):**
  - **Actor gradient — Dreamer-v3-faithful default: REINFORCE with normalized returns**, for continuous actions too (v3 uses the score-function estimator for *both* action types; dynamics-backprop was V1/V2 — the earlier draft misattributed this and hard-banned exactly the estimator v3 validated, while borrowing its reinforce-tuned constants η/lr/return-norm):
    `L_actor = −Σ_t w_t · sg((R^λ_t − v(s_t)) / max(1,S)) · log π(a_t|s_t) − η·H[π]`, with the tanh log-det-Jacobian included in `log π`.
    `actor_grad ∈ {"reinforce","dynamics"}` is an RSSMConfig switch; V2-style dynamics-backprop stays available as an ablation. **Iff dynamics mode is used:** `sg(v(s_t))` on the baseline, WM parameters frozen, critic inputs detached during the actor update — under backprop an un-stop-gradiented baseline injects `−∂v/∂θ` into the actor gradient (biased). What separates arm B from PPO is *training on imagined rollouts of a shared world model*, not the gradient estimator — reinforce-in-imagination is still fully model-based.
  - **Imagination weighting:** per-step actor and critic losses are weighted by `w_t = sg(∏_{k≤t} γ·continue_k)` (cumulative predicted discount), so post-death imagined states do not train the policy — material in a sim with frequent deaths.
  - **λ-returns:** `λ=0.95`, discount **γ=0.997** with the predicted **continue** folded into the per-step discount (`γ·continue`), horizon **H=15**.
  - **Return normalization:** `S = EMA(percentile(R,95) − percentile(R,5))`; advantages divided by `max(1, S)` (the `max(1,·)` clamp is required — a known Dreamer-v3 footgun otherwise).
  - **Critic stability:** Dreamer-v3 style — λ-returns computed with the *current* critic + a **slow-critic (EMA) regularizer** term added to the critic loss; symlog-twohot return target (same fixed 255-bin support as reward), final layer zero-initialized. Additionally a small **replay-critic loss** (scale 0.3) on real replayed sequences, grounding values under multi-agent nonstationarity (official-impl practice).
  - **Entropy:** fixed coefficient `η = 3e-4`, no schedule (Dreamer-v3 deliberately does not schedule it). tanh-Normal has no closed-form entropy — use the sample-based estimate (`−log π` of sampled actions).
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
2. **Imagination update (batched across agents — see §8):** sample start-states per §4.2 → **re-encode with the current WM** via a short burn-in → roll each agent's actor + WM prior H steps → predict reward/continue/value → λ-returns → actor+critic update per the configured estimator (reinforce default, §4.2), cumprod-continue weighted.
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
| Dispatch | `agent.py` `update()` 1344-1349 (the features→hidden→`if self.physics_v2:` block) | `if brain_arch=="rssm": brain_step = learner.act(self, features, self.hidden_state)` returning the same `{action_list, next_hidden, ...}` contract. |
| Centralized training | `simulation.py` `step()` | New post-step "every-K-steps" WM+imagination hook (hot file → **flag to core-lead at merge**). `agent.maybe_train()` no-ops under rssm. |
| Initial state | `agent.py:1346` / `ensure_fields` | rssm initial `(h,z)` comes from the learner; branch the **whole brain-init block** in `ensure_fields` on `brain_arch` — both the `INPUT_SIZE` rebuild guard (`agent.py:146-151`, *not* simulation.py) **and** the `not hasattr(agent,"brain")` construction branch (`agent.py:143-145`), which would otherwise build a v1 `Brain` for checkpoint-loaded rssm agents. Wire learner at every spawn site: `spawn_initial_population` (`:190`), `spawn_child_from_parent` (`:198`), `emergency_respawn` (`:305`) — and tag `spawn_origin` there (`"birth"`/`"initial"`/`"respawn"`, needed by the §10.2 cohort filter). |
| Checkpoint | `simulation.py` `_save/_load_checkpoint` | Add `brain_arch` key + mismatch guard (mirror `physics_v2` guard `:470-474`); bump `CHECKPOINT_FORMAT_VERSION`. Save WM as `.cpu()` `state_dict` + WM optimizer state + prototype actor + the **actor slab** (params, both Adam moment tensors, per-slot step counts, slot↔agent_id map, alive-mask — see §7). Agents pickle only their **slot index**. Replay not checkpointed (fine — A/B runs are single-process). |
| A/B runner | new `scripts/rssm_ab.py` | Constructs with `load_checkpoint=False` (stale root `checkpoint.pkl` gotcha). Must **not** set/inherit `CUDA_VISIBLE_DEVICES=-1` (§8). |
| serve/ dashboard | `serve/runner.py` | `SimulationRunner` pins `brain_arch="v1"` unless explicitly passed in params — otherwise the `AS_BRAIN_ARCH` env override silently flips dashboard-launched sims to rssm. |

**Hot-file routing (process):** every seam above except `scripts/rssm_ab.py` and `serve/` touches a frozen hot file (`simulation.py`, `agents/agent.py` — CLAUDE.md contract). On this experiment branch we edit them directly (branch-local exemption); **at merge time the entire seam set is core-lead-routed**, not just the `step()` hook.

---

## 6. Determinism & RNG

- **`seed_all` extended** ([rng.py](../../../artificial_society/rng.py)): add `torch.cuda.manual_seed_all(seed)` and, if any CUDA nondeterministic op remains, `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`. Simplest robust path: **run the act path on CPU** (repo default `AS_BRAIN_DEVICE=cpu`, 7–11× faster batch-1 anyway) and only batched WM-train/imagination on FP32 CUDA.
- **RNG isolation rule** (replaces the earlier "split `seed_all` into streams" idea, which has no substrate: `rng.py` is 27 lines seeding only the `random`/`numpy`/`torch` **globals** — no stream objects exist, and v1 draw sites use bare globals, so restructuring them is golden-breaking):
  - **rssm code never touches the global streams.** All rssm randomness (categorical latent sampling, action sampling, replay sampling, prefill actions) draws from rssm-owned `torch.Generator` objects.
  - Per-agent generators seeded from `(global_seed, agent_id)` so latent draws are independent of agent iteration order (which changes with births/deaths); the learner's own draws (replay indices) from a dedicated `(global_seed, "learner")` generator.
  - v1/global paths remain byte-identical; `seed_all` gains only `torch.cuda.manual_seed_all(seed)`.
- **Pairing honesty:** cross-arm env identity holds only until trajectories diverge behaviorally — v1-arm births draw `torch.randn_like` from the global stream (`brain.py:253`), and events (fires) are agent-triggered. Pairing buys shared founders/terrain/exogenous schedule, i.e. variance reduction, **not** a same-environment guarantee (§10.2 handles this in the analysis).
- **Golden test** ([tests/test_regression_golden.py](../../../tests/test_regression_golden.py)): the default arm (v1) is **unchanged**, so with gated construction (zero extra draws when off) the golden should **not** move. Only if it does (it must not) do we treat regeneration as a reviewed, intentional event. Because the golden digest is coarse (population/energy/health per tick), add an explicit unit assertion that an off-arm `Simulation.__init__` leaves `torch.get_rng_state()` unchanged. The rssm lane has its **own** determinism test that asserts reproducibility **across a run with births/deaths**, not just a static roster.
- **vmap + RNG:** do not rely on the global generator inside `vmap`; pass RNG explicitly.

---

## 7. Checkpointing

- Payload gains `brain_arch` + a shape/arch mismatch guard; `CHECKPOINT_FORMAT_VERSION` bumped (currently `2`, `simulation.py:47`).
- **Single owner: the learner.** The actor **slab** (§8) is the authoritative actor-critic state — there are no per-agent `nn.Module` actors to pickle. Checkpoint payload, all saved centrally via the sim payload: WM `.cpu()` `state_dict()` + WM `optimizer.state_dict()` + prototype actor + **slab params, both Adam moment tensors, per-slot step counts, slot↔agent_id map, alive-mask**.
- Agents pickle only their **slot index**; they hold no tensors, no learner reference (§5). (The earlier "per-agent actors pickle within `self.agents`" line is dead: with a slab it either pickles stale copies or drags the full slab storage into every agent, and it silently loses the actors' Adam state on resume.)

---

## 8. Compute / performance plan

- **Device policy & the `CUDA_VISIBLE_DEVICES` gotcha:** the standing GPU-PC convention runs the sim with `CUDA_VISIBLE_DEVICES=-1` (FP16-crash workaround, `docs/remote-host.md` + run scripts). `scripts/rssm_ab.py` **must not set or inherit it** — otherwise arm B silently trains on CPU. The learner takes an explicit `device` in `RSSMConfig`, **asserts `torch.cuda.is_available()`** at A/B start for arms B/C, and the batch-1 act path is pinned to CPU regardless (GPU is 11× slower batch-1 per perf-notes).
- **Act:** batch-1 CPU loop. Tensor math ~150–300 µs/agent, but budget Python glue honestly: historically a 55 µs forward became ~1.4 ms/agent end-to-end, so plan for **0.5–1 ms/agent → ~30–60 ms/tick at N≈64**. Note the A/B runs at pilot scale (~60×40 grid, world update ~3–8 ms — the earlier "~50 ms world update" figure is the 200×200 number and doesn't apply), so the act loop **dominates** the tick; accepted cost, batched acting stays **deferred**.
- **Training cadence:** every K env-steps, **K default = 8** (config knob, pre-registered in §9). Estimated hook cost (launch-bound consumer GPU): WM update (B=16×L=64 through the 256-d GRUCell, BPTT over 64 steps) ≈ 30–80 ms + batched imagination (N×H=15 fwd+bwd) ≈ 20–50 ms → **~60–150 ms per hook**, amortized <20 ms/tick at K=8. The young-actor boosted train-ratio has a pre-registered ceiling (§9); wall-clock per arm is a reported axis (§10.2).
- **Imagination:** batched **across agents** from day one — the only part that blows up (N sequential H=15 rollouts batch-1 = hundreds of ms). Because the WM is shared, stack all agents' (re-encoded) start-states into one `(N, ·)` batch and roll H=15 once, on FP32 CUDA.
- **Actor slab over a fluctuating population:** pre-allocate a **max-population slab** — actor/critic params as stacked tensors (`(N_max, out, in)` per layer) driven by **hand-vectorized batched linears** (einsum/`baddbmm`). Plain Adam on the slab tensors is inherently per-agent (elementwise), differentiable through imagination, and sidesteps vmap-over-optimizer questions. (`torch.func` vmap + `stack_module_state` is stable on torch 2.8 / Python 3.9 as a fallback; do **not** use the deprecated `functorch.combine_state_for_ensemble`.) The stacked shape stays constant (no recompilation cliffs, no per-birth gather/scatter); newborns fill free slots.
- **Slot hygiene — Adam is not inert under masking:** zero grads still move masked params (momentum/`v` tails decay over many steps), and a newborn filling a freed slot would inherit stale moments. On death/spawn: **zero the param rows AND both Adam moment rows AND reset the per-slot step count**; and multiply the *parameter update* by the alive-mask, not just the loss.
- **GPU only where it wins:** batched WM-train (`B·L`) and batched imagination (`N·H`); never the batch-1 act path.
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
| Actor | gradient estimator | `actor_grad="reinforce"` (v3-faithful default; `"dynamics"` = ablation w/ mandatory stop-grads, §4.2) |
| Actor | entropy η | 3e-4 (fixed; sample-based estimate for tanh-Normal) |
| Actor/Critic opt | Adam lr / eps | 3e-5 / 1e-8 |
| Grad clip | WM / actor-critic | 1000 / 100 (v3 defaults) |
| Twohot | bins (reward & value) | 255, symlog-spaced over [−20, 20], fixed; head final layers zero-init |
| Critic | target | symlog-twohot + slow-critic EMA reg (decay 0.98) + replay-critic loss scale 0.3 |
| Return norm | EMA decay | 0.99 |
| Prototype | EMA decay | 0.995 (config knob) |
| Cadence | train hook every K env-steps | K=8 default (pre-registered); young-actor boosted train-ratio ceiling 4× base |

All values are Dreamer-v3 defaults except where the sim forces adaptation (grouped decoder, gene-conditioned heads, per-agent seeding, cultural warm-start).

---

## 10. Verification plan

### 10.1 Unit tests (`tests/agents/rssm/`)
- RSSM tensor shapes; posterior/prior logit shapes; KL + free-bits math; symlog/twohot round-trips.
- **Actor-gradient correctness (mode-dependent):** reinforce mode — actor loss produces gradients on actor params only (WM params receive **zero** grad from the actor loss; baseline `v` contributes none). Dynamics mode — non-zero end-to-end action gradients through the rolled WM **and** zero gradient into the critic/baseline (stop-grads verified).
- **Imagination weighting:** post-death imagined steps (continue→0) contribute ~zero actor/critic loss (cumprod-discount weighting works).
- λ-return recursion with γ·continue; return-norm `max(1,S)` clamp.
- Replay: per-life segmentation (no death/birth seam crossing), padding+masking, stratified death sampling.
- Continue head: predicted continue drops before death on held-out sequences.
- **Determinism:** same seed → identical draws **including** a run with births/deaths.
- Cultural warm-start: newborn actor equals prototype at birth; diverges after fine-tuning.
- **Gating:** a `brain_arch!="rssm"` run draws zero extra torch RNG — golden digest unchanged **plus** explicit `torch.get_rng_state()` unchanged across an off-arm `Simulation.__init__`.
- **Slot hygiene:** after death/respawn into the same slot, params, both Adam moments and step count are zeroed; masked slots' params bit-identical before/after a training step.

### 10.2 A/B experiment (`scripts/rssm_ab.py`, run on GPU-PC)
- **Arms** (share one **env-RNG seed set**): **A** = v1 baseline (MLP-WM MPC + PPO; "at least as tuned as B" is operationalized as an **equal tuning-run budget per arm**, logged); **B** = full RSSM (imagination-trained actor + cultural warm-start); **C** = ablation — shared RSSM world model used **only** as an MPC evaluator with the v1 planner, **no** imagination-trained actor (isolates world-model quality from imagination learning).
- **Primary endpoint (pre-registered, frozen, arm-agnostic, from physical state):** **Restricted Mean Survival Time (RMST) to horizon τ** over **natural-born agents only** — the `spawn_origin` flag (§5) excludes `emergency_respawn` cohorts, because the respawn floor (MIN_POPULATION=8 → 6 random age-0 agents per tick below it, `simulation.py:38-39,305,640`) floods collapsing runs with short-lived spawns; cohort = agents born after a pre-declared founder-transient cutoff; agents alive at T are **right-censored** (RMST handles censoring and stays estimable when the KM median is undefined; Kaplan-Meier curves reported descriptively). *(Replaces "median survival-age at T", which was ambiguous and mechanically confounded: more births → younger living population; death-only medians ignore censoring.)*
- **Primary comparison regime (pre-committed): transitions-matched** — sample-efficiency is the claim; the compute-matched (gradient-steps/wallclock) view is a pre-registered secondary, and the sample-vs-compute frontier is reported so a train-ratio win is not mislabeled sample-efficiency.
- **Confirmatory set:** {A-vs-B, B-vs-C} on RMST, **Holm-corrected**. **All other secondaries exploratory:** cumulative births, mean population, post-hoc identical eval-reward AUC, per-capita reward, WM recon/reward/KL curves, actor entropy, achieved train-ratio, **behavioral diversity** (homogenization guardrail), **actor-competence-vs-age** curve.
- **Pairing (honest version):** arms share seeds, but env identity erodes after behavioral divergence (§6). Analysis = **Wilcoxon signed-rank on within-seed deltas**; the pilot **must estimate the seed-pair correlation** — if r < 0.3, fall back to unpaired analysis with the honest larger-N requirement.
- **Power (designed, not asserted):** the 3-seed pilot estimates (i) SD of within-seed deltas, (ii) seed-pair correlation, (iii) per-arm collapse fraction; N is then computed for the pre-declared **minimum effect of interest: +25% RMST**. Reality check: 8–10 paired seeds detect only d≈1.0–1.3 at 80% power — if required N exceeds budget, claims are **downgraded in advance to estimation-with-CI**, not testing.
- **Collapse handling (pre-registered before the pilot):** collapse := population below MIN_POPULATION for **≥200 consecutive ticks**. Collapsed runs are retained (RMST over natural-born agents already reflects collapse as low RMST); **worst-rank assignment** run as a sensitivity analysis.
- **Axes:** log every ~500 ticks vs **sim-ticks**, **cumulative agent-transitions**, AND **gradient-steps/wallclock**.
- **Protocol:** subprocess isolation per arm/seed; run length set to baseline plateau past the founder-overshoot transient; **no interim peeking** at the primary endpoint; a **pre-registration file** (endpoint, estimator, cohort filter, transient cutoff, regime, N rule, collapse rule, test) is committed to the repo **before** the confirmatory seeds run; analysis code frozen at pilot end.
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
4. **Actor-critic in imagination** — imagination rollout, reinforce estimator (+ dynamics ablation w/ stop-grads), cumprod-continue weighting, λ-returns, return-norm, critic (+ replay loss), entropy + the mode-dependent gradient tests; actor slab + slot hygiene.
5. **SharedLearner + prototype** — act path, centralized WM+imagination training hook, cultural warm-start, RNG split, per-agent seeding.
6. **Sim integration** — flag dispatch, learner-via-kwarg, spawn-site wiring + `spawn_origin` tagging, `ensure_fields` branch (both brain-init branches), checkpoint arch key/guard + slab payload, serve-runner pin; full pytest green + 500-tick no-collapse.
7. **A/B harness** — `scripts/rssm_ab.py` (arms A/B/C, RMST endpoint, transitions-matched regime, dual+compute axes, paired seeds, diversity + competence-vs-age metrics, device assertion) + the committed **pre-registration file**; 3-seed pilot on GPU-PC → power calc → commit N.
8. **Run + analyze** — commit N, paired analysis, sample-vs-compute frontier, write results note.

Each phase is independently testable; phases 2–4 are pure nets unit-testable without the sim; phase 6 is the only one touching hot files.
