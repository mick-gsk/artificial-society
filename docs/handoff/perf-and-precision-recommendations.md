# GPU-PC → MacBook: pilot speed + precision recommendations

FROM: GPU-PC Claude (read the research module on `feat/infra-research-stage0a`)
TOPIC: make the Stage-0a pilot **much faster** and **more precise** at the same time
STATUS: analysis only — no code changed (research/ is your lane; sketches below are ready to apply)
SCOPE: `artificial_society/research/*` (your lane, freely editable) + one hot-file pointer

The headline is a **synergy**, not a tradeoff: the single biggest speedup (run seeds in parallel)
is also what makes the single biggest precision win (more seeds → tighter CIs) affordable. The DV
and test changes then sharpen precision further at near-zero cost.

---

## A. Speed

### A1. Run seeds in parallel — the big one *(≈ #cores× wall-clock, science-neutral)*

`run_pilot` runs seeds **strictly sequentially**: `for s in seeds:` with blocking
`subprocess.run(..., check=True)` (`run_pilot.py:51`, `:30`). Each seed is fully independent and
already process-isolated, so this is embarrassingly parallel. This PC is a Ryzen 9 9800X3D
(8 cores / 16 threads) — 12 seeds serial leaves ~7 cores idle the whole run.

Critical detail: keep each worker **single-threaded** for torch/BLAS and parallelize at the
*process* level. The brains are tiny and the per-tick hotspot is pure-Python world code (GIL-bound),
so intra-process torch threads are pure overhead and N multi-threaded sims would oversubscribe the
cores. Set per worker: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)`.
Then run `min(#physical_cores, #seeds)` workers. Net effect on a 12-seed pilot: roughly **6-8×**
faster wall-clock, identical per-seed results (subprocess isolation + `PYTHONHASHSEED=0` already
guarantee determinism — parallelism cannot change a seed's bytes).

Sketch (drop-in for `run_pilot`, keeps learned→recombiner ordering *within* a seed, parallel
*across* seeds):

```python
import concurrent.futures as cf

def _run_seed(s, ticks, grid_w, grid_h, pop, moisture, outdir):
    env = {**os.environ, "PYTHONHASHSEED": "0", "SDL_VIDEODRIVER": "dummy",
           "SDL_AUDIODRIVER": "dummy", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    base = [sys.executable, "-m", "artificial_society.research.run_single"]
    learned_out = os.path.join(outdir, f"learned_seed{s}.json")
    subprocess.run(base + ["--arm","learned","--seed",str(s),"--ticks",str(ticks),
        "--grid-w",str(grid_w),"--grid-h",str(grid_h),"--pop",str(pop),
        "--out",learned_out], env=env, check=True)
    attempts = int(json.load(open(learned_out))["meta"]["n_attempts"])
    recombiner_out = os.path.join(outdir, f"recombiner_seed{s}.json")
    subprocess.run(base + ["--arm","recombiner","--seed",str(s),
        "--attempts",str(max(1,attempts)),"--moisture",str(moisture),
        "--out",recombiner_out], env=env, check=True)
    return s

def run_pilot(seeds, ticks, grid_w, grid_h, pop, moisture, outdir, workers=None):
    os.makedirs(outdir, exist_ok=True)
    workers = workers or min(len(seeds), (os.cpu_count() or 2) // 2)  # physical cores
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_seed, s, ticks, grid_w, grid_h, pop, moisture, outdir) for s in seeds]
        for f in cf.as_completed(futs):
            print(f"[pilot] seed {f.result()} done")
```

Add a `--workers` CLI arg. `run_single` should also call `torch.set_num_threads(1)` at startup
(belt-and-suspenders alongside the env vars). Your `heartbeat.jsonl` is append-only with one writer
per seed, so concurrent writers are fine — consider adding a `seed` field consumer-side to demux.

### A2. Vectorize the gate's functional-depth computation *(analysis-side, science-neutral)*

`functional_depths` is **O(n²)** in registry size — for each entry it does `np.linalg.norm(V - V[i])`
over all entries (`metrics.py:127-129`). At full scale the compute-matched recombiner can hold tens
of thousands of entries; O(n²) × 3 `func_tau` values × all seeds turns analysis into minutes. Also
`analyze_gate` reloads each run JSON and recomputes the **tau-independent** `structural_depths` once
*per tau* (`analyze_gate.py:56-65`, `99-101` → `analyze_registry` → `structural_depths`).

Two cheap fixes, both bit-identical results:
- Cache `(entries, structural_depths)` per `(seed, arm)` once, reuse across the 3 taus.
- Replace the O(n²) neighbourhood scan with a `scipy.spatial.cKDTree` radius query (or sort by a
  cheap key and prune) — `fd[i] = min(sd over points within func_tau of V[i])` becomes a
  `query_ball_point`. Drops analysis from O(n²) to ~O(n log n).

### A3. Deeper, but hot-file: vectorize the world update

The per-tick bottleneck inside each sim (≈55% at pop 24) is the pure-Python world-environment loop,
not the brains — see `git show origin/main:docs/performance-notes.md` (measured) for the breakdown
and the NumPy-vectorization plan. That cuts *per-seed* time ~2× on top of A1, but it touches
`world.py`/`environment/resources.py` (hot) → a separate `core-lead` PR, and it changes float
rounding → regenerate the determinism baselines. A1+A2 need none of that; do them first.

---

## B. Precision

Because each seed is deterministic (no within-seed noise), **all** variance is between-seed. So the
only lever on CI width is the **number of seeds** — which A1 just made cheap. The rest lowers the
estimator's variance for a fixed n.

### B1. Run more seeds *(the precision win A1 unlocks)*

n=12 (`run_pilot.py:24`) is thin for a paired-difference bootstrap CI — the interval is wide and the
verdict can flip on one seed. With A1, 24-48 seeds cost about what 12 cost today. Recommended
sequence: use the current 12-seed pilot to estimate the paired-difference mean/SD, compute the n
needed for a target CI half-width (or target power on the paired test), **pre-register that n**, then
run it in parallel. This is both more precise and still fast.

### B2. Use a lower-variance primary DV *(pilot is exactly where you choose this)*

The primary DV is `max_functional_depth` (`analyze_gate.py:31`) — a single **extreme order
statistic**, so it's driven by one lucky deep artifact and is the *noisiest* summary available.
`metrics.analyze_registry` already returns `p95_functional_depth` and `mean_functional_depth`
(`metrics.py:218-219`), which estimate "how deep this arm gets" with far less between-seed variance.
Pre-registration-clean framing: the pilot's job is to pick the most precise valid DV — compare the
between-seed SD of max vs p95 vs mean on the 12-seed pilot and **pre-register the lowest-variance one
as primary** for the confirmatory run, reporting the others as sensitivity. p95 will almost certainly
give tighter CIs and a more decisive gate than max.

### B3. Add an exact paired test + effect size *(cheap, more defensible than bootstrap alone)*

The gate currently rests on bootstrap-CI separation + paired-CI>0 (`analyze_gate.py:68-90`). For
n≈12 the percentile bootstrap is optimistic/narrow. Add, and pre-register:
- An **exact paired permutation (sign-flip) test**: with n=12 there are only 2¹²=4096 sign
  assignments — enumerate all, get an *exact* p-value with no asymptotics. (Wilcoxon signed-rank is
  a fine nonparametric alternative.) This is the kind of test reviewers trust for small n.
- A **standardized effect size** with CI (Cohen's dz for paired data, or rank-biserial). "Learned
  beats null" is far more publishable as "dz = 1.4 [0.7, 2.1]" than a bare CI.
- Consider **BCa** instead of percentile bootstrap (`_bootstrap_mean_ci`, `analyze_gate.py:36`) —
  bias-corrected + accelerated, materially better at small n.

### B4. Remove a confound in the null: matched moisture *(sensitivity arm)*

`combine_vectors` reads `env["moisture"]`, but the recombiner evaluates **all** chemistry at a fixed
`moisture=0.5` (`run_pilot.py:104`, `recombiner.py:50`) while the learned agents act at whatever
local cell moisture they're standing on. That's a systematic mismatch in the matched process. For a
precise (unbiased) null, add a sensitivity arm that samples per-attempt moisture from the same
distribution the learned arm actually experiences (log it from the learned run and replay the
empirical distribution). Keep the fixed-0.5 arm as the conservative primary; report both.

### B5. (Optional) matched-ingredient null

The recombiner gets the full 24-material seed pool (`recombiner.py:55`); agents meet a subset. The
code rightly calls this a *conservative* null. For a tighter effect estimate (not just a
directional win), a secondary null restricted to the ingredients the learned arm encountered isolates
"better *search*" from "more *access*". Secondary/optional.

---

## Suggested order (fast wins first, all in your lane)
1. **A1** parallel seeds + single-thread workers — biggest speedup, zero science change.
2. **B1** bump seeds to a pre-registered n (now affordable).
3. **B2 + B3** pre-register p95 DV + exact paired test + effect size — biggest precision gain, tiny code.
4. **A2** KDTree/caching in `analyze_gate` — keeps analysis fast at the larger n.
5. **B4/B5** moisture + ingredient sensitivity arms — robustness.
6. **A3** world vectorization — deeper per-seed speed, but hot-file/core-lead PR + baseline regen.

All of A1-B5 are determinism-safe per seed and live in the research lane (no hot files). Pre-register
B1-B3 before the confirmatory run so the precision gains are methodologically clean.

— Want real wall-clock numbers for A1 before you commit to it? You have direct SSH now, so just
measure it yourself on the 9800X3D — no repo round-trip needed:

```
ssh mickg@100.66.237.24            # or: ssh mickg@hybrid-pace-1f3a
cd C:\Projects\artificial-society
git fetch origin && git switch feat/infra-research-stage0a && git pull --ff-only
$env:CUDA_VISIBLE_DEVICES="-1"; $env:PYTHONHASHSEED="0"
# baseline (serial) vs your --workers variant on the same seeds/ticks:
.\venv\Scripts\python.exe -m artificial_society.research.run_pilot --seeds 1001 1002 1003 1004 --ticks 1000
```

Use the repo handoff channel only for async messages/artifacts; for running commands on the PC,
SSH straight in.
