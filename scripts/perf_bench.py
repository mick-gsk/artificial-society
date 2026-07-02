"""Performance benchmark + profiler for artificial-society.

Usage (from the repo root, with the venv active):
    venv\\Scripts\\python.exe scripts\\perf_bench.py sim   # CPU macro + cProfile + CPU forward microbench
    venv\\Scripts\\python.exe scripts\\perf_bench.py gpu   # GPU-vs-CPU forward microbench + episodic upload cost

Force CPU for the 'sim' mode with CUDA_VISIBLE_DEVICES=-1 (the GPU sim path currently crashes; see
docs/remote-host.md). Results and the optimization plan live in docs/performance-notes.md.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def build(pop: int, gw: int = 60, gh: int = 40):
    from artificial_society.simulation import Simulation

    return Simulation(
        headless=True, grid_w=gw, grid_h=gh, initial_population=pop, seed=42, load_checkpoint=False
    )


def macro(pop: int, warm: int = 5, n: int = 60):
    sim = build(pop)
    n0 = len(sim.agents)
    for _ in range(warm):
        sim.step()
    t0 = time.perf_counter()
    for _ in range(n):
        sim.step()
    dt = time.perf_counter() - t0
    return dt / n * 1000.0, n0, len(sim.agents)


def profile(pop: int, warm: int = 5, n: int = 140):
    sim = build(pop)
    for _ in range(warm):
        sim.step()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        sim.step()
    pr.disable()
    for sort in ("tottime", "cumulative"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(18)
        print(f"\n----- cProfile top 18 by {sort} (pop={pop}, {n} ticks) -----")
        print("\n".join(s.getvalue().splitlines()[:30]))


def time_forward(brain, batch: int, dev: torch.device, iters: int = 2000) -> float:
    from artificial_society.agents.brain import HIDDEN_SIZE, INPUT_SIZE

    obs = torch.randn(batch, INPUT_SIZE, device=dev)
    hid = torch.zeros(batch, HIDDEN_SIZE, device=dev)
    if dev.type == "cuda":
        for _ in range(50):
            brain.forward(obs, hid)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        brain.forward(obs, hid)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds per call


def micro_cpu() -> None:
    from artificial_society.agents.brain import Brain, device

    print(f"\n===== CPU forward microbench (device={device}) =====")
    b = Brain()
    with torch.inference_mode():
        for batch in (1, 8, 36):
            us = time_forward(b, batch, device)
            print(f"  batch={batch:>3}: {us:8.1f} us/forward   ({us / batch:7.2f} us per item)")


def micro_gpu() -> None:
    from artificial_society.agents.brain import INPUT_SIZE, Brain, device

    print(f"\n===== GPU vs CPU forward microbench (module device={device}) =====")
    if not torch.cuda.is_available():
        print("  CUDA not available; skipping.")
        return
    gb = Brain()  # on cuda (module device)
    cb = Brain().to("cpu")  # cpu copy
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    with torch.inference_mode():
        for batch in (1, 8, 36, 256):
            ug = time_forward(gb, batch, gpu)
            uc = time_forward(cb, batch, cpu)
            faster = "GPU" if ug < uc else "CPU"
            print(f"  batch={batch:>4}:  GPU {ug:8.1f} us   CPU {uc:8.1f} us   -> {faster} faster")
        print("\n  --- episodic-memory re-stack+upload cost (per planning step) ---")
        buf = [torch.randn(INPUT_SIZE) for _ in range(500)]  # CPU tensors, like the deque
        for _ in range(20):
            torch.stack(buf).to(gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            torch.stack(buf).to(gpu)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 1000 * 1e6
    print(
        f"  stack(500x{INPUT_SIZE})+to(cuda): {us:7.1f} us each (per planning step x agent x tick)"
    )


def bench_scale() -> None:
    """Target-scale baseline for the GPU-resident-engine design note (Tier 5).

    Fixed configurations — rerun after the engine lands for the before/after contract
    (docs/superpowers/specs/2026-07-02-gpu-resident-engine-design-note.md).
    """
    for pop, gw, gh, warm, ticks in ((36, 60, 40, 5, 60), (8, 200, 200, 2, 12), (200, 200, 200, 2, 12), (500, 200, 200, 1, 8)):
        sim = build(pop, gw, gh)
        for _ in range(warm):
            sim.step()
        t0 = time.perf_counter()
        for _ in range(ticks):
            sim.step()
        ms = (time.perf_counter() - t0) / ticks * 1000.0
        print(
            f"SCALE pop={pop:>4} grid={gw}x{gh:<4} ({gw * gh:>6} cells): {ms:8.1f} ms/tick  "
            f"{1000 / ms:6.2f} ticks/s  100k ticks = {ms * 100_000 / 3_600_000.0:6.2f} h"
        )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  mode={mode}")
    if mode == "sim":
        for pop in (8, 36):
            ms, n0, n1 = macro(pop)
            print(
                f"MACRO pop={pop:>2}: {ms:7.2f} ms/tick  ({1000 / ms:6.1f} ticks/s)  agents {n0}->{n1}"
            )
        micro_cpu()
        profile(8)
    elif mode == "gpu":
        micro_gpu()
    elif mode == "scale":
        bench_scale()
    else:
        print(f"unknown mode {mode!r}; use 'sim', 'gpu' or 'scale'")


if __name__ == "__main__":
    main()
