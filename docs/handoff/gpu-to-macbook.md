# GPU-PC → MacBook handback (Stage 0a)

STATUS: READY
TAILSCALE_IPV4: NOT_INSTALLED
TAILSCALE_NAME: -
SSH: sshd=Running ssh-agent=Running key_present=yes firewall22=yes password_auth=on
ENV: python=3.12.13 torch=2.11.0+cu128 numpy=2.5.0 cuda_available=false
SMOKE: PASS — 3 out files (learned_seed1001.json 105 KB, recombiner_seed1001.json 352 KB, heartbeat.jsonl); "[pilot] done. 1 paired seeds"
BRANCH: feat/infra-research-stage0a @ 36544fd
NOTES: see below — the pilot env is READY, but the Tailscale connection path is unavailable.

## Connection: Tailscale is NOT installed on the GPU-PC

`tailscale` is not present (no command, no `C:\Program Files\Tailscale`, no service). So I cannot
give you a `100.x.y.z` address, and `ssh mickg@<tailscale-ip>` will not work yet.

Because your Mac's Tailscale also routes the PC's LAN IP into the tunnel (your original blocker),
plain LAN SSH may likewise fail while Tailscale is up on the Mac. Options, easiest first:

1. **Install Tailscale on the GPU-PC** (recommended — symmetric tailnet, then `ssh mickg@<100.x>`):
   needs Administrator (UAC) **and** an interactive login to your Tailscale account. I did not do
   this unprompted. Say the word (here or via the human) and I'll run
   `winget install tailscale.tailscale`, bring it up, and push an updated handback with the
   `100.x.y.z`.
2. **LAN fallback** — PC LAN IPv4 `192.168.178.76`, SSH port 22 open on the Private profile,
   key auth ready: `ssh mickg@192.168.178.76`. To reach it from the Mac, temporarily turn Tailscale
   off on the Mac, or exclude `192.168.178.0/24` from the tunnel (`--exit-node` / accept-routes off),
   so the LAN IP isn't captured.
3. **Coordinate via the repo** (what we're doing now) if neither is convenient — you hand me commands
   through `docs/handoff/`, I run them on the PC and push results back.

## Pilot smoke — details

`CUDA_VISIBLE_DEVICES=-1` (forced CPU), `PYTHONHASHSEED=0`:
`python -m artificial_society.research.run_pilot --smoke` → PASS.
- `learned`  seed 1001: attempts=1896, discoveries=319
- `recombiner` seed 1001: attempts=1896, discoveries=1645
- Wrote `artificial_society/research/out/{learned_seed1001,recombiner_seed1001}.json` + `heartbeat.jsonl`.
- Next step it suggests: `python -m artificial_society.research.analyze_gate --outdir ...\out`.

## Run-on-CPU is also the *right* call, not just a workaround

Independent of the FP16 GPU crash, a perf analysis I committed to `main`
(`git show origin/main:docs/performance-notes.md`) shows that for these tiny nets the **CPU is
7-11× faster than the GPU**, and the world-environment update — not the brains — dominates runtime.
So driving the full pilot on CPU is the performant choice here. The GPU FP16 fix in `agents/brain.py`
remains a separate core-lead PR (root cause + options are in `origin/main:docs/remote-host.md`).

## Confirmations
- Did NOT start the full pilot (smoke only), did NOT touch hot files / determinism tests, did NOT
  commit `research/out/`, `checkpoint.pkl`, or `venv/`.
