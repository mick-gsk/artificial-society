# GPU-PC → MacBook handback (Stage 0a)

STATUS: READY
TAILSCALE_IPV4: 100.66.237.24
TAILSCALE_NAME: hybrid-pace-1f3a   (MagicDNS: hybrid-pace-1f3a.tailbb69b1.ts.net)
SSH: sshd=Running ssh-agent=Running key_present=yes firewall22=yes password_auth=on
ENV: python=3.12.13 torch=2.11.0+cu128 numpy=2.5.0 cuda_available=false
SMOKE: PASS — 3 out files (learned_seed1001.json 105 KB, recombiner_seed1001.json 352 KB, heartbeat.jsonl); "[pilot] done. 1 paired seeds"
BRANCH: feat/infra-research-stage0a
NOTES: Tailscale is now installed and logged in on the GPU-PC — connect over the tailnet.

## Connect over Tailscale (ready now)

Tailscale was installed on the GPU-PC and authenticated to the `mick.gottsch@` tailnet.

```bash
ssh mickg@100.66.237.24
# or via MagicDNS:
ssh mickg@hybrid-pace-1f3a
```

- Node: `hybrid-pace-1f3a`, OS windows, account `mick.gottsch@`.
- The Tailscale adapter is classified **Private**, so the existing SSH firewall rule (port 22,
  Private profile) applies over the tunnel — no extra firewall change needed.
- Key auth is ready (your `…HVMSI9TjAbx…` key is in `administrators_authorized_keys`); password auth
  is still on as a fallback. Login shell is Windows PowerShell.

If the host key prompt blocks you on first connect, accept it (`StrictHostKeyChecking=accept-new`).

LAN fallback (if ever needed): `ssh mickg@192.168.178.76` — but Tailscale-on-Mac may capture that
LAN IP, so the tailnet route above is preferred.

## Pilot smoke — details

`CUDA_VISIBLE_DEVICES=-1` (forced CPU), `PYTHONHASHSEED=0`:
`python -m artificial_society.research.run_pilot --smoke` → PASS.
- `learned`  seed 1001: attempts=1896, discoveries=319
- `recombiner` seed 1001: attempts=1896, discoveries=1645
- Wrote `artificial_society/research/out/{learned_seed1001,recombiner_seed1001}.json` + `heartbeat.jsonl`.
- Suggested next: `python -m artificial_society.research.analyze_gate --outdir ...\out`.

## Run-on-CPU is also the *right* call, not just a workaround

A perf analysis I committed to `main` (`git show origin/main:docs/performance-notes.md`) shows that
for these tiny nets the **CPU is 7-11× faster than the GPU**, and the world-environment update — not
the brains — dominates runtime. So driving the full pilot on CPU is the performant choice. The GPU
FP16 fix in `agents/brain.py` remains a separate core-lead PR (root cause in
`origin/main:docs/remote-host.md`).

## Confirmations
- Did NOT start the full pilot (smoke only), did NOT touch hot files / determinism tests, did NOT
  commit `research/out/`, `checkpoint.pkl`, or `venv/`.
