# Handoff: MacBook Claude → GPU-PC Claude — Stage 0a pilot enablement

Two Claude Code instances coordinate **via this repo**. I run on the MacBook; you run on the
Windows GPU PC. I could not SSH in directly because Tailscale on the Mac routes the PC's LAN IP
into the tunnel. So: **you set up the environment and push back the info I need to connect**
(primarily the PC's Tailscale IPv4) by committing a small response file. Then I pull it, connect,
and drive the research pilot while observing live.

If you are the GPU-PC Claude reading this: do the tasks below in order, then write
`docs/handoff/gpu-to-macbook.md` from the template and push it.

## Context (read for orientation)
- Project: turning *Artificial Society* into a peer-reviewed AI paper on confound-controlled
  open-ended innovation. Design spec: `docs/superpowers/specs/2026-06-28-open-ended-innovation-research-design.md`.
- We are at **Stage 0a**: a pilot that asks whether the learned/social agents' *functional*
  innovation complexity beats a compute-matched random-recombiner null (pre-registered gate).
- The pilot tooling is the non-invasive package `artificial_society/research/` (no simulation
  source was edited). Entry points: `run_pilot`, `run_single`, `analyze_gate`.
- Provisioned host facts are in `docs/remote-host.md` on `main`
  (`git show origin/main:docs/remote-host.md` if you need them).

## Your environment (already provisioned)
- Windows 11, host `HYBRID-PACE-1F3`, user `mickg` (Administrator), shell PowerShell.
- Repo: `C:\Projects\artificial-society`
- Python: `.\venv\Scripts\python.exe` (CPython 3.12, torch 2.11.0+cu128, CUDA available).
- **GPU runs currently CRASH** — `agents/brain.py` enables FP16 autocast but `imagine_rollout`
  calls the encoder outside an autocast block (Half vs Float dtype mismatch). **Run everything on
  CPU** via `CUDA_VISIBLE_DEVICES=-1` until that hot-file bug is fixed via a separate core-lead PR.

## Tasks (PowerShell, in order)

**1. Get on the pilot branch** (this file lives here):
```powershell
cd C:\Projects\artificial-society
git fetch origin
git checkout feat/infra-research-stage0a
git pull --ff-only
```

**2. Connection info I need — Tailscale IPv4 of this PC:**
```powershell
tailscale ip -4
tailscale status | Select-Object -First 3
```
If `tailscale` is **not recognized**, Tailscale is not installed on this PC — record
`TAILSCALE_IPV4: NOT_INSTALLED` (we will fall back to a LAN approach).

**3. Verify the SSH path** so `ssh mickg@<tailscale-ip>` will work for me:
```powershell
Get-Service sshd, ssh-agent | Select-Object Name, Status, StartType
Select-String -Path "$env:ProgramData\ssh\administrators_authorized_keys" -SimpleMatch "HVMSI9TjAbx"
Get-NetFirewallRule -DisplayName "*SSH*" | Select-Object DisplayName, Enabled, Profile
Select-String -Path "$env:ProgramData\ssh\sshd_config" -Pattern "PasswordAuthentication"
```
The `HVMSI9TjAbx` substring is from my public key (full key at the bottom). It must be present in
`administrators_authorized_keys`. If missing, add it (it is below).

**4. Verify the Python env on CPU:**
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
.\venv\Scripts\python.exe -c "import torch,numpy; print('torch',torch.__version__,'numpy',numpy.__version__,'cuda',torch.cuda.is_available())"
```

**5. Smoke-test the pilot on CPU** (≈1–2 min — proves the pipeline runs here):
```powershell
$env:CUDA_VISIBLE_DEVICES="-1"; $env:PYTHONHASHSEED="0"
.\venv\Scripts\python.exe -m artificial_society.research.run_pilot --smoke
```
Expect it to write `artificial_society\research\out\learned_seed1001.json` and
`recombiner_seed1001.json` and print `[pilot] done.`. Record PASS/FAIL + any error text.

**6. Push your findings back:**
```powershell
# write docs/handoff/gpu-to-macbook.md (template below), then:
git add docs/handoff/gpu-to-macbook.md
git commit -m "chore(handoff): GPU-PC status + connection info for MacBook Claude"
git pull --rebase origin feat/infra-research-stage0a
git push
```

## Do NOT
- Do **not** start the full pilot — the MacBook Claude drives it to observe live. **Smoke only.**
- Do **not** edit hot files (`simulation.py`, `world.py`, `agents/agent.py`, `agents/brain.py`,
  `environment/materials.py`, `systems/registry.py`) or any determinism test. The GPU FP16 fix is
  a separate core-lead PR, not part of this handoff.
- Do **not** commit `research/out/`, `checkpoint.pkl`, `venv/`, or `node_modules/`.

## My MacBook public key (add to administrators_authorized_keys if missing)
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHVMSI9TjAbx+0NggZNL36BxcszEJLO94x2uRUY9Fuzb moritzbecker@macbook -> gpu-pc (artificial-society pilot)
```

## Response template — write to `docs/handoff/gpu-to-macbook.md`
Replace every `<…>` with a real value, then commit + push (task 6).
```
# GPU-PC → MacBook handback (Stage 0a)

STATUS: READY | BLOCKED
TAILSCALE_IPV4: <100.x.y.z | NOT_INSTALLED>
TAILSCALE_NAME: <magicdns short name | ->
SSH: sshd=<Running|Stopped> ssh-agent=<Running|Stopped> key_present=<yes|no> firewall22=<yes|no> password_auth=<on|off>
ENV: python=<3.12.x> torch=<2.11.0+cu128> numpy=<ver> cuda_available=<true|false>
SMOKE: <PASS|FAIL> — <out file count / one-line error>
BRANCH: feat/infra-research-stage0a @ <short-sha>
NOTES: <anything blocking or noteworthy, e.g. tailscale not installed>
```

Once you push that file, I (MacBook Claude) pull it, connect via the Tailscale IP, and launch the
full pilot (`run_pilot`, 12 seeds × 8000 ticks, CPU) with live heartbeat monitoring, then run
`analyze_gate` for the gate verdict.
