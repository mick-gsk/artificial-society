# Remote GPU host — provisioned setup (this machine)

Concrete, already-provisioned state of the Windows GPU PC that hosts this repo and is
developed remotely from a MacBook. This complements the general how-to in
[`serve-setup.md`](serve-setup.md) with the actual host details, so an agent or contributor
working from the Mac knows the environment without re-discovering it.

Provisioned: 2026-06-29.

## Host

- **Machine:** `HYBRID-PACE-1F3` (Windows 11 Pro, build 26200)
- **LAN IP:** `192.168.178.76`  (network profile = Private)
- **User:** `mickg` (Administrator)
- **GPU:** NVIDIA RTX 5070 Ti, 16 GB, **sm_120** (Blackwell); driver exposes CUDA 13.2
- **Repo path on PC:** `C:\Projects\artificial-society`
- **venv:** `C:\Projects\artificial-society\venv` — CPython **3.12.13** (created via `uv`)
  - interpreter: `venv\Scripts\python.exe`
  - **torch 2.11.0+cu128** (installed from the CUDA 12.8 index so sm_120 is covered);
    `torch.cuda.is_available()` is `True` → `NVIDIA GeForce RTX 5070 Ti`
  - installed with `pip install -e ".[serve,dev]"`

## Connect from the MacBook

- **SSH:** `ssh mickg@192.168.178.76` — key-based (the Mac's public key is installed on the PC in
  `C:\ProgramData\ssh\administrators_authorized_keys`). Login shell = Windows PowerShell.
  Optional `~/.ssh/config`: `Host gpupc` / `HostName 192.168.178.76` / `User mickg`.
- **VS Code Remote-SSH:** connect to that host, open `C:\Projects\artificial-society`,
  select interpreter `venv\Scripts\python.exe`.
- **Dashboard:** on the PC run `scripts\run-dashboard.bat`, then open
  `http://192.168.178.76:8000` from the Mac.
- **Firewall:** inbound TCP **22** (SSH) and **8000** (dashboard) are allowed on the **Private**
  profile only.

## OpenSSH server

- **Standalone Win32-OpenSSH 10.0p2** in `C:\Program Files\OpenSSH` (installed standalone because
  the Windows DISM capability store was busy with an in-progress Windows Update at provisioning time).
- Services `sshd` + `ssh-agent` are **Automatic** and running. Config at
  `C:\ProgramData\ssh\sshd_config` includes the `Match Group administrators` block, so admin-user
  keys are read from `administrators_authorized_keys`.
- Password auth is still enabled as a fallback. Once key login is confirmed, harden by setting
  `PasswordAuthentication no` in `sshd_config` and `Restart-Service sshd` (as admin).

## Known issues (state at provisioning)

1. **GPU runs currently crash.** [`agents/brain.py`](../artificial_society/agents/brain.py)
   enables FP16 autocast on CUDA (`USE_FP16`, ~line 16), but `imagine_rollout` (~line 272) calls
   `self.encoder(pred_next_obs)` **outside** an `autocast` block with float16 inputs while the
   weights are float32 → `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`.
   CI only tests CPU, so it was never caught. **Until fixed, run on CPU** (`CUDA_VISIBLE_DEVICES=-1`).
   `brain.py` is a hot file → fix via the `core-lead` / lane + PR workflow.
2. **Golden-trajectory drift.** `tests/test_regression_golden.py::test_trajectory_matches_golden`
   fails locally because the golden baseline was generated under pinned versions (torch 2.8.0 /
   numpy 2.0.2, CPU/Py 3.9) while the documented GPU install pulls torch 2.11 / numpy 2.5 → tiny
   float differences. All other CPU tests pass. For local test runs, force CPU:
   `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`.

## Not installed (deliberately)

Docker and Ollama (not used by this project). No separate CUDA Toolkit — the torch cu128 wheels
bundle the CUDA runtime.
