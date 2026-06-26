# Remote GPU hosting — dashboard setup

Run the simulation **headless on the Windows PC** (RTX 5070 Ti) and drive it from the
MacBook through a small web dashboard over the home network (LAN). The simulation is
torch-based — each agent runs a neural net — and `agents/brain.py` auto-selects CUDA, so
the GPU is used automatically once torch sees it.

## One-time setup on the PC (Windows)

1. **Install Python 3.12** and clone the repo:
   ```
   git clone https://github.com/mick-gsk/artificial-society.git
   cd artificial-society
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```

2. **Install torch for Blackwell first** (from the CUDA 12.8 index — the generic
   `torch==2.8.0` pin does *not* reliably cover the RTX 5070 Ti's sm_120):
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install the rest** (core deps + the dashboard's `serve` extra), without pulling torch
   again:
   ```
   pip install -e ".[serve]"
   ```

4. **Verify the GPU is visible:**
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Must print `True NVIDIA GeForce RTX 5070 Ti`. If it prints `False`, the dashboard will
   still run but on CPU — the device badge turns red to warn you.

5. **Allow the port through Windows Firewall** — inbound TCP **8000** on the **Private**
   network profile only (keep it off Public).

## Running

```
scripts\run-dashboard.bat
```

The script sets `PYTHONHASHSEED=0` (needed for byte-reproducible seeded runs) and starts
the server on `0.0.0.0:8000`.

- On the PC: open `http://localhost:8000`.
- From the MacBook (same network): find the PC's LAN IP with `ipconfig` (IPv4 address),
  then open `http://<PC-IP>:8000`.

The status line shows the compute device (`cuda` / `cpu`). Set the run parameters
(seed / grid / pop), press **Start**, and watch the **live world**: agents move on a
biome grid with a food/water heat overlay, disturbances pulse, and the stat cards update
in real time. Frames are pushed over a WebSocket (`/ws`) at ~20 Hz — independent of the
sim's tick rate, so the view stays fluid even when the sim runs flat-out. **Stop** halts
the run.

## Frontend (dashboard UI)

The UI is a Svelte + Vite app (live world rendered with PixiJS/WebGL) under `frontend/`.
The **built assets are committed** to `artificial_society/serve/static/`, so the PC needs
**no Node toolchain** — `pip install -e ".[serve]"` + run is enough.

Rebuild the UI only when you change `frontend/` sources:

```
cd frontend
npm install
npm run build      # emits into ../artificial_society/serve/static (committed)
```

For UI development with hot-reload, run the API and the Vite dev server side by side:

```
PYTHONHASHSEED=0 python -m artificial_society.serve      # API on :8000
cd frontend && npm run dev                               # Vite on :5173, proxies /api + /ws
```

## Day-to-day: develop on the Mac, run on the PC

1. On the MacBook: commit + `git push` your changes.
2. On the PC: `git pull`, then restart `scripts\run-dashboard.bat`.

## Local testing on the MacBook (CPU)

`scripts/run-dashboard.sh` runs the same server locally (CPU) for UI checks. The test
suite runs anywhere: `venv/bin/python -m pytest -q`.

## Notes / not included

- One run at a time (no job queue), LAN-only (no auth/TLS), manual start (no Windows
  service). These were deliberately left out — see the implementation plan.
