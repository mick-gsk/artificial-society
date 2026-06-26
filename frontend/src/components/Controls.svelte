<script>
  let { snap } = $props();

  let seed = $state(1);
  let grid_w = $state(60);
  let grid_h = $state(40);
  let pop = $state(36);
  let busy = $state(false);

  let running = $derived(snap?.status === "running");

  async function start(ev) {
    ev.preventDefault();
    busy = true;
    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, grid_w, grid_h, pop }),
      });
      if (r.status === 409) alert("A run is already active — stop it first.");
    } finally {
      busy = false;
    }
  }

  async function stop() {
    await fetch("/api/stop", { method: "POST" });
  }
</script>

<form class="controls" onsubmit={start}>
  <label>seed <input type="number" bind:value={seed} /></label>
  <label>grid_w <input type="number" bind:value={grid_w} /></label>
  <label>grid_h <input type="number" bind:value={grid_h} /></label>
  <label>pop <input type="number" bind:value={pop} /></label>
  <button type="submit" disabled={running || busy}>Start</button>
  <button type="button" class="ghost" onclick={stop} disabled={!running}>
    Stop
  </button>
</form>
