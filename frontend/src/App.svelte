<script>
  import { onMount } from "svelte";
  import Controls from "./components/Controls.svelte";
  import Cards from "./components/Cards.svelte";
  import World from "./components/World.svelte";

  // Aggregate run status + device come from a light REST poll; the live grid and
  // stat values come from WebSocket frames.
  let snap = $state({ status: "idle", device: null, stats: {} });
  let frame = $state(null);

  onMount(() => {
    const poll = async () => {
      try {
        snap = await (await fetch("/api/status")).json();
      } catch {
        /* server not up yet — keep last snapshot */
      }
    };
    poll();
    const id = setInterval(poll, 1500);
    return () => clearInterval(id);
  });

  let stats = $derived(frame?.stats ?? snap.stats ?? {});
</script>

<header>
  <h1>Artificial Society — Live</h1>
  <span class="status">
    status: {snap.status}{#if snap.device}
      &nbsp;·&nbsp;device: {snap.device.type}{/if}
  </span>
</header>

<Controls {snap} />
<World onFrame={(f) => (frame = f)} />
<Cards {stats} />
