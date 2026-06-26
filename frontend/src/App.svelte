<script>
  import { onMount } from "svelte";
  import Controls from "./components/Controls.svelte";
  import Cards from "./components/Cards.svelte";
  import World from "./components/World.svelte";

  // Aggregate run status + device come from a light REST poll; the live field
  // and stat values come from WebSocket frames.
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
  let device = $derived(snap.device?.type ?? "—");
</script>

<header>
  <h1>Artificial <span class="tick">/</span> Society</h1>
  <div class="station">
    <span>Observation&nbsp;Station</span>
    <span class="pill" class:running={snap.status === "running"}>
      <span class="led"></span>{snap.status}
    </span>
    <span class="pill" class:cuda={device === "cuda"}>
      device&nbsp;<b>{device}</b>
    </span>
  </div>
</header>

<Controls {snap} />
<World onFrame={(f) => (frame = f)} />
<Cards {stats} />
