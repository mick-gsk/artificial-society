<script>
  import { onMount } from "svelte";
  import Controls from "./components/Controls.svelte";
  import Cards from "./components/Cards.svelte";
  import World from "./components/World.svelte";
  import Feed from "./components/Feed.svelte";
  import { createFeedDiffer } from "./lib/feed.js";

  // Aggregate run status + device come from a light REST poll; the live field
  // and stat values come from WebSocket frames.
  let snap = $state({ status: "idle", device: null, stats: {} });
  let frame = $state(null);
  let feed = $state([]);
  const differ = createFeedDiffer();
  let feedSeq = 0;

  function onFrame(f) {
    frame = f;
    const fresh = differ(f);
    if (fresh.length) {
      for (const e of fresh) e.key = feedSeq++;
      feed = [...fresh.reverse(), ...feed].slice(0, 80);
    }
  }

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
<div class="stage">
  <World {onFrame} />
  <Feed entries={feed} />
</div>
<Cards {stats} />

<style>
  .stage {
    display: flex;
    gap: 12px;
    align-items: stretch;
  }
  .stage :global(.viewport) {
    flex: 1;
    min-width: 0;
  }
  @media (max-width: 900px) {
    .stage {
      flex-direction: column;
    }
    .stage :global(.feed) {
      width: 100%;
      max-height: 220px;
    }
  }
</style>
