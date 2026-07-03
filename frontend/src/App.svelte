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
  // Selected agent id lifted from World, via onSelect. Consumed by the personal
  // Chronik (Inspector): fed to the differ as a cap-bypass getter, and used to
  // filter the 600-entry ring buffer down to one agent's story.
  let selectedId = $state(null);
  // The differ reads the getter live (not a snapshot) so cap-bypass always
  // reflects the *current* selection, not the selection at differ-creation time.
  const differ = createFeedDiffer(() => selectedId);
  let feedSeq = 0;
  // Ring buffer of every entry the differ ever produced (incl. personalOnly
  // ones the global feed filters out) — the raw material for the personal
  // Chronik. Kept separate from the capped 80-entry global `feed`.
  let chronikBuf = $state([]);

  function onFrame(f) {
    frame = f;
    const fresh = differ(f);
    if (fresh.length) {
      for (const e of fresh) e.key = feedSeq++;
      chronikBuf = [...chronikBuf, ...fresh].slice(-600);
      // personalOnly entries exist purely to keep the per-agent chronicle
      // lückenlos past the global caps — they must never also appear in the
      // capped world feed (that would both duplicate them and let the cap
      // bypass leak into the global view).
      const globalFresh = fresh.filter((e) => !e.personalOnly);
      if (globalFresh.length) {
        feed = [...globalFresh.reverse(), ...feed].slice(0, 80);
      }
    }
  }

  function onSelect(id) {
    selectedId = id;
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
  <World {onFrame} {onSelect} chronik={chronikBuf} />
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
