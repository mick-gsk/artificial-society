<script>
  import { onMount, onDestroy } from "svelte";
  import { WorldScene } from "../lib/world-render.js";
  import { connectWS } from "../lib/ws.js";

  let { onFrame } = $props();

  let host;
  let scene;
  let closeWS;

  onMount(async () => {
    scene = new WorldScene();
    await scene.init(host);
    closeWS = connectWS({
      onHello: (m) => scene.setLegend(m.biomes),
      onFrame: (f) => {
        scene.update(f);
        onFrame?.(f);
      },
    });
  });

  onDestroy(() => {
    closeWS?.();
    scene?.destroy();
  });
</script>

<div class="world" bind:this={host}></div>

<style>
  .world {
    width: 100%;
    height: 60vh;
    min-height: 320px;
    background: #0b0d12;
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }
</style>
