<script>
  import { onMount, onDestroy } from "svelte";
  import { WorldScene } from "../lib/world-render.js";
  import { connectWS } from "../lib/ws.js";

  let { onFrame } = $props();

  let host;
  let scene;
  let closeWS;

  let online = $state(false);
  let hud = $state({ tick: 0, agents: 0, w: 0, h: 0, fps: 0, events: 0 });

  onMount(async () => {
    scene = new WorldScene();
    await scene.init(host);
    scene.onHud = (h) => (hud = { ...hud, ...h });
    closeWS = connectWS({
      onOpen: () => (online = true),
      onClose: () => (online = false),
      onHello: (m) => scene.setLegend(m.biomes),
      onFrame: (f) => {
        scene.update(f);
        hud = {
          ...hud,
          tick: f.tick,
          w: f.grid.w,
          h: f.grid.h,
          agents: f.agents.length,
          events: f.events.length,
        };
        onFrame?.(f);
      },
    });
  });

  onDestroy(() => {
    closeWS?.();
    scene?.destroy();
  });
</script>

<div class="viewport">
  <div class="canvas-host" bind:this={host}></div>

  <!-- instrument frame: corner crop-marks + ambiance, all non-interactive -->
  <div class="frame" aria-hidden="true">
    <span class="brk tl"></span>
    <span class="brk tr"></span>
    <span class="brk bl"></span>
    <span class="brk br"></span>
    <div class="scan"></div>
    <div class="vignette"></div>
  </div>

  <div class="hud" aria-hidden="true">
    <span class="dot" class:live={online}></span>
    <span class="hud-label">FIELD</span>
    <span class="hud-val">{hud.w}×{hud.h}</span>
    <span class="hud-sep">·</span>
    <span class="hud-label">T</span>
    <span class="hud-val">{hud.tick}</span>
    <span class="hud-sep">·</span>
    <span class="hud-label">AGENTS</span>
    <span class="hud-val">{hud.agents}</span>
    {#if hud.events}
      <span class="hud-sep">·</span>
      <span class="hud-label amber">EVENTS</span>
      <span class="hud-val amber">{hud.events}</span>
    {/if}
    <span class="hud-sep">·</span>
    <span class="hud-label">FPS</span>
    <span class="hud-val">{hud.fps}</span>
  </div>

  <div class="legend" aria-hidden="true">
    <span class="lg-group">
      <span class="chip" style="--c:#49d17c"></span><span class="lg-label">sammeln</span>
      <span class="chip" style="--c:#3fc5f0"></span><span class="lg-label">kooperieren</span>
      <span class="chip" style="--c:#ff5d6c"></span><span class="lg-label">kampf</span>
      <span class="chip" style="--c:#ffb54d"></span><span class="lg-label">bauen</span>
      <span class="chip" style="--c:#9b8cff"></span><span class="lg-label">schlafen</span>
    </span>
    <span class="lg-sep">·</span>
    <span class="lg-group">
      <span class="icon tri" style="--c:#ffb54d"></span><span class="lg-label">camp</span>
      <span class="icon sq" style="--c:#74c69d"></span><span class="lg-label">farm</span>
      <span class="icon ring" style="--c:#64b5f6"></span><span class="lg-label">brunnen</span>
    </span>
  </div>
</div>

<style>
  .viewport {
    position: relative;
    width: 100%;
    height: 62vh;
    min-height: 340px;
    background: #05070b;
    border: 1px solid var(--line);
    border-radius: 4px;
    overflow: hidden;
  }
  .canvas-host {
    position: absolute;
    inset: 0;
  }
  .frame {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .brk {
    position: absolute;
    width: 14px;
    height: 14px;
    border: 1px solid var(--accent);
    opacity: 0.7;
  }
  .brk.tl {
    top: 8px;
    left: 8px;
    border-right: 0;
    border-bottom: 0;
  }
  .brk.tr {
    top: 8px;
    right: 8px;
    border-left: 0;
    border-bottom: 0;
  }
  .brk.bl {
    bottom: 8px;
    left: 8px;
    border-right: 0;
    border-top: 0;
  }
  .brk.br {
    bottom: 8px;
    right: 8px;
    border-left: 0;
    border-top: 0;
  }
  .scan {
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      rgba(0, 0, 0, 0) 0px,
      rgba(0, 0, 0, 0) 2px,
      rgba(4, 10, 16, 0.35) 3px
    );
    mix-blend-mode: multiply;
  }
  .vignette {
    position: absolute;
    inset: 0;
    background: radial-gradient(
      120% 120% at 50% 45%,
      rgba(0, 0, 0, 0) 55%,
      rgba(2, 4, 8, 0.55) 100%
    );
  }
  .hud {
    position: absolute;
    top: 10px;
    left: 28px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10.5px;
    letter-spacing: 0.12em;
    pointer-events: none;
    text-shadow: 0 0 6px rgba(0, 0, 0, 0.9);
  }
  .hud .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #45506a;
    margin-right: 2px;
  }
  .hud .dot.live {
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
  }
  .hud-label {
    color: var(--muted);
  }
  .hud-val {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .hud-sep {
    color: #2c3650;
  }
  .amber {
    color: var(--amber) !important;
  }

  .legend {
    position: absolute;
    bottom: 10px;
    left: 28px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 10px;
    letter-spacing: 0.1em;
    pointer-events: none;
    text-shadow: 0 0 6px rgba(0, 0, 0, 0.9);
    opacity: 0.85;
  }
  .lg-group {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .lg-label {
    color: var(--muted);
    margin-right: 6px;
    text-transform: uppercase;
  }
  .lg-sep {
    color: #2c3650;
  }
  .chip {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c);
    box-shadow: 0 0 5px var(--c);
  }
  .icon {
    width: 8px;
    height: 8px;
    display: inline-block;
  }
  .icon.tri {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 8px solid var(--c);
  }
  .icon.sq {
    border: 1.5px solid var(--c);
  }
  .icon.ring {
    border: 1.5px solid var(--c);
    border-radius: 50%;
  }

  @media (prefers-reduced-motion: reduce) {
    .scan {
      display: none;
    }
  }
</style>
