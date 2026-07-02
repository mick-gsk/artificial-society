<script>
  import { onMount, onDestroy } from "svelte";
  import { WorldScene } from "../lib/world-render.js";
  import { connectWS } from "../lib/ws.js";

  let { onFrame } = $props();

  let host;
  let scene;
  let closeWS;

  let online = $state(false);
  let hud = $state({ tick: 0, agents: 0, w: 0, h: 0, fps: 0, events: 0, zoom: 1 });
  let sel = $state(null); // live data of the inspected agent

  const ACT_NAME = ["unterwegs", "sammelt", "kooperiert", "kämpft", "baut", "schläft"];
  const STAGE_NAME = ["Kind", "Erwachsen", "Ältester"];
  const TOOL_NAME = ["—", "Stein", "scharfe Klinge"];

  let selectedId = null;

  function refreshSel(f) {
    if (selectedId == null) return;
    const a = f.agents.find((x) => x.id === selectedId);
    if (!a) {
      selectedId = null;
      sel = null;
      if (scene) scene.selectedId = null;
      return;
    }
    sel = a;
  }

  onMount(async () => {
    scene = new WorldScene();
    await scene.init(host);
    scene.onHud = (h) => (hud = { ...hud, ...h });
    scene.onPick = (id) => {
      selectedId = id;
      if (id == null) sel = null;
    };
    window.__scene = scene; // debug/testing hook
    closeWS = connectWS({
      onOpen: () => (online = true),
      onClose: () => (online = false),
      onHello: (m) => scene.setLegend(m.biomes),
      onFrame: (f) => {
        scene.update(f);
        refreshSel(f);
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

  function closeInspector() {
    selectedId = null;
    sel = null;
    if (scene) scene.selectedId = null;
  }

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
    {#if hud.zoom > 1}
      <span class="hud-sep">·</span>
      <span class="hud-label">ZOOM</span>
      <span class="hud-val">{hud.zoom}×</span>
    {/if}
  </div>

  {#if sel}
    <div class="inspector">
      <div class="ins-head">
        <span class="ins-dot" style="--c:{sel.col}"></span>
        <span class="ins-title">Agent {sel.id}</span>
        <button class="ins-close" onclick={closeInspector}>×</button>
      </div>
      <div class="ins-row">
        <span class="ins-k">Status</span>
        <span class="ins-v">{STAGE_NAME[sel.st] ?? "?"} · {ACT_NAME[sel.act] ?? "?"}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Stamm</span>
        <span class="ins-v">{sel.tribe ?? "—"}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Energie</span>
        <span class="ins-bar"><span class="ins-fill e" style="width:{Math.min(100, (sel.e / 240) * 100)}%"></span></span>
        <span class="ins-num">{sel.e}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Gesundheit</span>
        <span class="ins-bar"><span class="ins-fill h" style="width:{Math.min(100, sel.hp)}%"></span></span>
        <span class="ins-num">{sel.hp}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Werkzeug</span>
        <span class="ins-v">{TOOL_NAME[sel.tl ?? 0]}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Traglast</span>
        <span class="ins-v">{sel.cg > 0 ? "█".repeat(Math.min(9, sel.cg)) : "leer"}</span>
      </div>
      <div class="ins-row">
        <span class="ins-k">Position</span>
        <span class="ins-v">({sel.x}, {sel.y})</span>
      </div>
    </div>
  {/if}

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
    <span class="lg-sep">·</span>
    <span class="lg-group">
      <span class="glyph" style="--c:#cfd8e3">⚒</span><span class="lg-label">werkzeug</span>
      <span class="glyph" style="--c:#ff8a3c">▲</span><span class="lg-label">feuer</span>
      <span class="glyph" style="--c:#c084fc">◆</span><span class="lg-label">entdeckung</span>
    </span>
    <span class="lg-sep">·</span>
    <span class="lg-hint">rad&nbsp;zoomen · ziehen&nbsp;schwenken · agent&nbsp;anklicken</span>
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
  .glyph {
    color: var(--c);
    font-size: 10px;
    line-height: 1;
  }
  .lg-hint {
    color: #3d4a63;
    text-transform: uppercase;
  }

  .inspector {
    position: absolute;
    top: 34px;
    right: 12px;
    width: 208px;
    background: rgba(5, 8, 13, 0.92);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 8px 10px 10px;
    font-size: 11px;
    backdrop-filter: blur(2px);
  }
  .ins-head {
    display: flex;
    align-items: center;
    gap: 7px;
    padding-bottom: 6px;
    margin-bottom: 6px;
    border-bottom: 1px solid var(--line);
  }
  .ins-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c);
    box-shadow: 0 0 6px var(--c);
  }
  .ins-title {
    color: var(--text);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 11px;
    flex: 1;
  }
  .ins-close {
    background: none;
    border: none;
    color: var(--muted);
    font-size: 14px;
    cursor: pointer;
    padding: 0 2px;
    line-height: 1;
  }
  .ins-close:hover {
    color: var(--text);
  }
  .ins-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2.5px 0;
  }
  .ins-k {
    color: var(--muted);
    width: 72px;
    flex: none;
    text-transform: uppercase;
    font-size: 9.5px;
    letter-spacing: 0.08em;
  }
  .ins-v {
    color: var(--text);
  }
  .ins-bar {
    flex: 1;
    height: 5px;
    background: #101724;
    border-radius: 2px;
    overflow: hidden;
  }
  .ins-fill {
    display: block;
    height: 100%;
    border-radius: 2px;
  }
  .ins-fill.e {
    background: #ffd166;
  }
  .ins-fill.h {
    background: #49d17c;
  }
  .ins-num {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    width: 26px;
    text-align: right;
    flex: none;
    font-size: 10px;
  }

  @media (prefers-reduced-motion: reduce) {
    .scan {
      display: none;
    }
  }
</style>
