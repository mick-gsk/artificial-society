<script>
  import { onMount, onDestroy } from "svelte";
  import { WorldScene } from "../lib/world-render.js";
  import { connectWS } from "../lib/ws.js";
  import { initWarum, warumLine, needWord, actWord } from "../lib/warum.js";
  import Inspector from "./Inspector.svelte";

  let { onFrame, onSelect } = $props();

  let host;
  let scene;
  let ws; // { dispose, send }

  let online = $state(false);
  let hud = $state({ tick: 0, agents: 0, w: 0, h: 0, fps: 0, events: 0, zoom: 1 });
  let sel = $state(null); // compact data of the inspected agent (current frame)
  let selDetail = $state(null); // deep detail blob (kept until a newer one lands)
  let lastFrame = $state(null);
  let following = $state(false);

  // hover tooltip
  let hover = $state(null); // { id, act, need } | null
  let hoverPos = $state({ x: 0, y: 0 });

  let selectedId = null;

  // Send the current inspect registration. Called on select/deselect AND after
  // every hello (initial connect + auto-reconnect) so a reconnect re-subscribes.
  function sendInspect() {
    ws?.send({ type: "inspect", id: selectedId });
  }

  function select(id) {
    selectedId = id;
    following = false;
    if (scene) {
      scene.followId = null;
      scene.selectedId = id; // keep the canvas ring/label in sync (esp. on close)
      if (id == null) scene.setSelectedLabel("");
    }
    if (id == null) {
      sel = null;
      selDetail = null;
      onSelect?.(null);
    } else {
      onSelect?.(id);
    }
    sendInspect();
  }

  function refreshSel(f) {
    if (selectedId == null) {
      if (sel !== null) sel = null;
      scene?.setSelectedLabel("");
      return;
    }
    const a = f.agents.find((x) => x.id === selectedId);
    if (!a) {
      // selected agent vanished (died / left frame)
      selectedId = null;
      sel = null;
      selDetail = null;
      following = false;
      if (scene) {
        scene.selectedId = null;
        scene.followId = null;
        scene.setSelectedLabel("");
      }
      onSelect?.(null);
      sendInspect();
      return;
    }
    sel = a;
    // keep the last detail until a newer one lands; drop to null only on deselect
    selDetail = f.detail?.[String(selectedId)] ?? selDetail;
    scene?.setSelectedLabel(warumLine(a));
  }

  onMount(async () => {
    scene = new WorldScene();
    await scene.init(host);
    scene.onHud = (h) => (hud = { ...hud, ...h });
    scene.onPick = (id) => select(id);
    // scene cleared follow itself (manual drag / followed agent gone) — reflect
    // it in the button state.
    scene.onFollowChange = (id) => (following = id != null);
    scene.onHover = (id) => {
      if (id == null) {
        hover = null;
        return;
      }
      const a = lastFrame?.agents.find((x) => x.id === id);
      hover = a
        ? { id, act: actWord(a.act ?? 0), need: needWord(a.nd) }
        : { id, act: actWord(0), need: needWord(0) };
    };
    // track cursor for tooltip placement
    host.addEventListener("pointermove", (e) => {
      const r = host.getBoundingClientRect();
      hoverPos = { x: e.clientX - r.left, y: e.clientY - r.top };
    });
    window.__scene = scene; // debug/testing hook
    ws = connectWS({
      onOpen: () => (online = true),
      onClose: () => (online = false),
      onHello: (m) => {
        scene.setLegend(m.biomes);
        initWarum(m.behavior);
        sendInspect(); // re-subscribe our current inspect after (re)connect
      },
      onFrame: (f) => {
        lastFrame = f;
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
    select(null);
  }

  function toggleFollow() {
    following = !following;
    if (scene) scene.followId = following ? selectedId : null;
  }

  // Inspector jump: to a world cell, or select + jump to another agent.
  function onJump(target) {
    if (!scene) return;
    if (target.agent != null) {
      select(target.agent);
      scene.selectedId = target.agent;
      scene.panToAgent(target.agent);
    } else if (target.x != null && target.y != null) {
      scene.panToCell(target.x, target.y);
    }
  }

  onDestroy(() => {
    ws?.dispose();
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

  <Inspector
    {sel}
    detail={selDetail}
    frame={lastFrame}
    {following}
    {onJump}
    onFollowToggle={toggleFollow}
    onClose={closeInspector}
  />

  {#if hover && hover.id !== selectedId}
    <div
      class="tooltip"
      style="left:{hoverPos.x + 14}px; top:{hoverPos.y + 14}px"
      aria-hidden="true"
    >
      #{hover.id} · {hover.act} · {hover.need}
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

  .tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(5, 8, 13, 0.95);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 2px 7px;
    font-size: 10.5px;
    color: var(--text);
    font-family: ui-monospace, Menlo, monospace;
    white-space: nowrap;
    z-index: 5;
    text-shadow: 0 0 6px rgba(0, 0, 0, 0.9);
  }

  @media (prefers-reduced-motion: reduce) {
    .scan {
      display: none;
    }
  }
</style>
