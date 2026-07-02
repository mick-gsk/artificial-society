<script>
  // Rich inspector panel for one selected agent. Fed by World.svelte with the
  // compact agent (`sel`), the on-demand `detail` blob (may be null until the
  // first detail frame arrives), and the whole `frame` (for global context).
  // All German-labelled; presentation only — the "why" string comes from
  // warum.js, the camera jumps go back up through callbacks.
  import {
    warumLine,
    needWord,
    PROP_DE,
    roleWord,
  } from "../lib/warum.js";

  let {
    sel = null, // compact agent (v2 fields)
    detail = null, // deep blob or null
    frame = null, // last frame (unused for now beyond context)
    onJump = () => {}, // ({x,y}) pan-to-cell | ({agent:id}) select+jump
    onFollowToggle = () => {},
    onClose = () => {},
    following = false,
    chronik = null, // optional slot content — filled by a later task
  } = $props();

  const STAGE_NAME = ["Kind", "Erwachsen", "Ältester"];
  const TOOL_NAME = ["—", "Stein", "scharfe Klinge"];
  const NEED_KEYS = ["hunger", "durst", "kaelte", "krank", "werkzeug", "forschung", "muede"];
  const NEED_LABEL = {
    hunger: "Hunger",
    durst: "Durst",
    kaelte: "Kälte",
    krank: "Krankheit",
    werkzeug: "Werkzeug",
    forschung: "Neugier",
    muede: "Müdigkeit",
  };
  // hormone order mirrors frame.py agent.endocrine.h (8 floats)
  const HORM_LABEL = [
    "Cortisol",
    "Adrenalin",
    "Melatonin",
    "Serotonin",
    "Dopamin",
    "Oxytocin",
    "Entzündung",
    "Stoffwechsel",
  ];
  // need-word → accent colour for the headline
  const NEED_COLOR = {
    Hunger: "#ffd166",
    Durst: "#5ec8ff",
    Kälte: "#8fd3ff",
    Krankheit: "#ff7a9c",
    Werkzeugbedarf: "#c9a86a",
    Neugier: "#b58cff",
    Müdigkeit: "#9b8cff",
    Alltag: "#9fb0c8",
  };

  // --- client-side reward ring buffer for the footer sparkline ----------------
  // Keyed by agent id so switching selection starts a fresh trace.
  let rewBufId = null;
  let rewBuf = $state([]);
  $effect(() => {
    const id = sel?.id ?? null;
    if (id !== rewBufId) {
      rewBufId = id;
      rewBuf = [];
    }
    const r = detail?.rew;
    if (id != null && r != null) {
      // append only on a fresh detail (avoid dupes across ticker re-renders)
      rewBuf = [...rewBuf, r].slice(-60);
    }
  });

  let why = $derived(warumLine(sel));
  let needHl = $derived(needWord(sel?.nd));
  let needCol = $derived(NEED_COLOR[needHl] ?? "#9fb0c8");

  function pct(v, max) {
    return Math.max(0, Math.min(100, (v / max) * 100));
  }
  function fmtTrust(v) {
    return (v >= 0 ? "+" : "") + v.toFixed(1);
  }

  // Sparkline polyline points from the reward buffer (auto-scaled).
  let sparkPts = $derived.by(() => {
    const b = rewBuf;
    if (b.length < 2) return "";
    const lo = Math.min(...b);
    const hi = Math.max(...b);
    const span = hi - lo || 1;
    const W = 96;
    const H = 18;
    return b
      .map((v, i) => {
        const x = (i / (b.length - 1)) * W;
        const y = H - ((v - lo) / span) * H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  });

  // status flags from fl bitfield
  let flags = $derived.by(() => {
    const f = sel?.fl ?? 0;
    const out = [];
    if (f & 1) out.push({ t: "krank", c: "#ff7a9c" });
    if (f & 2) out.push({ t: "schwanger", c: "#ff9ecb" });
    if (f & 4) out.push({ t: "kommuniziert", c: "#9fd0ff" });
    return out;
  });

  let invEntries = $derived(detail?.inv ? Object.entries(detail.inv) : []);
</script>

{#if sel}
  <div class="inspector">
    <div class="ins-head">
      <span class="ins-dot" style="--c:{sel.col}"></span>
      <span class="ins-title">Agent {sel.id}</span>
      <button
        class="ins-follow"
        class:on={following}
        title="Kamera folgen"
        onclick={() => onFollowToggle()}>folgen</button
      >
      <button class="ins-close" title="Schließen" onclick={() => onClose()}>×</button>
    </div>

    <!-- 1 · Warum-Headline -->
    <div class="why">
      <span class="why-need" style="color:{needCol}">{needHl}</span>
      <span class="why-rest">{why.replace(/^[^→]*→\s*/, "→ ")}</span>
    </div>

    <!-- 2 · Ziel (goal object + progress + clickable target cell) -->
    {#if sel.gl}
      <div class="sec">
        <div class="sec-h">Ziel</div>
        <div class="goal-row">
          <span class="goal-name">{PROP_DE[sel.gl] ?? `Ziel ${sel.gl}`}</span>
          {#if sel.gx != null && sel.gy != null}
            <button class="cell-link" onclick={() => onJump({ x: sel.gx, y: sel.gy })}
              >({sel.gx}, {sel.gy})</button
            >
          {/if}
        </div>
        {#if sel.gm}
          <div class="bar">
            <span class="fill goal" style="width:{pct(sel.gp ?? 0, sel.gm)}%"></span>
          </div>
          <span class="bar-num">{sel.gp ?? 0}/{sel.gm}</span>
        {/if}
      </div>
    {/if}

    <!-- 3 · Bedürfnisse -->
    <div class="sec">
      <div class="sec-h">Bedürfnisse</div>
      {#if detail?.needs}
        {#each NEED_KEYS as k}
          <div class="need-row">
            <span class="need-k">{NEED_LABEL[k]}</span>
            <span class="bar sm">
              <span class="fill need" style="width:{pct(detail.needs[k] ?? 0, 1)}%"></span>
            </span>
          </div>
        {/each}
      {:else}
        <div class="hint">{needHl}</div>
      {/if}
    </div>

    <!-- 4 · Vitalwerte -->
    <div class="sec">
      <div class="sec-h">Vitalwerte</div>
      <div class="need-row">
        <span class="need-k">Energie</span>
        <span class="bar sm"><span class="fill e" style="width:{pct(sel.e, 240)}%"></span></span>
        <span class="bar-num">{sel.e}</span>
      </div>
      <div class="need-row">
        <span class="need-k">Gesundheit</span>
        <span class="bar sm"><span class="fill h" style="width:{pct(sel.hp, 100)}%"></span></span>
        <span class="bar-num">{sel.hp}</span>
      </div>
      {#if detail?.hyd != null}
        <div class="need-row">
          <span class="need-k">Hydration</span>
          <span class="bar sm"><span class="fill w" style="width:{pct(detail.hyd, 100)}%"></span></span>
          <span class="bar-num">{Math.round(detail.hyd)}</span>
        </div>
      {/if}
    </div>

    <!-- 5 · Hormone -->
    {#if detail?.horm}
      <div class="sec">
        <div class="sec-h">Hormone</div>
        <div class="horm-grid">
          {#each detail.horm as v, i}
            <div class="horm-cell">
              <span class="horm-k">{HORM_LABEL[i] ?? `H${i}`}</span>
              <span class="bar xs"><span class="fill horm" style="width:{pct(v, 1)}%"></span></span>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- 6 · Inventar -->
    <div class="sec">
      <div class="sec-h">Inventar</div>
      <div class="need-row">
        <span class="need-k">Werkzeug</span>
        <span class="need-v">{TOOL_NAME[sel.tl ?? 0]}</span>
      </div>
      {#if invEntries.length}
        <div class="inv-list">
          {#each invEntries as [m, q]}
            <span class="inv-chip">{m}<b>{q}</b></span>
          {/each}
        </div>
      {:else}
        <div class="hint">leer</div>
      {/if}
    </div>

    <!-- 7 · Soziales -->
    <div class="sec">
      <div class="sec-h">Soziales</div>
      <div class="need-row">
        <span class="need-k">Stamm</span>
        <span class="need-v">{sel.tribe ?? "—"}</span>
      </div>
      {#if detail?.trust?.length}
        <div class="chip-row">
          {#each detail.trust as [id, v]}
            <button class="trust-chip" onclick={() => onJump({ agent: id })}
              >Agent {id} <b class:pos={v >= 0} class:neg={v < 0}>{fmtTrust(v)}</b></button
            >
          {/each}
        </div>
      {/if}
      {#if detail?.tom?.length}
        <div class="chip-row">
          {#each detail.tom as [id, role]}
            <button class="tom-badge" onclick={() => onJump({ agent: id })}
              >Agent {id}: {roleWord(role)}</button
            >
          {/each}
        </div>
      {/if}
    </div>

    <!-- 8 · Status -->
    {#if flags.length}
      <div class="sec">
        <div class="chip-row">
          {#each flags as f}
            <span class="status-badge" style="--c:{f.c}">{f.t}</span>
          {/each}
        </div>
      </div>
    {/if}

    <!-- 9 · Footer -->
    <div class="footer">
      <span class="foot-k">Alter</span><span class="foot-v">{detail?.age ?? "—"}</span>
      <span class="foot-k">Geschl.</span><span class="foot-v"
        >{detail?.sex === "m" ? "m" : detail?.sex === "f" ? "w" : "—"}</span
      >
      <span class="foot-k">Gen</span><span class="foot-v">{detail?.gen ?? "—"}</span>
      {#if detail?.par != null}
        <span class="foot-k">Eltern</span>
        <button class="cell-link" onclick={() => onJump({ agent: detail.par })}>Agent {detail.par}</button>
      {/if}
      <span class="foot-k">Kinder</span><span class="foot-v">{detail?.kids ?? "—"}</span>
    </div>
    {#if sparkPts}
      <div class="spark">
        <span class="foot-k">Belohnung</span>
        <svg viewBox="0 0 96 18" preserveAspectRatio="none" class="spark-svg">
          <polyline points={sparkPts} fill="none" stroke="#5ec8ff" stroke-width="1" />
        </svg>
      </div>
    {/if}

    {#if chronik}
      <div class="sec chronik-slot">{@render chronik()}</div>
    {/if}
  </div>
{/if}

<style>
  .inspector {
    position: absolute;
    top: 34px;
    right: 12px;
    width: 236px;
    max-height: calc(100% - 48px);
    overflow-y: auto;
    background: rgba(5, 8, 13, 0.94);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 8px 10px 10px;
    font-size: 11px;
    backdrop-filter: blur(2px);
    scrollbar-width: thin;
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
  .ins-follow {
    background: none;
    border: 1px solid var(--line);
    color: var(--muted);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 6px;
    border-radius: 3px;
    cursor: pointer;
  }
  .ins-follow:hover {
    color: var(--text);
  }
  .ins-follow.on {
    color: #05070b;
    background: var(--accent);
    border-color: var(--accent);
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

  .why {
    margin: 2px 0 8px;
    font-size: 12px;
    line-height: 1.35;
    color: var(--text);
    font-family: ui-monospace, Menlo, monospace;
  }
  .why-need {
    font-weight: 700;
  }
  .why-rest {
    color: #cdd7e6;
  }

  .sec {
    padding: 6px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }
  .sec-h {
    color: var(--muted);
    text-transform: uppercase;
    font-size: 9px;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
  }
  .hint {
    color: var(--muted);
    font-size: 10px;
  }

  .goal-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    margin-bottom: 3px;
  }
  .goal-name {
    color: var(--text);
  }

  .need-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 1.5px 0;
  }
  .need-k {
    color: var(--muted);
    width: 66px;
    flex: none;
    font-size: 9.5px;
    letter-spacing: 0.04em;
  }
  .need-v {
    color: var(--text);
  }
  .bar {
    flex: 1;
    height: 5px;
    background: #101724;
    border-radius: 2px;
    overflow: hidden;
  }
  .bar.sm {
    height: 5px;
  }
  .bar.xs {
    height: 4px;
  }
  .fill {
    display: block;
    height: 100%;
    border-radius: 2px;
    background: #5ec8ff;
  }
  .fill.e {
    background: #ffd166;
  }
  .fill.h {
    background: #49d17c;
  }
  .fill.w {
    background: #5ec8ff;
  }
  .fill.need {
    background: #d98a5b;
  }
  .fill.goal {
    background: #b58cff;
  }
  .fill.horm {
    background: #7fb3d5;
  }
  .bar-num {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    width: 34px;
    text-align: right;
    flex: none;
    font-size: 10px;
  }

  .horm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 8px;
  }
  .horm-cell {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .horm-k {
    color: var(--muted);
    font-size: 8.5px;
    width: 58px;
    flex: none;
  }

  .inv-list,
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 3px;
  }
  .inv-chip {
    background: #101724;
    border-radius: 3px;
    padding: 1px 5px;
    color: var(--muted);
    font-size: 9.5px;
  }
  .inv-chip b {
    color: var(--text);
    margin-left: 4px;
    font-weight: 600;
  }

  .cell-link,
  .trust-chip,
  .tom-badge {
    background: #101724;
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 1px 6px;
    color: #cdd7e6;
    font-size: 9.5px;
    cursor: pointer;
    font-family: inherit;
  }
  .cell-link:hover,
  .trust-chip:hover,
  .tom-badge:hover {
    border-color: var(--accent);
    color: var(--text);
  }
  .trust-chip b.pos {
    color: #49d17c;
  }
  .trust-chip b.neg {
    color: #ff7a9c;
  }

  .status-badge {
    background: color-mix(in srgb, var(--c) 22%, transparent);
    border: 1px solid var(--c);
    color: var(--c);
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .footer {
    display: grid;
    grid-template-columns: auto 1fr auto 1fr;
    gap: 2px 6px;
    align-items: center;
    padding-top: 6px;
    margin-top: 4px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 9.5px;
  }
  .foot-k {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .foot-v {
    color: var(--text);
  }
  .footer .cell-link {
    justify-self: start;
  }
  .spark {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
  }
  .spark-svg {
    flex: 1;
    height: 18px;
  }
</style>
