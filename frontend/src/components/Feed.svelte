<script>
  let { entries = [] } = $props();
</script>

<div class="feed">
  <div class="feed-head">Chronik</div>
  <div class="feed-body">
    {#if !entries.length}
      <div class="empty">Noch keine Ereignisse — die Welt erwacht…</div>
    {/if}
    {#each entries as e (e.key)}
      <div class="row {e.cls}">
        <span class="t">T{e.tick}</span>
        <span class="ic">{e.icon}</span>
        <span class="tx">{e.text}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .feed {
    display: flex;
    flex-direction: column;
    width: 264px;
    flex: none;
    background: #05070b;
    border: 1px solid var(--line);
    border-radius: 4px;
    overflow: hidden;
  }
  .feed-head {
    padding: 8px 12px;
    font-size: 10.5px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--line);
  }
  .feed-body {
    flex: 1;
    overflow-y: auto;
    padding: 6px 0;
    scrollbar-width: thin;
  }
  .empty {
    padding: 12px;
    color: var(--muted);
    font-size: 11px;
    font-style: italic;
  }
  .row {
    display: flex;
    gap: 8px;
    align-items: baseline;
    padding: 3px 12px;
    font-size: 11.5px;
    line-height: 1.45;
    animation: slide-in 0.25s ease-out;
  }
  .t {
    color: var(--muted);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
    flex: none;
    width: 44px;
  }
  .ic {
    flex: none;
    width: 14px;
    text-align: center;
  }
  .tx {
    color: var(--text);
  }
  .row.birth .ic { color: #49d17c; }
  .row.death .ic { color: #8a93a6; }
  .row.death .tx { color: var(--muted); }
  .row.attack .ic { color: #ff5d6c; }
  .row.build .ic { color: #ffb54d; }
  .row.tech .ic { color: #ffd166; }
  .row.tribe .ic { color: #3fc5f0; }
  .row.event .ic { color: var(--amber, #f0b030); }

  @keyframes slide-in {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: none; }
  }
</style>
