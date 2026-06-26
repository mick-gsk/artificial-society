<script>
  let { stats = {} } = $props();

  // [statsKey, label] — mirrors the aggregate block in serve/frame.py.
  const CARDS = [
    ["tick", "tick"],
    ["population", "population"],
    ["avg_energy", "avg energy"],
    ["tribes", "tribes"],
    ["technologies", "technologies"],
    ["knowledge", "knowledge sites"],
    ["cooperation", "cooperation"],
    ["avg_sick", "avg sick"],
    ["avg_reward", "avg reward"],
    ["n_child", "children"],
    ["n_adult", "adults"],
    ["n_elder", "elders"],
  ];

  function fmt(v) {
    if (v == null) return "–";
    if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(2);
    return v;
  }

  // A non-zero numeric reading is "live signal" — accent it.
  function active(v) {
    return typeof v === "number" && v !== 0;
  }
</script>

<div class="cards">
  {#each CARDS as [key, label] (key)}
    <div class="card" class:active={active(stats[key])}>
      <div class="k">{label}</div>
      <div class="v">{fmt(stats[key])}</div>
    </div>
  {/each}
</div>
