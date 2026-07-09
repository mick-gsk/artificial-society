// warum.js — pure German mapping/formatting for the "why" of an agent.
//
// This module is deliberately free of any Pixi/Svelte imports so it can be
// unit-tested in isolation (see warum.test.js). It turns the compact behaviour
// fields the backend ships per agent (nd/gl/gp/gm/gx/gy/tg/act …) into one
// human-readable German line and a few small lookups the UI reuses.
//
// The backend omits every falsy key (frame-schema v2), so `nd`, `gl`, `tg`,
// `gx`, `gy`, `gp`, `gm` may simply be absent on an agent object — every reader
// below treats "absent" as "not pressing / unknown", never as an error.

// Dominant-need code (nd, 0..7) → German word. Code 0 (none) has no word — the
// caller renders "Alltag" for it. Indices mirror serve/frame.py `_NEED_NAMES`.
export const NEED_DE = {
  1: "Hunger",
  2: "Thirst",
  3: "Cold",
  4: "Sickness",
  5: "Tool need",
  6: "Curiosity",
  7: "Fatigue",
};

// Action code (act, 0..5) → German verb phrase. Mirrors serve/frame.py `acts`
// (idle/forage/cooperate/attack/build + the sleeping override at 5).
export const ACT_DE = {
  0: "idles",
  1: "forages",
  2: "cooperates",
  3: "attacks",
  4: "builds",
  5: "sleeps",
};

// Goal code (gl, 1..12) → short German goal noun. The goal code is the 1-based
// index into the material-property vector PROP_DIMS; the default order is baked
// in here and overridden from the hello legend via initWarum() so the mapping
// stays correct even if the backend property order ever shifts.
const PROP_DE_BY_NAME = {
  flammable: "Fuel",
  hardness: "Hard material",
  edibility: "Food",
  toxicity: "Poison",
  heat_emission: "Warmth",
  light_emission: "Light source",
  mass: "Heavy material",
  dryness: "Dry matter",
  sharpness: "Blade",
  solubility: "Solubles",
  conductivity: "Conductor",
  scent: "Scent",
};

// Default PROP_DIMS order (serve/environment/materials.py). Rebuilt from the
// hello legend by initWarum(); this constant keeps warumLine() correct before
// the legend arrives and gives the unit tests a fixed reference.
const DEFAULT_PROP_ORDER = [
  "flammable",
  "hardness",
  "edibility",
  "toxicity",
  "heat_emission",
  "light_emission",
  "mass",
  "dryness",
  "sharpness",
  "solubility",
  "conductivity",
  "scent",
];

// gl (1-based) → German goal noun. Mutable so initWarum() can rebuild it from
// the live legend; seeded from the default order for pre-legend correctness.
export let PROP_DE = {};
function _rebuildPropDe(order) {
  const table = {};
  order.forEach((prop, i) => {
    table[i + 1] = PROP_DE_BY_NAME[prop] ?? prop; // fallback = raw property name
  });
  PROP_DE = table;
}
_rebuildPropDe(DEFAULT_PROP_ORDER);

// Theory-of-mind role → German role noun (used by the Inspector's ToM badges).
export const ROLE_DE = {
  hunter: "Hunter",
  maker: "Maker",
  elder: "Elder",
  scout: "Scout",
  warrior: "Warrior",
  sleeper: "Sleeper",
};

export function roleWord(role) {
  return ROLE_DE[role] ?? "Unknown";
}

// Legends captured from the WS hello message. `acts` and `needs` are kept for
// completeness; `goal_props` drives the gl→German mapping.
let _legends = null;

/**
 * Adopt the hello-frame behaviour legends. Safe to call repeatedly (every
 * reconnect re-sends hello). `goal_props` (the PROP_DIMS order) rebuilds PROP_DE
 * so gl codes map to the right German noun even if the backend order changes.
 */
export function initWarum(legends) {
  _legends = legends ?? null;
  const order = legends?.goal_props;
  _rebuildPropDe(Array.isArray(order) && order.length ? order : DEFAULT_PROP_ORDER);
}

/** Word for a dominant-need code (nd). 0/absent → "Routine". */
export function needWord(nd) {
  return NEED_DE[nd] ?? "Routine";
}

/** Verb phrase for an action code (act). Absent → "idles". */
export function actWord(act) {
  return ACT_DE[act] ?? "idles";
}

/**
 * One human-readable German "why" line for an agent, from its compact fields.
 *
 * Template (bracketed parts appended only when their data is present):
 *   "<Bedürfnis> → <Verb>[ Agent <tg>][ (Ziel: gx,gy)][ – <PROP_DE[gl]> (gp/gm)]"
 *
 * Rules:
 *   - act===5 (sleeping) overrides everything → "Müdigkeit → schläft".
 *   - nd absent/0 → the need word is "Alltag".
 *   - act===3 with tg → "… → greift Agent <tg> an" (verb wraps the target id).
 *   - act===2/3 with tg → append/embed the partner/victim id; a missing tg just
 *     omits that clause (cooperate can legitimately pool without a partner id).
 *   - goal clause needs gl; the (gp/gm) progress is appended only when gm>0.
 *
 * @param {object} a           compact agent (v2 fields; falsy keys absent)
 * @param {Map|object} [_byId] optional id→agent lookup (reserved; not required)
 */
export function warumLine(a, _byId) {
  if (!a) return "";
  const act = a.act ?? 0;

  // Sleeping overrides all other drives — a sleeping agent runs no actions.
  if (act === 5) return "Fatigue → sleeps";

  const need = needWord(a.nd); // 0/absent → "Routine"

  // Verb clause. Attack with a known victim reads as one phrase around the id;
  // a plain attack (no tg) or any other action uses the bare verb.
  let verb;
  if (act === 3 && a.tg != null) {
    verb = `attacks Agent ${a.tg}`;
  } else {
    verb = actWord(act);
    // cooperate with a known partner: name them after the verb.
    if (act === 2 && a.tg != null) verb += ` with Agent ${a.tg}`;
  }

  let line = `${need} → ${verb}`;

  // Target cell of the current goal (where the agent is headed).
  if (a.gx != null && a.gy != null) {
    line += ` (target: ${a.gx},${a.gy})`;
  }

  // Concrete goal object + progress.
  if (a.gl) {
    const goalWord = PROP_DE[a.gl] ?? `goal ${a.gl}`;
    line += ` – ${goalWord}`;
    const gm = a.gm ?? 0;
    if (gm > 0) line += ` (${a.gp ?? 0}/${gm})`;
  }

  return line;
}
