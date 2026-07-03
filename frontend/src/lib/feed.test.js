// Unit tests for feed.js — run with `node --test src/lib/feed.test.js`
// (native Node test runner, matches warum.test.js's zero-config stance).

import { test } from "node:test";
import assert from "node:assert/strict";

import { createFeedDiffer } from "./feed.js";

// Minimal frame builder: only the fields the differ reads. `agents` is an
// array of partial agent objects (v2 compact fields); everything else
// defaults to "nothing happened" so tests can focus on one axis at a time.
function frame(tick, agents, extra = {}) {
  return {
    tick,
    agents,
    items: [],
    structures: [],
    events: [],
    stats: {},
    ...extra,
  };
}

test("births and deaths carry structured ids", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0 }])); // baseline, no entries
  const out = diff(frame(1, [{ id: 1, x: 0, y: 0 }, { id: 2, x: 1, y: 1 }]));
  assert.equal(out.length, 1);
  assert.equal(out[0].cls, "birth");
  assert.deepEqual(out[0].ids, [2]);

  const out2 = diff(frame(2, [{ id: 2, x: 1, y: 1 }])); // agent 1 vanished
  assert.equal(out2.length, 1);
  assert.equal(out2[0].cls, "death");
  assert.deepEqual(out2[0].ids, [1]);
});

test("birth entry includes the mother's id via pa when she's alive", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0 }]));
  const out = diff(frame(1, [{ id: 1, x: 0, y: 0 }, { id: 2, x: 1, y: 1, pa: 1 }]));
  assert.equal(out.length, 1);
  assert.deepEqual(out[0].ids, [2, 1]);
  assert.match(out[0].text, /Mutter Agent 1/);
});

test("fight with a target (tg) produces one entry with BOTH attacker and victim ids", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0, act: 0 }, { id: 2, x: 0, y: 0, act: 0 }]));
  const out = diff(
    frame(1, [
      { id: 1, x: 3, y: 4, act: 3, tg: 2 },
      { id: 2, x: 3, y: 4, act: 0 },
    ]),
  );
  const fight = out.find((e) => e.cls === "attack");
  assert.ok(fight, "expected an attack entry");
  assert.deepEqual(fight.ids.sort(), [1, 2]);
  assert.match(fight.text, /Agent 1/);
});

test("fight without a target only carries the attacker's id", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0, act: 0 }]));
  const out = diff(frame(1, [{ id: 1, x: 0, y: 0, act: 3 }]));
  const fight = out.find((e) => e.cls === "attack");
  assert.ok(fight);
  assert.deepEqual(fight.ids, [1]);
});

test("goal start and end (gl/gp/gm transitions), honest single wording for end", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0 }])); // baseline
  const startOut = diff(frame(1, [{ id: 1, x: 0, y: 0, gl: 3, gp: 0, gm: 20 }]));
  const start = startOut.find((e) => e.cls === "goal");
  assert.ok(start, "expected a goal-start entry");
  assert.deepEqual(start.ids, [1]);
  assert.match(start.text, /nimmt sich vor/);
  assert.match(start.text, /Essbares/); // gl=3 -> edibility in default PROP_DE order

  // goal disappears (agent alive) — one honest "beendet" wording, no
  // success/failure claim since the data can't distinguish them.
  const endOut = diff(frame(2, [{ id: 1, x: 0, y: 0 }]));
  const end = endOut.find((e) => e.cls === "goal");
  assert.ok(end, "expected a goal-end entry");
  assert.deepEqual(end.ids, [1]);
  assert.match(end.text, /beendet/);
});

test("goal switching directly (gl changes without going through absent) fires only a start for the new goal", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0, gl: 3 }])); // baseline already has a goal
  const out = diff(frame(1, [{ id: 1, x: 0, y: 0, gl: 9 }])); // switched to gl=9 directly
  const goalEntries = out.filter((e) => e.cls === "goal");
  assert.equal(goalEntries.length, 1);
  assert.match(goalEntries[0].text, /nimmt sich vor/);
});

test("sickness transitions: erkrankt / genesen", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0, fl: 0 }]));
  const sickOut = diff(frame(1, [{ id: 1, x: 0, y: 0, fl: 1 }]));
  const sick = sickOut.find((e) => e.cls === "sick");
  assert.ok(sick);
  assert.deepEqual(sick.ids, [1]);
  assert.match(sick.text, /erkrankt/);

  const healOut = diff(frame(2, [{ id: 1, x: 0, y: 0, fl: 0 }]));
  const heal = healOut.find((e) => e.cls === "sick");
  assert.ok(heal);
  assert.deepEqual(heal.ids, [1]);
  assert.match(heal.text, /genesen/);
});

test("sickness flag bit1/bit2 (pregnant/communicating) do not trigger bit0 sick transitions", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [{ id: 1, x: 0, y: 0, fl: 0 }]));
  const out = diff(frame(1, [{ id: 1, x: 0, y: 0, fl: 6 }])); // bit1+bit2, not bit0
  const sick = out.find((e) => e.cls === "sick");
  assert.equal(sick, undefined);
});

test("reset (disjoint population swap) emits exactly one world entry, no mass birth/death", () => {
  const diff = createFeedDiffer();
  const before = Array.from({ length: 10 }, (_, i) => ({ id: i, x: 0, y: 0 }));
  diff(frame(0, before));
  // entirely new population (a run restart), same size
  const after = Array.from({ length: 10 }, (_, i) => ({ id: i + 100, x: 0, y: 0 }));
  const out = diff(frame(1, after));
  assert.equal(out.length, 1);
  assert.equal(out[0].cls, "epoch");
  assert.match(out[0].text, /Neuer Lauf gestartet/);
  assert.deepEqual(out[0].ids, []);

  // the frame right after a reset should diff normally against the new
  // population baseline (no leftover birth spam from the swapped-in ids)
  const out2 = diff(frame(2, after));
  assert.equal(out2.length, 0);
});

test("small population churn (a few real deaths) is NOT mistaken for a reset", () => {
  const diff = createFeedDiffer();
  const before = Array.from({ length: 20 }, (_, i) => ({ id: i, x: 0, y: 0 }));
  diff(frame(0, before));
  // 2 deaths, 1 birth — well under the reset threshold
  const after = before.slice(2).concat([{ id: 999, x: 0, y: 0 }]);
  const out = diff(frame(1, after));
  const deaths = out.filter((e) => e.cls === "death");
  const births = out.filter((e) => e.cls === "birth");
  assert.equal(deaths.length, 2);
  assert.equal(births.length, 1);
  assert.ok(!out.some((e) => e.cls === "epoch" && /Neuer Lauf/.test(e.text)));
});

test("personalOnly bypass: births beyond the per-tick cap for a non-selected agent are dropped, but the selected agent's birth still appears (flagged personalOnly)", () => {
  let selected = 106;
  const diff = createFeedDiffer(() => selected);
  // Large surviving base population so 8 new ids in one tick reads as a busy
  // baby boom, not a population-swap reset (reset needs newIds*2 >= total).
  const before = Array.from({ length: 40 }, (_, i) => ({ id: i, x: 0, y: 0 }));
  diff(frame(0, before));
  // 8 new agents in one tick (MAX_PER_TICK=6); id 106 is the 7th, beyond the cap
  const news = Array.from({ length: 8 }, (_, i) => ({ id: 100 + i, x: 0, y: 0 }));
  const out = diff(frame(1, [...before, ...news]));
  const births = out.filter((e) => e.cls === "birth");
  // 6 capped + 1 bypass for the selected agent = 7 (the selected one is the
  // 7th new agent, index 6, so it would have been dropped without bypass)
  assert.equal(births.length, 7);
  const mine = births.find((e) => e.ids.includes(106));
  assert.ok(mine, "the selected agent's birth must be present");
  assert.equal(mine.personalOnly, true);
  // the ones under the cap must NOT be personalOnly
  const cappedOnes = births.filter((e) => e.ids[0] !== 106);
  assert.ok(cappedOnes.every((e) => !e.personalOnly));
});

test("personalOnly bypass: a fight suppressed by cooldown still surfaces once for the selected agent", () => {
  let selected = 1;
  const diff = createFeedDiffer(() => selected);
  diff(frame(0, [{ id: 1, x: 0, y: 0, act: 0 }]));
  diff(frame(1, [{ id: 1, x: 0, y: 0, act: 3 }])); // first fight, sets cooldown
  diff(frame(2, [{ id: 1, x: 0, y: 0, act: 0 }])); // drop out of combat
  // re-enter combat well inside the 40-tick cooldown window
  const out = diff(frame(5, [{ id: 1, x: 0, y: 0, act: 3 }]));
  const fight = out.find((e) => e.cls === "attack");
  assert.ok(fight, "the selected agent's cooldown-suppressed fight must still surface");
  assert.equal(fight.personalOnly, true);
  assert.deepEqual(fight.ids, [1]);
});

test("without a selection, cooldown-suppressed fights stay suppressed (no bypass leak)", () => {
  const diff = createFeedDiffer(); // no selection getter override — defaults to null
  diff(frame(0, [{ id: 1, x: 0, y: 0, act: 0 }]));
  diff(frame(1, [{ id: 1, x: 0, y: 0, act: 3 }]));
  diff(frame(2, [{ id: 1, x: 0, y: 0, act: 0 }]));
  const out = diff(frame(5, [{ id: 1, x: 0, y: 0, act: 3 }]));
  const fight = out.find((e) => e.cls === "attack");
  assert.equal(fight, undefined);
});

test("world events (structures, tech, tribes, weather) always carry an empty ids array", () => {
  const diff = createFeedDiffer();
  diff(frame(0, [], { structures: [], stats: { technologies: 0, tribes: 0 } }));
  const out = diff(
    frame(1, [], {
      structures: [{ k: "camp", x: 2, y: 2 }],
      stats: { technologies: 1, tribes: 1 },
      events: [{ kind: "storm", x: 1, y: 1 }],
    }),
  );
  assert.ok(out.length > 0);
  for (const e of out) assert.deepEqual(e.ids, []);
});
