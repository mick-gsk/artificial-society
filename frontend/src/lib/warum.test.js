// Unit tests for warum.js — run with `node --test src/lib/warum.test.js`
// (native Node test runner; no extra deps, matches the project's zero-config
// stance). Pure ESM import of the module under test.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  warumLine,
  needWord,
  actWord,
  roleWord,
  initWarum,
  PROP_DE,
} from "./warum.js";

// gl codes for the default PROP_DIMS order (1-based): edibility=3, sharpness=9,
// heat_emission=5, flammable=1, light_emission=6, hardness=2.
const GL_EDIBILITY = 3;
const GL_SHARPNESS = 9;
const GL_HEAT = 5;

test("brief fixture: hunger foraging toward a food cell", () => {
  const a = { nd: 1, act: 1, gx: 12, gy: 8, gl: GL_EDIBILITY, gp: 3, gm: 20 };
  assert.equal(
    warumLine(a),
    "Hunger → sucht Nahrung (Ziel: 12,8) – Essbares (3/20)",
  );
});

test("sleeping overrides everything", () => {
  // act===5 must ignore nd/gl/gx entirely.
  const a = { nd: 1, act: 5, gl: GL_EDIBILITY, gx: 5, gy: 5, gp: 2, gm: 9, tg: 7 };
  assert.equal(warumLine(a), "Müdigkeit → schläft");
});

test("no need code → Alltag", () => {
  const a = { act: 1 }; // nd absent
  assert.equal(warumLine(a), "Alltag → sucht Nahrung");
  const b = { nd: 0, act: 0 }; // explicit 0
  assert.equal(warumLine(b), "Alltag → wartet");
});

test("attack with a victim id embeds the id in the verb", () => {
  const a = { nd: 4, act: 3, tg: 42 };
  assert.equal(warumLine(a), "Krankheit → greift Agent 42 an");
});

test("attack without a target still reads cleanly", () => {
  const a = { act: 3 }; // no tg
  assert.equal(warumLine(a), "Alltag → greift an");
});

test("cooperate names a known partner", () => {
  const a = { nd: 6, act: 2, tg: 9 };
  assert.equal(warumLine(a), "Neugier → kooperiert mit Agent 9");
});

test("cooperate without a partner (pooling path) omits the id, no error", () => {
  const a = { act: 2 }; // tg absent — legitimate per L1 review
  assert.equal(warumLine(a), "Alltag → kooperiert");
});

test("goal clause omits progress when gm is 0/absent", () => {
  const a = { nd: 5, act: 1, gl: GL_SHARPNESS };
  assert.equal(warumLine(a), "Werkzeugbedarf → sucht Nahrung – Klinge");
});

test("target cell without a goal object", () => {
  const a = { nd: 3, act: 0, gx: 4, gy: 7 };
  assert.equal(warumLine(a), "Kälte → wartet (Ziel: 4,7)");
});

test("cold seeking a heat source with full clause", () => {
  const a = { nd: 3, act: 1, gx: 2, gy: 2, gl: GL_HEAT, gp: 5, gm: 30 };
  assert.equal(
    warumLine(a),
    "Kälte → sucht Nahrung (Ziel: 2,2) – Wärmequelle (5/30)",
  );
});

test("empty / null agent is a safe empty string", () => {
  assert.equal(warumLine(null), "");
  assert.equal(warumLine(undefined), "");
});

test("needWord and actWord lookups", () => {
  assert.equal(needWord(1), "Hunger");
  assert.equal(needWord(7), "Müdigkeit");
  assert.equal(needWord(0), "Alltag");
  assert.equal(needWord(undefined), "Alltag");
  assert.equal(actWord(3), "greift an");
  assert.equal(actWord(undefined), "wartet");
});

test("roleWord maps ToM roles, unknown → Unbekannt", () => {
  assert.equal(roleWord("hunter"), "Jäger");
  assert.equal(roleWord("maker"), "Macher");
  assert.equal(roleWord("elder"), "Ältester");
  assert.equal(roleWord("scout"), "Späher");
  assert.equal(roleWord("warrior"), "Krieger");
  assert.equal(roleWord("sleeper"), "Schläfer");
  assert.equal(roleWord("wat"), "Unbekannt");
  assert.equal(roleWord(undefined), "Unbekannt");
});

test("initWarum rebuilds PROP_DE from a reordered legend", () => {
  // Reversed order: edibility now lands at index 9 (gl=10) etc. Verify the
  // mapping follows the legend, not a hardcoded position.
  const reordered = {
    goal_props: [
      "scent",
      "conductivity",
      "solubility",
      "sharpness",
      "dryness",
      "mass",
      "light_emission",
      "heat_emission",
      "toxicity",
      "edibility",
      "hardness",
      "flammable",
    ],
  };
  initWarum(reordered);
  assert.equal(PROP_DE[10], "Essbares"); // edibility moved to gl=10
  assert.equal(PROP_DE[4], "Klinge"); // sharpness moved to gl=4
  // restore default order for any later test isolation
  initWarum({
    goal_props: [
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
    ],
  });
  assert.equal(PROP_DE[3], "Essbares");
});

test("initWarum with unknown property falls back to the raw name", () => {
  initWarum({ goal_props: ["edibility", "mystery_prop"] });
  assert.equal(PROP_DE[2], "mystery_prop");
  // restore default
  initWarum(null);
  assert.equal(PROP_DE[3], "Essbares");
});
