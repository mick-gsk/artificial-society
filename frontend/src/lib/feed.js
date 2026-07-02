// Derive a human-readable event feed from consecutive WS frames.
//
// The frame stream already carries everything needed to narrate the world:
// agents appearing (births) and disappearing (deaths), action transitions
// (fights), the sparse structure list (new construction), aggregate counters
// (technologies discovered, tribes formed) and world events. No backend
// support required — this is pure client-side diffing.

const MAX_PER_TICK = 6; // don't flood the feed when a lot happens at once

const FIGHT_COOLDOWN = 40; // ticks an agent stays quiet in the feed after a fight entry
const TOOL_COOLDOWN = 120; // a lost-and-recrafted tool shouldn't re-announce immediately
const TECH_WINDOW = 60; // batch discovery entries: at most one per window

export function createFeedDiffer() {
  let prevIds = null; // Set of agent ids
  let prevActs = new Map(); // id -> act
  let prevStructs = null; // Set "k@x,y"
  let prevSpecial = null; // Set "k@x,y" of shard/wonder/fire ground items
  let prevTech = null;
  let prevTribes = null;
  let prevEvents = new Set(); // "kind@x,y" (coarse; events drift slowly)
  const lastFight = new Map(); // id -> tick of the last reported fight
  const prevTools = new Map(); // id -> tl (0 none, 1 blunt, 2 sharp)
  const lastTool = new Map(); // id -> tick of the last tool entry
  let firstBlade = false; // the run's very first sharp blade — an epoch
  let firstFire = false;
  let techAccum = 0; // discoveries waiting for the next batched feed entry
  let lastTechTick = null;

  return function diff(frame) {
    const out = [];
    const tick = frame.tick;
    const ids = new Set(frame.agents.map((a) => a.id));

    if (prevIds) {
      let n = 0;
      for (const a of frame.agents) {
        if (!prevIds.has(a.id) && n < MAX_PER_TICK) {
          out.push({ tick, icon: "✦", cls: "birth", text: `Agent ${a.id} geboren` });
          n++;
        }
      }
      n = 0;
      for (const id of prevIds) {
        if (!ids.has(id) && n < MAX_PER_TICK) {
          out.push({ tick, icon: "✝", cls: "death", text: `Agent ${id} gestorben` });
          n++;
        }
      }
      // fights: act transition into 3, with a per-agent cooldown so flickering
      // in/out of combat doesn't flood the feed; multiple fights aggregate.
      const fights = [];
      for (const a of frame.agents) {
        if (a.act === 3 && prevActs.get(a.id) !== 3) {
          const last = lastFight.get(a.id);
          if (last == null || tick - last >= FIGHT_COOLDOWN) {
            lastFight.set(a.id, tick);
            fights.push(a);
          }
        }
      }
      if (fights.length === 1) {
        const a = fights[0];
        out.push({
          tick,
          icon: "⚔",
          cls: "attack",
          text: `Kampf bei (${a.x}, ${a.y}) — Agent ${a.id}`,
        });
      } else if (fights.length > 1) {
        out.push({ tick, icon: "⚔", cls: "attack", text: `${fights.length} Kämpfe entbrannt` });
      }

      // tools: hands that were empty (or held a dull stone) now hold a blade
      for (const a of frame.agents) {
        const tl = a.tl ?? 0;
        const prev = prevTools.get(a.id) ?? 0;
        if (tl <= prev) continue;
        const last = lastTool.get(a.id);
        if (last != null && tick - last < TOOL_COOLDOWN) continue;
        lastTool.set(a.id, tick);
        if (tl === 2 && !firstBlade) {
          firstBlade = true;
          out.push({
            tick,
            icon: "⚒",
            cls: "epoch",
            text: `Die erste Klinge — Agent ${a.id} hat ein scharfes Werkzeug erschaffen`,
          });
        } else if (tl === 2) {
          out.push({
            tick,
            icon: "⚒",
            cls: "tool",
            text: `Agent ${a.id} schlägt sich eine scharfe Klinge`,
          });
        } else {
          out.push({
            tick,
            icon: "⚒",
            cls: "tool",
            text: `Agent ${a.id} nimmt einen Stein als Werkzeug`,
          });
        }
      }
    }
    for (const a of frame.agents) prevTools.set(a.id, a.tl ?? 0);

    // ground materials: knapped flakes, discovered matter and fire appearing
    const special = new Set();
    const items = frame.items ?? [];
    for (let i = 0; i < items.length; i += 3) {
      if (items[i] >= 9) special.add(`${items[i]}@${items[i + 1]},${items[i + 2]}`);
    }
    if (prevSpecial) {
      const fresh = { 9: [], 10: [], 11: [] };
      for (const sk of special) {
        if (!prevSpecial.has(sk)) {
          const [k, pos] = sk.split("@");
          fresh[k]?.push(pos);
        }
      }
      if (fresh[11].length && !firstFire) {
        firstFire = true;
        out.push({
          tick,
          icon: "▲",
          cls: "epoch",
          text: `Das erste Feuer der Welt brennt bei (${fresh[11][0]})`,
        });
        fresh[11].shift();
      }
      const SPECIAL_TEXT = {
        11: [(p) => `Feuer brennt bei (${p})`, (n) => `${n} Feuer brennen`],
        10: [(p) => `Steinschlag — scharfe Splitter bei (${p})`, (n) => `${n} Zellen mit frischen Splittern`],
        9: [(p) => `Unbekanntes Material liegt bei (${p})`, (n) => `${n} unbekannte Materialien aufgetaucht`],
      };
      const SPECIAL_STYLE = { 11: ["▲", "fire"], 10: ["⚒", "tool"], 9: ["◆", "wonder"] };
      for (const k of [11, 10, 9]) {
        const list = fresh[k];
        if (!list.length) continue;
        const [icon, cls] = SPECIAL_STYLE[k];
        const [one, many] = SPECIAL_TEXT[k];
        if (list.length <= 2) for (const p of list) out.push({ tick, icon, cls, text: one(p) });
        else out.push({ tick, icon, cls, text: many(list.length) });
      }
    }
    prevSpecial = special;

    const structs = new Set((frame.structures ?? []).map((s) => `${s.k}@${s.x},${s.y}`));
    if (prevStructs) {
      const NAME = { camp: "Camp", farm: "Farm", well: "Brunnen" };
      for (const s of structs) {
        if (!prevStructs.has(s)) {
          const [k, pos] = s.split("@");
          out.push({
            tick,
            icon: "⌂",
            cls: "build",
            text: `${NAME[k] ?? k} errichtet bei (${pos})`,
          });
        }
      }
    }

    // discoveries tick up almost continuously once invention gets going — batch
    // them into one entry per window instead of flooding the feed
    const tech = frame.stats?.technologies;
    if (prevTech != null && typeof tech === "number" && tech > prevTech) {
      techAccum += tech - prevTech;
      if (lastTechTick == null || tick - lastTechTick >= TECH_WINDOW) {
        out.push({
          tick,
          icon: "✧",
          cls: "tech",
          text: techAccum === 1 ? "Neue Entdeckung" : `${techAccum} neue Entdeckungen`,
        });
        techAccum = 0;
        lastTechTick = tick;
      }
    }

    const tribes = frame.stats?.tribes;
    if (prevTribes != null && typeof tribes === "number" && tribes > prevTribes) {
      out.push({ tick, icon: "⚑", cls: "tribe", text: "Neuer Stamm gegründet" });
    }

    const KIND = { drought: "Dürre", storm: "Sturm", fire: "Feuer", blight: "Fäule" };
    const evs = new Set((frame.events ?? []).map((e) => e.kind));
    for (const k of evs) {
      if (!prevEvents.has(k)) {
        out.push({ tick, icon: "⚠", cls: "event", text: `${KIND[k] ?? k} zieht auf` });
      }
    }

    prevIds = ids;
    prevActs = new Map(frame.agents.map((a) => [a.id, a.act]));
    prevStructs = structs;
    prevTech = typeof tech === "number" ? tech : prevTech;
    prevTribes = typeof tribes === "number" ? tribes : prevTribes;
    prevEvents = evs;
    return out;
  };
}
