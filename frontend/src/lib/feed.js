// Derive a human-readable event feed from consecutive WS frames.
//
// The frame stream already carries everything needed to narrate the world:
// agents appearing (births) and disappearing (deaths), action transitions
// (fights), the sparse structure list (new construction), aggregate counters
// (technologies discovered, tribes formed) and world events. No backend
// support required — this is pure client-side diffing.

const MAX_PER_TICK = 6; // don't flood the feed when a lot happens at once

const FIGHT_COOLDOWN = 40; // ticks an agent stays quiet in the feed after a fight entry

export function createFeedDiffer() {
  let prevIds = null; // Set of agent ids
  let prevActs = new Map(); // id -> act
  let prevStructs = null; // Set "k@x,y"
  let prevTech = null;
  let prevTribes = null;
  let prevEvents = new Set(); // "kind@x,y" (coarse; events drift slowly)
  const lastFight = new Map(); // id -> tick of the last reported fight

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
    }

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

    const tech = frame.stats?.technologies;
    if (prevTech != null && typeof tech === "number" && tech > prevTech) {
      const d = tech - prevTech;
      out.push({
        tick,
        icon: "✧",
        cls: "tech",
        text: d === 1 ? "Neue Entdeckung" : `${d} neue Entdeckungen`,
      });
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
