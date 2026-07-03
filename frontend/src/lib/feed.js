// Derive a human-readable event feed from consecutive WS frames.
//
// The frame stream already carries everything needed to narrate the world:
// agents appearing (births) and disappearing (deaths), action transitions
// (fights), the sparse structure list (new construction), aggregate counters
// (technologies discovered, tribes formed), world events, goal transitions
// (gl/gp/gm) and sickness transitions (fl bit0). No backend support required
// — this is pure client-side diffing.
//
// Every entry carries `ids: [agentId, ...]` (empty for world events) so a
// consumer (the Inspector's personal Chronik) can filter the stream down to
// one agent's story without any backend change. `createFeedDiffer` optionally
// takes a `getSelectedId` getter: entries that involve the selected agent
// bypass the per-tick caps/cooldowns below (so the personal chronicle never
// silently drops one of "its" events); such bypass-only entries are flagged
// `personalOnly: true` and must NOT be pushed into the global 80-entry feed
// (that would let the caps leak into the world view). An entry that would
// have been emitted anyway (under the caps) is never `personalOnly`, even if
// it also happens to involve the selected agent.
import { PROP_DE } from "./warum.js";

const MAX_PER_TICK = 6; // don't flood the feed when a lot happens at once

const FIGHT_COOLDOWN = 40; // ticks an agent stays quiet in the feed after a fight entry
const TOOL_COOLDOWN = 120; // a lost-and-recrafted tool shouldn't re-announce immediately

const TECH_WINDOW = 60; // batch discovery entries: at most one per window

export function createFeedDiffer(getSelectedId = () => null) {
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
  const prevGoal = new Map(); // id -> gl (goal code) of the current goal, if any
  const prevSick = new Map(); // id -> bool (fl bit0)

  return function diff(frame) {
    const out = [];
    const tick = frame.tick;
    const ids = new Set(frame.agents.map((a) => a.id));

    // Run-restart detection: mirrors world-render.js's `_syncAgents` heuristic
    // (`!connected || (newIds > 3 && newIds*2 >= agents.length)`). On a genuine
    // restart the whole population is swapped out from under us — diffing that
    // naively would fabricate a mass-death + mass-birth wave that never
    // happened. Report one honest world entry instead and reseed all state.
    let reset = false;
    if (prevIds) {
      let newIds = 0;
      for (const a of frame.agents) if (!prevIds.has(a.id)) newIds++;
      reset = newIds > 3 && newIds * 2 >= frame.agents.length;
    }

    if (reset) {
      out.push({
        tick,
        icon: "↺",
        cls: "epoch",
        text: "Neuer Lauf gestartet",
        ids: [],
      });
      lastFight.clear();
      lastTool.clear();
      prevTools.clear();
      prevGoal.clear();
      prevSick.clear();
      firstBlade = false;
      firstFire = false;
      techAccum = 0;
      lastTechTick = null;
      prevSpecial = null;
      prevStructs = null;
      prevTech = typeof frame.stats?.technologies === "number" ? frame.stats.technologies : null;
      prevTribes = typeof frame.stats?.tribes === "number" ? frame.stats.tribes : null;
      prevEvents = new Set((frame.events ?? []).map((e) => e.kind));
      for (const a of frame.agents) {
        prevTools.set(a.id, a.tl ?? 0);
        if (a.gl) prevGoal.set(a.id, a.gl);
        prevSick.set(a.id, !!(a.fl & 1));
      }
      prevIds = ids;
      prevActs = new Map(frame.agents.map((a) => [a.id, a.act]));
      return out;
    }

    const selId = getSelectedId();
    // Push an entry, honouring the per-tick cap unless the entry involves the
    // selected agent — in which case it bypasses the cap but is marked
    // personalOnly so the caller can keep it out of the global feed.
    function pushCapped(entry, counterState) {
      const isSelected = selId != null && entry.ids.includes(selId);
      if (counterState.n < MAX_PER_TICK) {
        counterState.n++;
        out.push(entry);
      } else if (isSelected) {
        out.push({ ...entry, personalOnly: true });
      }
    }

    if (prevIds) {
      const birthCounter = { n: 0 };
      for (const a of frame.agents) {
        if (!prevIds.has(a.id)) {
          const hasParent = a.pa != null && prevIds.has(a.pa);
          pushCapped(
            {
              tick,
              icon: "✦",
              cls: "birth",
              text: hasParent
                ? `Agent ${a.id} geboren (Mutter Agent ${a.pa})`
                : `Agent ${a.id} geboren`,
              ids: hasParent ? [a.id, a.pa] : [a.id],
            },
            birthCounter,
          );
        }
      }
      const deathCounter = { n: 0 };
      for (const id of prevIds) {
        if (!ids.has(id)) {
          pushCapped(
            { tick, icon: "✝", cls: "death", text: `Agent ${id} gestorben`, ids: [id] },
            deathCounter,
          );
        }
      }
      // fights: act transition into 3, with a per-agent cooldown so flickering
      // in/out of combat doesn't flood the feed; multiple fights aggregate.
      // The victim comes from `tg` (last_action_target) when present.
      const fights = [];
      const bypassFights = []; // fights that only fire because the selected agent is involved
      for (const a of frame.agents) {
        if (a.act === 3 && prevActs.get(a.id) !== 3) {
          const last = lastFight.get(a.id);
          const onCooldown = last != null && tick - last < FIGHT_COOLDOWN;
          const involvesSelected = selId != null && (a.id === selId || a.tg === selId);
          if (!onCooldown) {
            lastFight.set(a.id, tick);
            fights.push(a);
          } else if (involvesSelected) {
            bypassFights.push(a);
          }
        }
      }
      if (fights.length === 1) {
        const a = fights[0];
        const ids2 = a.tg != null ? [a.id, a.tg] : [a.id];
        out.push({
          tick,
          icon: "⚔",
          cls: "attack",
          text: `Kampf bei (${a.x}, ${a.y}) — Agent ${a.id}`,
          ids: ids2,
        });
      } else if (fights.length > 1) {
        // aggregate entry still carries every participant's id (+ their
        // victims) so the personal chronicle can pick it up even though the
        // global text is a summary.
        const idSet = new Set();
        for (const a of fights) {
          idSet.add(a.id);
          if (a.tg != null) idSet.add(a.tg);
        }
        out.push({
          tick,
          icon: "⚔",
          cls: "attack",
          text: `${fights.length} Kämpfe entbrannt`,
          ids: [...idSet],
        });
      }
      // fights that were suppressed by the cooldown but involve the selected
      // agent: emit individually as personalOnly so the chronicle stays
      // lückenlos without duplicating anything into the global feed.
      for (const a of bypassFights) {
        const ids2 = a.tg != null ? [a.id, a.tg] : [a.id];
        out.push({
          tick,
          icon: "⚔",
          cls: "attack",
          text: `Kampf bei (${a.x}, ${a.y}) — Agent ${a.id}`,
          ids: ids2,
          personalOnly: true,
        });
      }

      // tools: hands that were empty (or held a dull stone) now hold a blade
      for (const a of frame.agents) {
        const tl = a.tl ?? 0;
        const prev = prevTools.get(a.id) ?? 0;
        if (tl <= prev) continue;
        const last = lastTool.get(a.id);
        const onCooldown = last != null && tick - last < TOOL_COOLDOWN;
        const involvesSelected = selId != null && a.id === selId;
        if (onCooldown && !involvesSelected) continue;
        if (!onCooldown) lastTool.set(a.id, tick);
        const personalOnly = onCooldown && involvesSelected;
        if (tl === 2 && !firstBlade) {
          firstBlade = true;
          out.push({
            tick,
            icon: "⚒",
            cls: "epoch",
            text: `Die erste Klinge — Agent ${a.id} hat ein scharfes Werkzeug erschaffen`,
            ids: [a.id],
            ...(personalOnly ? { personalOnly } : {}),
          });
        } else if (tl === 2) {
          out.push({
            tick,
            icon: "⚒",
            cls: "tool",
            text: `Agent ${a.id} schlägt sich eine scharfe Klinge`,
            ids: [a.id],
            ...(personalOnly ? { personalOnly } : {}),
          });
        } else {
          out.push({
            tick,
            icon: "⚒",
            cls: "tool",
            text: `Agent ${a.id} nimmt einen Stein als Werkzeug`,
            ids: [a.id],
            ...(personalOnly ? { personalOnly } : {}),
          });
        }
      }

      // goal transitions (gl/gp/gm): a fresh goal code appearing is a "start";
      // the previous goal code disappearing (agent still alive) is a "beendet"
      // — the frame schema cannot tell success from abandonment (gl is simply
      // whatever sits on top of the goal stack), so we deliberately use one
      // honest wording rather than inventing a success/failure distinction.
      for (const a of frame.agents) {
        const cur = a.gl || null;
        const prev = prevGoal.get(a.id) ?? null;
        if (cur === prev) continue;
        if (cur && cur !== prev) {
          const name = PROP_DE[cur] ?? `Ziel ${cur}`;
          out.push({
            tick,
            icon: "◇",
            cls: "goal",
            text: `Agent ${a.id} nimmt sich vor: ${name}`,
            ids: [a.id],
          });
        } else if (!cur && prev) {
          const name = PROP_DE[prev] ?? `Ziel ${prev}`;
          out.push({
            tick,
            icon: "◈",
            cls: "goal",
            text: `Agent ${a.id} beendet Vorhaben: ${name}`,
            ids: [a.id],
          });
        }
        if (cur) prevGoal.set(a.id, cur);
        else prevGoal.delete(a.id);
      }

      // sickness transitions (fl bit0)
      for (const a of frame.agents) {
        const sick = !!(a.fl & 1);
        const was = prevSick.get(a.id) ?? false;
        if (sick && !was) {
          out.push({
            tick,
            icon: "☣",
            cls: "sick",
            text: `Agent ${a.id} erkrankt`,
            ids: [a.id],
          });
        } else if (!sick && was) {
          out.push({
            tick,
            icon: "✚",
            cls: "sick",
            text: `Agent ${a.id} genesen`,
            ids: [a.id],
          });
        }
        prevSick.set(a.id, sick);
      }
    } else {
      // first frame after (re)connect: seed goal/sick state without emitting
      // "start" entries for goals already in progress — those weren't a
      // transition we witnessed.
      for (const a of frame.agents) {
        if (a.gl) prevGoal.set(a.id, a.gl);
        prevSick.set(a.id, !!(a.fl & 1));
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
          ids: [],
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
        if (list.length <= 2) for (const p of list) out.push({ tick, icon, cls, text: one(p), ids: [] });
        else out.push({ tick, icon, cls, text: many(list.length), ids: [] });
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
            ids: [],
          });
        }
      }
    }
    prevStructs = structs;

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
          ids: [],
        });
        techAccum = 0;
        lastTechTick = tick;
      }
    }

    const tribes = frame.stats?.tribes;
    if (prevTribes != null && typeof tribes === "number" && tribes > prevTribes) {
      out.push({ tick, icon: "⚑", cls: "tribe", text: "Neuer Stamm gegründet", ids: [] });
    }

    const KIND = { drought: "Dürre", storm: "Sturm", fire: "Feuer", blight: "Fäule" };
    const evs = new Set((frame.events ?? []).map((e) => e.kind));
    for (const k of evs) {
      if (!prevEvents.has(k)) {
        out.push({ tick, icon: "⚠", cls: "event", text: `${KIND[k] ?? k} zieht auf`, ids: [] });
      }
    }

    prevIds = ids;
    prevActs = new Map(frame.agents.map((a) => [a.id, a.act]));
    prevTech = typeof tech === "number" ? tech : prevTech;
    prevTribes = typeof tribes === "number" ? tribes : prevTribes;
    prevEvents = evs;
    return out;
  };
}
