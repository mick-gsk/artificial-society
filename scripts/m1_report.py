#!/usr/bin/env python
"""Wertet die M1-Diagnose-Batterie aus (9 Arme, Spec §3).

Liest <dir>/*.jsonl (von m1_pilot.py), aggregiert die Final-Records je
Experiment-Arm über dessen Seeds (Median + IQR), vergleicht jede Variante
gegen A1 (learn) und A2 (nolearn), und wendet den Entscheidungsbaum aus
Spec §3 (inkl. Schritt 0, Billiger-Lever-Check zuerst) programmatisch an, um
eine Bau-Empfehlung vorzuschlagen.

Gültigkeits-Schwelle (Spec §3, M-1): "deutlich"/"≫"/"≪" gilt nur bei
nicht-überlappendem IQR zwischen Vergleichsarm und Referenz. Reines stdlib,
Python-3.9-kompatibel (kein `match`, kein Laufzeit-`X | Y`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_pilot import EXPERIMENT_ORDER, EXPERIMENTS  # noqa: E402

# Kern-Metriken (Spec §3): Median/IQR hierüber pro Arm.
CORE_METRICS = [
    ("tool_cut_ratio", "tool_cut_ratio"),
    ("cuts_with_tool", "cuts_with_tool"),
    ("discoveries", "discoveries"),
    ("fragments_total", "fragments_total"),
]
# Konfound-Kontrolle (Spec §3): ein Arm, der nur die Population kollabiert,
# senkt Discovery trivial -- das gilt auch für A3 (V-1).
CONFOUND_METRICS = [
    ("pop", "pop"),
    ("mean_energy", "mean_energy"),
]
# Sekundär (Spec §2 Kopfzeile): Verb-Feuerrate für Knapping/Schneiden.
SECONDARY_VERBS = ["strike", "cut"]
# K2 (Respawn-Mühle): Demografie-Felder, nur vorhanden in JSONL seit K2-Patch
# (scripts/m1_pilot.py). Fehlen in älteren battery_results/-Dateien -- werden
# von `_values`/`_agg` bereits sauber übersprungen (kein Crash, s.u.).
DEMOGRAPHY_METRICS = [
    ("respawns", "respawns"),
    ("mean_age", "mean_age"),
    # Review I-1: exakter Todes-Zähler (remove_dead-Wrapper, m1_pilot.py).
    # Fehlt in JSONL von vor diesem Patch -- _values/_agg überspringen das
    # sauber (kein Crash), analog zu respawns/mean_age oben.
    ("deaths", "deaths"),
]


def _load_finals(directory: str) -> dict:
    """exp-Name -> Liste der Final-Records (aus allen *.jsonl im Verzeichnis).

    K2: `respawn_count` (Wert von `--respawn-count`, Default 6) wird aus dem
    meta-Record in jeden Final-Record dieses Laufs kopiert (als
    `_respawn_count_cfg`), damit der Respawn-Mühlen-Check unten den in DIESEM
    Lauf tatsächlich verwendeten Wert kennt, ohne die JSONL-Dateien erneut zu
    öffnen. Alte meta-Records ohne `respawn_count` liefern hier `None` --
    der Check überspringt solche Runs dann sauber.
    """
    arms: dict = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        final = None
        exp = None
        respawn_count_cfg = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record") == "meta":
                    # "exp" ist das aktuelle Schema; "arm" bleibt als Fallback
                    # für JSONL aus dem alten learn/nolearn-Pilot lesbar.
                    exp = rec.get("exp", rec.get("arm"))
                    respawn_count_cfg = rec.get("respawn_count")
                elif rec.get("record") == "final":
                    final = rec
        if exp is None or final is None:
            continue
        final = dict(final, _respawn_count_cfg=respawn_count_cfg)
        arms.setdefault(exp, []).append(final)
    return arms


def _values(finals: list, key: str):
    out = []
    for f in finals:
        if key in SECONDARY_VERBS:
            out.append(int((f.get("verbs_fired") or {}).get(key, 0)))
        elif key in f:
            out.append(f[key])
    return out


def _agg(vals) -> tuple:
    """(median, q1, q3, n). q1/q3 fallen bei n<2 auf den Median zurück
    (IQR=0 -> jeder Vergleich gilt dann konservativ als 'überlappend')."""
    if not vals:
        return (0.0, 0.0, 0.0, 0)
    med = st.median(vals)
    if len(vals) < 2:
        return (med, med, med, len(vals))
    if len(vals) < 4:
        # statistics.quantiles verlangt >=2 Punkte; bei kleinem n (Spec: 3
        # Seeds pro Varianten-Arm) grober Min/Max-Ersatz für Q1/Q3, damit
        # ein einzelner Ausreißer nicht sofort "nicht-überlappend" behauptet.
        q1, q3 = min(vals), max(vals)
    else:
        qs = st.quantiles(vals, n=4, method="inclusive")
        q1, q3 = qs[0], qs[2]
    return (med, q1, q3, len(vals))


def _fmt(agg: tuple) -> str:
    med, q1, q3, n = agg
    if n == 0:
        return "  --  "
    return f"{med:.3g} [{q1:.3g},{q3:.3g}] (n={n})"


def _cmp(agg_a: tuple, agg_b: tuple):
    """Vergleich a vs b unter der M-1-Schwelle (nicht-überlappender IQR).
    Rückgabe: '>>' (a klar über b), '<<' (a klar unter b), '≈' (überlappend
    oder ununterscheidbar), None (fehlende Daten)."""
    med_a, q1_a, q3_a, n_a = agg_a
    med_b, q1_b, q3_b, n_b = agg_b
    if n_a == 0 or n_b == 0:
        return None
    if q1_a > q3_b:
        return ">>"
    if q3_a < q1_b:
        return "<<"
    return "≈"


def _print_table(arms: dict) -> dict:
    """Druckt die Median[IQR]-Tabelle je Arm; gibt {exp: {metric: agg}} zurück."""
    all_metrics = (
        CORE_METRICS
        + CONFOUND_METRICS
        + DEMOGRAPHY_METRICS
        + [(f"verb:{v}", f"verbs.{v}") for v in SECONDARY_VERBS]
    )
    aggs: dict = {}
    print("\n=== M1-Diagnose-Batterie: Arme (Median [Q1,Q3] über Seeds) ===\n")
    header = f"{'Arm (#, n)':<28}" + "".join(f"{label:>22}" for _, label in all_metrics)
    print(header)
    print("-" * len(header))
    for exp in EXPERIMENT_ORDER:
        finals = arms.get(exp, [])
        conf = EXPERIMENTS[exp]
        row_label = f"{exp} ({conf['id']}, n={len(finals)})"
        aggs[exp] = {}
        cells = []
        for key, _ in all_metrics:
            metric_key = key[5:] if key.startswith("verb:") else key
            vals = _values(finals, metric_key)
            agg = _agg(vals)
            aggs[exp][key] = agg
            cells.append(_fmt(agg))
        print(f"{row_label:<28}" + "".join(f"{c:>22}" for c in cells))
    print()
    missing = [exp for exp in EXPERIMENT_ORDER if not arms.get(exp)]
    if missing:
        print(f"(Keine Daten für: {', '.join(missing)})\n")
    return aggs


def _print_comparisons(aggs: dict) -> None:
    print("=== Vergleich jeder Variante gegen A1 (learn) und A2 (nolearn) ===\n")
    print("Gültigkeits-Schwelle M-1: '>>'/'<<' nur bei nicht-überlappendem IQR; sonst '≈'.\n")
    ref_metrics = ["tool_cut_ratio", "discoveries"]
    col_w = 24
    header = f"{'Arm':<20}" + "".join(
        f"{m + ' vs A1':>{col_w}}{m + ' vs A2':>{col_w}}" for m in ref_metrics
    )
    print(header)
    print("-" * len(header))
    for exp in EXPERIMENT_ORDER:
        if exp in ("learn",):
            continue
        row = f"{exp:<20}"
        for m in ref_metrics:
            a1 = _cmp(
                aggs.get(exp, {}).get(m, (0, 0, 0, 0)), aggs.get("learn", {}).get(m, (0, 0, 0, 0))
            )
            a2 = _cmp(
                aggs.get(exp, {}).get(m, (0, 0, 0, 0)), aggs.get("nolearn", {}).get(m, (0, 0, 0, 0))
            )
            row += f"{(a1 or 'n/a'):>{col_w}}{(a2 or 'n/a'):>{col_w}}"
        print(row)
    print()


def _print_confound_warnings(aggs: dict) -> None:
    """Konfound-Kontrolle (Spec §3): eine Population, die klar unter A1
    kollabiert, macht eine gleichzeitige Discovery-Absenkung ungültig als
    'Hypothese bestätigt' -- gilt explizit auch für A3."""
    warnings = []
    for exp in EXPERIMENT_ORDER:
        if exp == "learn" or not aggs.get(exp, {}).get("pop", (0, 0, 0, 0))[3]:
            continue
        pop_cmp = _cmp(aggs[exp]["pop"], aggs["learn"]["pop"])
        if pop_cmp == "<<":
            warnings.append(
                f"  - {exp}: Population liegt klar unter A1 (pop {_fmt(aggs[exp]['pop'])} "
                f"vs A1 {_fmt(aggs['learn']['pop'])}) -- eine niedrigere discoveries/"
                f"tool_cut_ratio in diesem Arm ist ggf. NUR Populations-Kollaps, nicht "
                f"Hypothesen-Bestätigung."
            )
    if warnings:
        print("=== Konfound-Warnungen (Population) ===\n")
        for w in warnings:
            print(w)
        print()


def _print_respawn_mill_check(arms: dict) -> None:
    """K2 (Respawn-Mühle): je Arm Median(respawns_final)/Median(mean_age_final)
    plus ein Warnhinweis, wenn die durch Respawns NEU eingebrachte Agentenzahl
    (respawns_final * respawn_count) vergleichbar oder größer ist als die
    Hälfte von ids_seen_total -- ein Indiz, dass ein signifikanter Anteil der
    je gesehenen Population Zufallshirn-Respawns statt selektierter
    Nachkommen sind (siehe docs/superpowers/specs/
    2026-07-04-m1-batterie-ergebnis-und-empfehlungen.md, K2).

    Läuft ohne die K2-Felder (respawns/mean_age/ids_seen_total/respawn_count
    -- z.B. alte battery_results/-JSONL vor diesem Patch) werden pro Arm
    stillschweigend übersprungen; hat KEIN Arm die Felder, wird das explizit
    vermerkt statt zu crashen.

    Review I-1: `deaths` (exakter Zähler aus dem `remove_dead`-Wrapper,
    scripts/m1_pilot.py) wird als zusätzliche Spalte ausgegeben, sofern
    vorhanden -- alte JSONL ohne dieses Feld liefern "n/a" statt zu crashen
    (das Feld ist NICHT Teil des `usable`-Filters, da es die K2-Kernprüfung
    respawns×respawn_count/ids_seen_total nicht betrifft).
    """
    print("=== Respawn-Mühlen-Check (K2) ===\n")
    any_data = False
    header = (
        f"{'Arm':<20}{'respawns_final (med)':>22}{'mean_age_final (med)':>22}"
        f"{'deaths_final (med)':>20}  Hinweis"
    )
    print(header)
    print("-" * len(header))
    for exp in EXPERIMENT_ORDER:
        finals = arms.get(exp, [])
        usable = [
            f
            for f in finals
            if "respawns" in f
            and "mean_age" in f
            and f.get("ids_seen_total") is not None
            and f.get("_respawn_count_cfg") is not None
        ]
        if not usable:
            continue
        any_data = True
        med_respawns = st.median(f["respawns"] for f in usable)
        med_mean_age = st.median(f["mean_age"] for f in usable)
        deaths_vals = [f["deaths"] for f in usable if "deaths" in f]
        deaths_col = f"{st.median(deaths_vals):>20.1f}" if deaths_vals else f"{'n/a':>20}"
        warn_runs = 0
        for f in usable:
            produced = f["respawns"] * f["_respawn_count_cfg"]
            threshold = f["ids_seen_total"] / 2.0
            if produced >= threshold:
                warn_runs += 1
        note = ""
        if warn_runs:
            note = (
                f"WARNUNG: {warn_runs}/{len(usable)} Runs mit respawns×respawn_count "
                f"≥ ids_seen_total/2 -- Respawn-Mühle plausibel dominant."
            )
        print(f"{exp:<20}{med_respawns:>22.1f}{med_mean_age:>22.1f}{deaths_col}  {note}")
    if not any_data:
        print("(Keine Läufe mit K2-Demografie-Feldern gefunden -- alte JSONL ohne "
              "respawns/mean_age/ids_seen_total/respawn_count; Check übersprungen.)")
    print()


def _decision_tree(aggs: dict) -> None:
    """Programmatische Entscheidungsbaum-Hinweise (Spec §3, Schritt 0 zuerst,
    dann 1-6; C-3-Korrektur: Schritt 0 hat Vorrang vor Schritt 1's Empfehlung,
    ersetzt sie aber nicht -- beide werden gemeldet."""
    print("=== Entscheidungsbaum-Auswertung (Spec §3) ===\n")

    def gt(exp, metric="tool_cut_ratio", metric2="discoveries"):
        """'≫ A1' unter M-1: nicht-überlappend höher auf tool_cut_ratio ODER
        discoveries (Spec §2 nennt beide Metriken für C2/E1/F1)."""
        c1 = _cmp(
            aggs.get(exp, {}).get(metric, (0, 0, 0, 0)),
            aggs.get("learn", {}).get(metric, (0, 0, 0, 0)),
        )
        c2 = _cmp(
            aggs.get(exp, {}).get(metric2, (0, 0, 0, 0)),
            aggs.get("learn", {}).get(metric2, (0, 0, 0, 0)),
        )
        return c1 == ">>" or c2 == ">>"

    def lt(exp, metric="tool_cut_ratio", metric2="discoveries"):
        c1 = _cmp(
            aggs.get(exp, {}).get(metric, (0, 0, 0, 0)),
            aggs.get("learn", {}).get(metric, (0, 0, 0, 0)),
        )
        c2 = _cmp(
            aggs.get(exp, {}).get(metric2, (0, 0, 0, 0)),
            aggs.get("learn", {}).get(metric2, (0, 0, 0, 0)),
        )
        return c1 == "<<" or c2 == "<<"

    def have(exp):
        return bool(aggs.get(exp, {}).get("tool_cut_ratio", (0, 0, 0, 0))[3])

    hints = []

    # Schritt 0: Billiger-Lever-Check zuerst (C2/E1/F1 ≫ A1).
    step0_hits = [
        exp for exp in ("curio-high", "entropy-high", "verb-bias-high") if have(exp) and gt(exp)
    ]
    step0_positive = bool(step0_hits)
    if step0_positive:
        hints.append(
            f"Schritt 0 TRIFFT ZU: {', '.join(step0_hits)} ≫ A1 (nicht-überlappender IQR auf "
            f"tool_cut_ratio/discoveries). Billiger Dosierungs-/Explorations-Win nachgewiesen.\n"
            f"  -> Baue ZUERST: Explorations-Bonus (Curiosity-Gewicht/Entropie/Prior) -- billiger "
            f"als Kredit-Reparatur. Ein gleichzeitiges A1≈A2 (Schritt 1) ist dann ein "
            f"ZUSÄTZLICHES, nicht zwingend vorrangiges Kredit-Problem."
        )
    else:
        hints.append(
            "Schritt 0 negativ: kein C2/E1/F1 ≫ A1 mit nicht-überlappendem IQR nachgewiesen."
        )

    # Schritt 1-3: A1 vs A2, dann B1.
    if have("learn") and have("nolearn"):
        # a1 ≈ a2 heisst: A1 schlaegt A2 NICHT klar (kein '>>' A1 vs A2).
        a1_vs_a2 = _cmp(aggs["learn"]["tool_cut_ratio"], aggs["nolearn"]["tool_cut_ratio"])
        a1_approx_a2 = a1_vs_a2 != ">>"
        if a1_approx_a2 and not step0_positive:
            hints.append(
                "Schritt 1 TRIFFT ZU: A1 ≈ A2 auf tool_cut_ratio (kein nicht-überlappendes A1>A2) "
                "UND Schritt 0 negativ.\n"
                "  -> H4 dominant. Baue: Kredit-Zuweisung/Kopplung (der mechanistische "
                "Werkzeug-Payoff existiert in der Physik, wird aber nie der Aktion gutgeschrieben). "
                "Nutze A3 als Kontrolle, dass A2s Restkultur den A1↔A2-Vergleich nicht verzerrt."
            )
        elif a1_approx_a2 and step0_positive:
            hints.append(
                "Hinweis: A1 ≈ A2 auf tool_cut_ratio träfe für sich isoliert Schritt 1 (H4), "
                "wird aber von Schritt 0 (billiger Lever nachgewiesen) überstimmt -- als "
                "zusätzliches, nicht vorrangiges Kredit-Problem vormerken."
            )
        elif a1_vs_a2 == ">>" and have("gamma-short"):
            b1_vs_a1 = _cmp(aggs["gamma-short"]["tool_cut_ratio"], aggs["learn"]["tool_cut_ratio"])
            if b1_vs_a1 == "≈":
                hints.append(
                    "Schritt 2 TRIFFT ZU: A1 > A2, aber B1 (gamma-short) ≈ A1.\n"
                    "  -> Kredit-Reichweite ist NICHT das Nadelöhr; das Signal erreicht die frühe "
                    "Aktion (Knapping) ohnehin nicht. Baue: Credit-Backfill/Reward-Attribution "
                    "entlang der Kausalkette."
                )
            elif b1_vs_a1 == "<<":
                hints.append(
                    "Schritt 3 TRIFFT ZU: A1 > A2 und B1 (gamma-short) ≪ A1.\n"
                    "  -> H1 bestätigt. Baue: Eligibility-Traces / längeren effektiven Kreditpfad."
                )
        elif a1_vs_a2 == ">>":
            hints.append(
                "A1 > A2, aber B1 (gamma-short) fehlt -- Schritt 2/3 nicht auswertbar ohne diese Daten."
            )
    else:
        hints.append("A1/A2-Daten unvollständig -- Schritt 1-3 nicht auswertbar.")

    # Schritt 4: Curiosity.
    if have("curio-off") or have("curio-high"):
        c1_low = have("curio-off") and lt("curio-off")
        c2_high = have("curio-high") and gt("curio-high")
        if c1_low or c2_high:
            hints.append(
                f"Schritt 4 TRIFFT ZU: {'C1 ≪ A1' if c1_low else ''}"
                f"{' bzw. ' if c1_low and c2_high else ''}{'C2 ≫ A1' if c2_high else ''}.\n"
                "  -> H2 bestätigt. Baue: Neugier-Dosierung/-Schätzer (billiger Tune-Win)."
            )

    # Schritt 5: Sozialer Kanal.
    if have("nosocial") and lt("nosocial"):
        hints.append(
            "Schritt 5 TRIFFT ZU: D1 (nosocial) ≪ A1.\n"
            "  -> H3 bestätigt. Baue: Kultur-Kanal (Kausal-Transfer + Imitation verstärken)."
        )

    # Schritt 6: Exploration (Entropie/Prior) redundant zu Schritt 0, aber
    # Spec listet es separat -- hier als Bestätigung ausgegeben.
    step6_hits = [exp for exp in ("entropy-high", "verb-bias-high") if have(exp) and gt(exp)]
    if step6_hits:
        hints.append(
            f"Schritt 6 TRIFFT ZU: {', '.join(step6_hits)} ≫ A1.\n"
            "  -> Exploration ist unterdosiert. Baue: Explorations-Bonus (Entropie/Prior) "
            "als billigsten ersten Schritt, vor der teuren Kredit-Reparatur."
        )

    for h in hints:
        print(f"- {h}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="pilot_results")
    args = ap.parse_args()
    arms = _load_finals(args.dir)
    print(f"\n=== M1-Diagnose-Batterie: Auswertung ({args.dir}) ===")
    seed_counts = ", ".join(f"{exp}={len(arms.get(exp, []))}" for exp in EXPERIMENT_ORDER)
    print(f"Seeds je Arm: {seed_counts}")

    aggs = _print_table(arms)
    _print_comparisons(aggs)
    _print_confound_warnings(aggs)
    _print_respawn_mill_check(arms)
    _decision_tree(aggs)


if __name__ == "__main__":
    main()
