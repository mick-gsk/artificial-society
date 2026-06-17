"""
Emergent Visual System
------------------------
Das Kernproblem: Wie visualisiert man etwas, das noch keinen Namen hat?

Loesung: Der Materialvektor IST die visuelle Beschreibung.
Jede Dimension mapt auf einen visuellen Parameter:

  flammable     → Rotanteil im HSL-Farbmodell
  edibility     → Gruenanteil
  toxicity      → Blau/Lila-Verschiebung + Puls
  heat_emission → Glow-Radius (orange Aura)
  light_emission→ Helligkeit + weisser Kern
  mass          → Opazitaet + Groesse
  hardness      → Form (hart=eckig, weich=rund)
  sharpness     → spitze Ecken am Symbol
  scent         → Pulsfrequenz (volatile = schneller Puls)
  conductivity  → metallischer Glaenzeffekt
  dryness       → Saettigung (trocken=gedaempft, feucht=saettig)
  solubility    → Transparenz

Ein unbekanntes Material mat_0042 bekommt damit automatisch:
  - eine konsistente Farbe (immer gleich fuer denselben Vektor)
  - eine passende Form (hartes Material = Quadrat, weiches = Kreis)
  - einen passenden Glyph (heiss = \u2600, essbar = \u25cf, giftig = \u2620)
  - eine Aura (gluehend = orangener Glow)
Ohne dass jemand mat_0042 je definiert hat.
"""

import math
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from artificial_society.environment.materials import IDX, N_PROPS, PROP_DIMS


# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------
Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]  # mit Alpha


# ---------------------------------------------------------------------------
# HSL -> RGB Hilfsfunktionen
# ---------------------------------------------------------------------------
def _hsl_to_rgb(h: float, s: float, l: float) -> Color:
    """h,s,l alle in [0..1]. Gibt (R,G,B) 0..255 zurueck."""
    if s == 0:
        v = int(l * 255)
        return (v, v, v)
    def f(n):
        k = (n + h * 12) % 12
        a = s * min(l, 1 - l)
        return l - a * max(-1, min(k - 3, 9 - k, 1))
    return (
        int(max(0, min(255, f(0) * 255))),
        int(max(0, min(255, f(8) * 255))),
        int(max(0, min(255, f(4) * 255))),
    )


def _clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# VectorVisual: visuelle Parameter fuer ein Material
# ---------------------------------------------------------------------------
@dataclass
class VectorVisual:
    color:       Color        # Hauptfarbe (R,G,B)
    alpha:       int          # Transparenz 0..255
    radius:      float        # Groesse in Pixeln (relativ zu cell_px)
    shape:       str          # 'circle','square','triangle','diamond'
    glyph:       str          # Unicode-Zeichen
    glyph_color: Color        # Farbe des Glyphs
    aura_radius: float        # Glow-Radius in Pixeln (0 = kein Glow)
    aura_color:  ColorA       # Glow-Farbe mit Alpha
    pulse_freq:  float        # Pulse pro Sekunde (0 = kein Puls)
    outline:     bool         # Umriss zeichnen?
    outline_color: Color


GLYPH_MAP = {
    # Eigenschaft       Schwellwert   Glyph
    'heat_emission':    (0.5,  '\u2600'),   # Sonne
    'light_emission':   (0.4,  '\u2605'),   # Stern
    'toxicity':         (0.4,  '\u26a0'),   # Warnung
    'edibility':        (0.5,  '\u25cf'),   # Kreis gefuellt (Nahrung)
    'sharpness':        (0.6,  '\u25b2'),   # Dreieck (Waffe)
    'flammable':        (0.7,  '\u25b6'),   # Play (Feuer)
    'conductivity':     (0.5,  '\u26a1'),   # Blitz (Leiter)
    'scent':            (0.6,  '\u223f'),   # Welle (Duft)
}


def vector_to_visual(vec: np.ndarray, cell_px: int = 24) -> VectorVisual:
    """
    Kernfunktion: 12D-Vektor -> VectorVisual.
    Deterministisch -- gleicher Vektor = immer gleiche Visuals.
    """
    v = vec  # shorthand

    # --- Farbe: HSL aus dominanten Dimensionen ---
    # Hue:
    #   edibility (gruen) = 0.33
    #   heat (rot)        = 0.0
    #   toxic (lila)      = 0.78
    #   conductor (blau)  = 0.6
    #   flammable (orange)= 0.08
    hue = (
        float(v[IDX['edibility']])   * 0.33 +
        float(v[IDX['heat_emission']]) * 0.0 +
        float(v[IDX['toxicity']])    * 0.78 +
        float(v[IDX['conductivity']]) * 0.6 +
        float(v[IDX['flammable']])   * 0.08
    )
    total_w = (
        float(v[IDX['edibility']]) +
        float(v[IDX['heat_emission']]) +
        float(v[IDX['toxicity']]) +
        float(v[IDX['conductivity']]) +
        float(v[IDX['flammable']]) + 1e-6
    )
    hue = (hue / total_w) % 1.0

    # Saturation: feucht = saettig, trocken = gedaempft
    sat = 0.4 + float(v[IDX['dryness']]) * (-0.2) + float(v[IDX['scent']]) * 0.3
    sat = max(0.1, min(1.0, sat))

    # Lightness: leichte Dinge heller, schwere dunkler
    lig = 0.35 + (1.0 - float(v[IDX['mass']])) * 0.3 + float(v[IDX['light_emission']]) * 0.25
    lig = max(0.15, min(0.85, lig))

    color = _hsl_to_rgb(hue, sat, lig)

    # --- Alpha: Opazitaet aus mass + solubility ---
    alpha = int(255 * (0.4 + float(v[IDX['mass']]) * 0.4 +
                       (1.0 - float(v[IDX['solubility']])) * 0.2))
    alpha = _clamp(alpha, 60, 255)

    # --- Groesse: mass ---
    radius = cell_px * (0.2 + float(v[IDX['mass']]) * 0.3)

    # --- Form: hardness + sharpness ---
    hardness  = float(v[IDX['hardness']])
    sharpness = float(v[IDX['sharpness']])
    if sharpness > 0.6:
        shape = 'triangle'  # scharf = spitz
    elif hardness > 0.7:
        shape = 'square'    # hart = eckig
    elif hardness < 0.2:
        shape = 'circle'    # weich = rund
    else:
        shape = 'diamond'

    # --- Glyph: dominante Eigenschaft ---
    glyph = '\u00b7'  # default: Punkt
    best_val = 0.0
    for prop, (thresh, sym) in GLYPH_MAP.items():
        val = float(v[IDX[prop]])
        if val > thresh and val > best_val:
            best_val = val
            glyph    = sym

    # Glyph-Farbe: Komplement zur Hauptfarbe
    glyph_color = (
        _clamp(255 - color[0]),
        _clamp(255 - color[1]),
        _clamp(255 - color[2]),
    )

    # --- Aura: Glow aus heat + light ---
    heat  = float(v[IDX['heat_emission']])
    light = float(v[IDX['light_emission']])
    aura_r = (heat * 12 + light * 8) * (cell_px / 24)
    if heat > 0.3:
        aura_color = (255, int(120 + heat * 80), 40, int(heat * 80))
    elif light > 0.3:
        aura_color = (255, 255, int(180 + light * 75), int(light * 70))
    else:
        aura_color = (0, 0, 0, 0)

    # --- Puls: scent (volatile = schneller Puls) ---
    pulse_freq = float(v[IDX['scent']]) * 2.0  # 0..2 Hz

    # --- Outline ---
    outline = hardness > 0.5
    outline_color = (
        _clamp(color[0] - 40),
        _clamp(color[1] - 40),
        _clamp(color[2] - 40),
    )

    return VectorVisual(
        color        = color,
        alpha        = alpha,
        radius       = radius,
        shape        = shape,
        glyph        = glyph,
        glyph_color  = glyph_color,
        aura_radius  = aura_r,
        aura_color   = aura_color,
        pulse_freq   = pulse_freq,
        outline      = outline,
        outline_color= outline_color,
    )


# ---------------------------------------------------------------------------
# CompositeObjectVisual: geformte Objekte (geometry.py)
# ---------------------------------------------------------------------------
SHAPE_DRAW_PARAMS = {
    # shape       outline_fn    fill_fn
    'hollow':    'circle_outline',
    'flat':      'rect_fill',
    'sealed':    'circle_fill',
    'layered':   'rect_layered',
    'pointed':   'triangle_up',
    'woven':     'diamond_grid',
    'dome':      'arc_dome',
    'amorphous': 'blob',
}


def composite_to_visual(
    obj,          # CompositeObject
    cell_px: int,
    tick: int = 0,
) -> dict:
    """
    Berechnet visuelle Parameter fuer ein CompositeObject.
    Gibt ein dict mit allen Zeichenanweisungen zurueck.
    """
    if not obj.vectors:
        avg_vec = np.zeros(N_PROPS, dtype=np.float32)
    else:
        avg_vec = np.mean(obj.vectors, axis=0)

    base_vis = vector_to_visual(avg_vec, cell_px)

    # Emergente Eigenschaften beeinflussen die Visualisierung
    comfort_glow    = obj.comfort > 0.3
    shelter_visible = obj.shelter > 0.3
    capacity_ring   = obj.capacity > 0.2
    weapon_glow     = obj.attack_bonus > 0.3

    # Groesse: groessere Objekte = groessere Darstellung
    scale = min(2.5, 0.8 + obj.capacity * 0.5 + obj.shelter * 0.8)

    # Farbe modifizieren fuer emergente Eigenschaften
    r, g, b = base_vis.color
    if comfort_glow:
        g = _clamp(g + 40)       # weich = gruenlicher
    if weapon_glow:
        r = _clamp(r + 50)       # Waffe = roetlicher
    if shelter_visible:
        b = _clamp(b + 30)       # Schutz = blaulicher

    # Puls-Frequenz: wertvolle Objekte pulsieren leicht
    value_score = obj.comfort + obj.shelter + obj.capacity + obj.attack_bonus
    pulse = min(1.5, value_score * 0.4)

    return {
        'shape':          obj.shape,
        'draw_fn':        SHAPE_DRAW_PARAMS.get(obj.shape, 'blob'),
        'color':          (r, g, b),
        'alpha':          min(255, int(base_vis.alpha * 1.1)),
        'scale':          scale,
        'pulse':          pulse,
        'capacity_ring':  capacity_ring,
        'shelter_halo':   shelter_visible,
        'durability':     obj.durability,
        'glyph':          _object_glyph(obj),
        'tick':           tick,
    }


def _object_glyph(obj) -> str:
    """Passender Glyph fuer ein CompositeObject."""
    if obj.shelter > 0.3:      return '\u2302'  # Haus
    if obj.comfort > 0.4:      return '\u25ac'  # Bett (Rechteck)
    if obj.capacity > 0.3:     return '\u25cb'  # Behaelter (Kreis leer)
    if obj.attack_bonus > 0.3: return '\u2191'  # Waffe (Pfeil)
    if obj.insulation > 0.3:   return '\u2248'  # Isolation (Wellen)
    return '\u25a1'  # default: leeres Quadrat


# ---------------------------------------------------------------------------
# TokenVisual: Proto-Sprache sichtbar machen
# ---------------------------------------------------------------------------
def token_to_visual(token, shared: bool = False, tick: int = 0) -> dict:
    """
    Macht ein Sprach-Token sichtbar.
    shape_bits → Glyph-Auswahl aus einem festen Set.
    shared=True → Gold-Puls (konvergiertes Symbol)
    """
    # shape_bits -> einer von 16 Glyph-Zeichen
    SYMBOL_GLYPHS = [
        '\u25b2','\u25bc','\u25c6','\u25cf','\u2605','\u2736',
        '\u273f','\u2741','\u2665','\u2660','\u2666','\u2663',
        '\u0416','\u04d6','\u16a0','\u16b9',  # Runen
    ]
    glyph_idx = token.shape_bits % len(SYMBOL_GLYPHS)
    glyph     = SYMBOL_GLYPHS[glyph_idx]

    if shared:
        # Konvergiertes Symbol: Gold, stark sichtbar
        color      = (255, 215, 0)
        alpha      = 220
        pulse      = 1.0 + 0.5 * math.sin(tick * 0.1)
        border_col = (255, 255, 100)
    else:
        # Individuelles Token: gedaempft
        base_h = (token.shape_bits % 256) / 256.0
        color  = _hsl_to_rgb(base_h, 0.6, 0.55)
        alpha  = int(180 * token.durability)
        pulse  = 0.0
        border_col = color

    return {
        'glyph':      glyph,
        'color':      color,
        'alpha':      _clamp(alpha),
        'pulse':      pulse,
        'border':     border_col,
        'shared':     shared,
        'creator_id': token.creator_id,
    }


# ---------------------------------------------------------------------------
# FermentationVisual
# ---------------------------------------------------------------------------
STAGE_COLORS = {
    0: (180, 220, 100),   # frisch: gelbgruen
    1: (220, 160,  40),   # fermentierend: amber
    2: (160,  80,  20),   # gereift: dunkelbraun
    3: ( 80,  40,  80),   # verfault: dunkelviolett
}


def fermentation_to_visual(state, cell_px: int, tick: int) -> dict:
    color   = STAGE_COLORS.get(state.stage, (128, 128, 128))
    # Blasen-Animation bei Stage 1 (Gaerung aktiv)
    bubbles = []
    if state.stage == 1:
        rng = np.random.default_rng(tick % 60 + state.mat_id.__hash__() % 100)
        n_bubbles = int(state.moisture_acc * 5)
        for _ in range(n_bubbles):
            bx = rng.uniform(0.2, 0.8)
            by = rng.uniform(0.2, 0.8)
            br = rng.uniform(1, 3)
            bubbles.append((bx, by, br))

    opacity = int(120 + state.stage * 30)
    return {
        'color':   color,
        'alpha':   _clamp(opacity),
        'stage':   state.stage,
        'bubbles': bubbles,
    }


# ---------------------------------------------------------------------------
# GrowthVisual
# ---------------------------------------------------------------------------
def growth_to_visual(mat_id: str, qty: float, cell_px: int) -> dict:
    """
    Wachstumsstadien als Groessen-Gradient.
    qty < 0.2 = Samen (Punkt)
    qty < 0.5 = Keim (kleines Dreieck)
    qty < 1.0 = Jungpflanze (mittel)
    qty >= 1.0 = ausgewachsen (volle Groesse)
    """
    if qty < 0.2:
        shape, scale, glyph = 'dot',      0.15, '\u00b7'
    elif qty < 0.5:
        shape, scale, glyph = 'triangle', 0.25, '\u25b4'
    elif qty < 1.0:
        shape, scale, glyph = 'triangle', 0.45, '\u25b2'
    else:
        shape, scale, glyph = 'tree',     0.70, '\u2663'  # Kleeblatt als Baum

    # Farbe: gruene Toene, heller je reifer
    green_val = _clamp(80 + int(qty * 80), 80, 200)
    color     = (40, green_val, 30)

    return {
        'shape':  shape,
        'scale':  scale,
        'glyph':  glyph,
        'color':  color,
        'alpha':  180,
        'qty':    qty,
    }


# ---------------------------------------------------------------------------
# ConductivityVisual
# ---------------------------------------------------------------------------
def conductivity_chain_visual(chain: list[tuple], tick: int, cell_px: int) -> dict:
    """
    Leitfaehige Kette als leuchtende Linie.
    Energie-Fluss-Animation: helle Punkte wandern entlang der Kette.
    """
    # Fluss-Animation: Position bewegt sich mit tick
    flow_pos = (tick * 0.15) % 1.0  # 0..1 entlang der Kette

    return {
        'chain':      chain,
        'color':      (80, 200, 255),     # blau-weiss
        'width':      max(1, cell_px // 8),
        'alpha':      160,
        'flow_pos':   flow_pos,
        'flow_color': (255, 255, 255),
        'flow_size':  max(2, cell_px // 6),
    }


def gate_to_visual(gate, cell_px: int) -> dict:
    """Logikgatter als farbiges Quadrat."""
    TYPE_COLORS = {
        'AND':   (100, 200, 100),
        'OR':    (100, 100, 220),
        'NOT':   (220, 100, 100),
        'LATCH': (220, 180,  40),
        'WIRE':  ( 80, 160, 200),
    }
    color  = TYPE_COLORS.get(gate.gate_type, (150, 150, 150))
    active = gate.state
    return {
        'color':  color if active else tuple(c//3 for c in color),
        'active': active,
        'label':  gate.gate_type[0],  # 'A', 'O', 'N', 'L', 'W'
        'alpha':  220 if active else 100,
    }
