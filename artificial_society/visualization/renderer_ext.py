"""
Renderer Extension: Emergente Objekte zeichnen
-------------------------------------------------
Dieses Modul haengt sich in den bestehenden Renderer (renderer.py)
ohne ihn zu veraendern. Es fuegt draw-Methoden fuer alle neuen
emergierten Objekte hinzu.

Verwendung:
    from artificial_society.visualization.renderer_ext import patch_renderer
    patch_renderer(renderer_instance)

Danach hat renderer_instance folgende neue Methoden:
    draw_emergent_materials(screen, world)
    draw_composite_objects(screen, world)
    draw_tokens(screen, world)
    draw_conductivity_net(screen, world, conductivity_events)
    draw_fermentation(screen, world, fermentation_manager)
    draw_growth(screen, world)
    draw_logic_gates(screen, gate_network)

Die bestehende draw()-Methode bleibt unveraendert.
Einfach nach draw_world() und vor draw_agents() aufrufen.
"""

import math
import numpy as np

try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

from artificial_society.visualization.emergent_visuals import (
    vector_to_visual,
    composite_to_visual,
    token_to_visual,
    fermentation_to_visual,
    growth_to_visual,
    conductivity_chain_visual,
    gate_to_visual,
    VectorVisual,
)
from artificial_society.environment.materials import (
    IDX, MATERIALS, DISCOVERY_REGISTRY, get_vector
)

# Samen die als Wachstum visualisiert werden
GROWTH_MATS = {'seed_grain', 'seed_herb', 'seed_fiber', 'spore', 'root_cut', 'leaf'}


# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer Pygame-Zeichnen
# ---------------------------------------------------------------------------
def _draw_shape(screen, shape: str, cx: int, cy: int,
                r: float, color, alpha: int, outline_color=None):
    if not _PYGAME:
        return
    r = max(2, int(r))
    surf = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
    c = (*color, alpha)

    if shape == 'circle':
        pygame.draw.circle(surf, c, (r*2, r*2), r)
        if outline_color:
            pygame.draw.circle(surf, (*outline_color, 200), (r*2, r*2), r, 1)

    elif shape == 'square':
        pygame.draw.rect(surf, c, (r//2, r//2, r*3, r*3))
        if outline_color:
            pygame.draw.rect(surf, (*outline_color, 200), (r//2, r//2, r*3, r*3), 1)

    elif shape == 'triangle':
        pts = [(r*2, r//2), (r//3, r*3+r//2), (r*3+r*2//3, r*3+r//2)]
        pygame.draw.polygon(surf, c, pts)
        if outline_color:
            pygame.draw.polygon(surf, (*outline_color, 200), pts, 1)

    elif shape == 'diamond':
        pts = [(r*2, r//2), (r*3+r//2, r*2), (r*2, r*3+r//2), (r//2, r*2)]
        pygame.draw.polygon(surf, c, pts)

    screen.blit(surf, (cx - r*2, cy - r*2))


def _draw_glyph(screen, font, glyph: str, cx: int, cy: int,
                color, alpha: int):
    if not _PYGAME or font is None:
        return
    try:
        rendered = font.render(glyph, True, color)
        rendered.set_alpha(alpha)
        rect = rendered.get_rect(center=(cx, cy))
        screen.blit(rendered, rect)
    except Exception:
        pass


def _draw_aura(screen, cx: int, cy: int, radius: float, color_a):
    if not _PYGAME or radius < 1:
        return
    r = int(radius)
    surf = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
    for i in range(3, 0, -1):
        a = color_a[3] // (i + 1)
        pygame.draw.circle(surf, (*color_a[:3], a), (r*2, r*2), r * i)
    screen.blit(surf, (cx - r*2, cy - r*2))


# ---------------------------------------------------------------------------
# draw_emergent_materials
# ---------------------------------------------------------------------------
def _draw_emergent_materials(self, screen, world):
    """
    Zeichnet alle unbekannten (entdeckten) Materialien in Zellen.
    Bekannte Seed-Materialien werden vom Biome-Renderer gehandhabt.
    """
    if not _PYGAME:
        return
    cp   = self.cell_px
    font = getattr(self, 'small', None)
    tick = getattr(self, '_tick', 0)

    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            slot = cell.get('materials', {})
            cx_px = x * cp + cp // 2
            cy_px = y * cp + cp // 2

            for mat_id, qty in slot.items():
                if qty < 0.1:
                    continue
                # Nur entdeckte Materialien (mat_XXXX) hier zeichnen
                if mat_id in MATERIALS:
                    continue
                if not mat_id.startswith('mat_'):
                    continue

                vec = DISCOVERY_REGISTRY.peek_vector(mat_id)
                if vec is None or float(vec.sum()) < 0.05:
                    continue

                vis = vector_to_visual(vec, cp)

                # Puls-Modulation
                if vis.pulse_freq > 0:
                    pulse = 0.7 + 0.3 * math.sin(tick * vis.pulse_freq * 0.15)
                else:
                    pulse = 1.0

                # Aura
                if vis.aura_radius > 1:
                    _draw_aura(screen, cx_px, cy_px,
                               vis.aura_radius * pulse, vis.aura_color)

                # Form
                r = vis.radius * pulse * min(1.5, qty)
                _draw_shape(screen, vis.shape, cx_px, cy_px, r,
                            vis.color, vis.alpha,
                            vis.outline_color if vis.outline else None)

                # Glyph (nur bei groesseren Cells)
                if cp >= 16 and font:
                    _draw_glyph(screen, font, vis.glyph,
                                cx_px, cy_px, vis.glyph_color, vis.alpha)


# ---------------------------------------------------------------------------
# draw_composite_objects
# ---------------------------------------------------------------------------
def _draw_composite_objects(self, screen, world):
    """
    Zeichnet geformte Objekte (geometry.py CompositeObjects).
    Jede Shape hat eine eigene Zeichenroutine.
    """
    if not _PYGAME:
        return
    cp   = self.cell_px
    font = getattr(self, 'small', None)
    tick = getattr(self, '_tick', 0)

    for y in range(world.height):
        for x in range(world.width):
            cell    = world.cells[y][x]
            objects = cell.get('objects', [])
            cx_px   = x * cp + cp // 2
            cy_px   = y * cp + cp // 2

            for obj in objects:
                params = composite_to_visual(obj, cp, tick)
                color  = params['color']
                alpha  = params['alpha']
                scale  = params['scale']
                shape  = params['shape']
                glyph  = params['glyph']
                r      = int(cp * 0.35 * scale)

                # Puls
                if params['pulse'] > 0:
                    pf = 0.85 + 0.15 * math.sin(tick * params['pulse'] * 0.1)
                    r  = int(r * pf)

                draw_fn = params['draw_fn']

                if draw_fn == 'circle_outline':
                    pygame.draw.circle(screen, color, (cx_px, cy_px), r, max(2, r//4))

                elif draw_fn == 'rect_fill':
                    pygame.draw.rect(screen, color,
                        (cx_px - r, cy_px - r//2, r*2, r))

                elif draw_fn == 'circle_fill':
                    surf = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*color, alpha), (r+2, r+2), r)
                    screen.blit(surf, (cx_px - r - 2, cy_px - r - 2))

                elif draw_fn == 'rect_layered':
                    for i in range(3):
                        pygame.draw.rect(screen,
                            tuple(max(0, c - i*20) for c in color),
                            (cx_px - r + i*2, cy_px - r//2 + i*3, r*2 - i*4, r//2), 1)

                elif draw_fn == 'triangle_up':
                    pts = [(cx_px, cy_px - r),
                           (cx_px - r//2, cy_px + r//2),
                           (cx_px + r//2, cy_px + r//2)]
                    pygame.draw.polygon(screen, color, pts)

                elif draw_fn == 'diamond_grid':
                    pts = [(cx_px, cy_px - r), (cx_px + r, cy_px),
                           (cx_px, cy_px + r), (cx_px - r, cy_px)]
                    pygame.draw.polygon(screen, color, pts, 2)
                    pygame.draw.line(screen, color,
                                     (cx_px - r, cy_px), (cx_px + r, cy_px), 1)
                    pygame.draw.line(screen, color,
                                     (cx_px, cy_px - r), (cx_px, cy_px + r), 1)

                elif draw_fn == 'arc_dome':
                    pygame.draw.arc(screen, color,
                        (cx_px - r, cy_px - r, r*2, r*2), 0, math.pi,
                        max(2, r//3))
                    pygame.draw.line(screen, color,
                        (cx_px - r, cy_px), (cx_px + r, cy_px), 1)

                else:  # blob
                    surf = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*color, alpha//2), (r+2, r+2), r)
                    screen.blit(surf, (cx_px - r - 2, cy_px - r - 2))

                # Shelter-Halo
                if params['shelter_halo']:
                    h_surf = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
                    pygame.draw.circle(h_surf, (100, 140, 255, 40),
                                       (r*2, r*2), r + cp//3)
                    screen.blit(h_surf, (cx_px - r*2, cy_px - r*2))

                # Capacity-Ring (Behaelter-Anzeige)
                if params['capacity_ring']:
                    pygame.draw.circle(screen, (200, 200, 255),
                                       (cx_px, cy_px), r + 3, 1)

                # Glyph
                if cp >= 14 and font:
                    _draw_glyph(screen, font, glyph,
                                cx_px, cy_px, (255, 255, 255), 200)


# ---------------------------------------------------------------------------
# draw_tokens
# ---------------------------------------------------------------------------
def _draw_tokens(self, screen, world):
    """Zeichnet alle Sprach-Token in der Welt."""
    if not _PYGAME:
        return
    cp   = self.cell_px
    font = getattr(self, 'font', None)
    tick = getattr(self, '_tick', 0)

    from artificial_society.systems.language import TOKEN_WORLD
    for token_id, token in TOKEN_WORLD.tokens.items():
        cx_px = token.x * cp + cp // 2
        cy_px = token.y * cp + cp // 2

        # Ist dieses Symbol geteilt?
        shared = any(
            token_id in mem.associations
            and mem.associations[token_id].is_shared
            for mem in getattr(world, '_agent_token_memories', [])
        )

        params = token_to_visual(token, shared=shared, tick=tick)
        color  = params['color']
        alpha  = params['alpha']
        glyph  = params['glyph']

        # Shared: goldene Umrandung + Puls
        if shared:
            pulse_r = int(cp * 0.3 * (1.0 + 0.2 * math.sin(tick * 0.1)))
            ring_surf = pygame.Surface((pulse_r*4, pulse_r*4), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (255, 215, 0, 120),
                               (pulse_r*2, pulse_r*2), pulse_r, 2)
            screen.blit(ring_surf, (cx_px - pulse_r*2, cy_px - pulse_r*2))

        # Glyph
        if font:
            _draw_glyph(screen, font, glyph, cx_px, cy_px - cp//4, color, alpha)


# ---------------------------------------------------------------------------
# draw_conductivity_net
# ---------------------------------------------------------------------------
def _draw_conductivity_net(self, screen, world, chains=None):
    """Zeichnet leitfaehige Ketten als animierte Linien."""
    if not _PYGAME:
        return
    cp   = self.cell_px
    tick = getattr(self, '_tick', 0)

    if chains is None:
        from artificial_society.systems.conductivity import find_conductive_chains
        chains = find_conductive_chains(world, min_length=2)

    for chain in chains:
        if len(chain) < 2:
            continue
        params = conductivity_chain_visual(chain, tick, cp)
        color  = params['color']
        width  = params['width']
        alpha  = params['alpha']

        # Linie zeichnen
        points = [(x * cp + cp//2, y * cp + cp//2) for x, y in chain]
        surf   = pygame.Surface((world.width*cp, world.height*cp), pygame.SRCALPHA)
        for i in range(len(points)-1):
            pygame.draw.line(surf, (*color, alpha), points[i], points[i+1], width)
        screen.blit(surf, (0, 0))

        # Energie-Fluss-Punkt
        fp    = params['flow_pos']
        idx_f = fp * (len(points) - 1)
        i0    = int(idx_f)
        i1    = min(i0 + 1, len(points) - 1)
        t     = idx_f - i0
        fx    = int(points[i0][0] * (1-t) + points[i1][0] * t)
        fy    = int(points[i0][1] * (1-t) + points[i1][1] * t)
        pygame.draw.circle(screen, params['flow_color'], (fx, fy), params['flow_size'])


# ---------------------------------------------------------------------------
# draw_fermentation
# ---------------------------------------------------------------------------
def _draw_fermentation(self, screen, world, fermentation_manager=None):
    """Zeichnet Gaerungszustand als Farb-Overlay + Blasen."""
    if not _PYGAME:
        return
    cp   = self.cell_px
    tick = getattr(self, '_tick', 0)

    if fermentation_manager is None:
        from artificial_society.environment.fermentation import FERMENTATION_MANAGER
        fermentation_manager = FERMENTATION_MANAGER

    for (x, y, mat_id), state in fermentation_manager.states.items():
        if state.stage == 0:
            continue
        cx_px = x * cp + cp // 2
        cy_px = y * cp + cp // 2
        params = fermentation_to_visual(state, cp, tick)

        # Farb-Overlay ueber die Zelle
        overlay = pygame.Surface((cp, cp), pygame.SRCALPHA)
        overlay.fill((*params['color'], params['alpha'] // 2))
        screen.blit(overlay, (x * cp, y * cp))

        # Blasen bei Stage 1
        for bx_rel, by_rel, br in params.get('bubbles', []):
            bx = int(x * cp + bx_rel * cp)
            by = int(y * cp + by_rel * cp)
            pygame.draw.circle(screen, (*params['color'], 180),
                               (bx, by), max(1, int(br)))


# ---------------------------------------------------------------------------
# draw_growth
# ---------------------------------------------------------------------------
def _draw_growth(self, screen, world):
    """Zeichnet biologisches Wachstum als Groessen-Gradient."""
    if not _PYGAME:
        return
    cp   = self.cell_px
    font = getattr(self, 'small', None)

    for y in range(world.height):
        for x in range(world.width):
            cell = world.cells[y][x]
            slot = cell.get('materials', {})
            cx_px = x * cp + cp // 2
            cy_px = y * cp + cp // 2

            for mat_id, qty in slot.items():
                if mat_id not in GROWTH_MATS:
                    continue
                if qty < 0.05:
                    continue

                params = growth_to_visual(mat_id, qty, cp)
                r      = int(cp * params['scale'])
                color  = params['color']
                shape  = params['shape']

                if shape == 'dot':
                    pygame.draw.circle(screen, color, (cx_px, cy_px), max(1, r))

                elif shape == 'triangle':
                    pts = [(cx_px, cy_px - r),
                           (cx_px - r//2, cy_px + r//2),
                           (cx_px + r//2, cy_px + r//2)]
                    surf = pygame.Surface((r*3, r*3), pygame.SRCALPHA)
                    pygame.draw.polygon(screen, (*color, 200), pts)

                elif shape == 'tree':
                    # Stamm
                    pygame.draw.line(screen, (80, 50, 20),
                                     (cx_px, cy_px + r//2),
                                     (cx_px, cy_px), max(1, r//4))
                    # Krone
                    pygame.draw.circle(screen, color, (cx_px, cy_px - r//4), r//2)

                # Glyph fuer groessere Darstellungen
                if cp >= 18 and font and r > 4:
                    _draw_glyph(screen, font, params['glyph'],
                                cx_px, cy_px, (255, 255, 255), 160)


# ---------------------------------------------------------------------------
# draw_logic_gates
# ---------------------------------------------------------------------------
def _draw_logic_gates(self, screen, gate_network=None):
    """Zeichnet Logikgatter als farbige Quadrate auf den Switch-Zellen."""
    if not _PYGAME:
        return
    cp   = self.cell_px
    font = getattr(self, 'small', None)

    if gate_network is None:
        from artificial_society.systems.logic_gates import GATE_NETWORK
        gate_network = GATE_NETWORK

    for gate_id, gate in gate_network.gates.items():
        for sx, sy in gate.switch_cells:
            cx_px = sx * cp + cp // 2
            cy_px = sy * cp + cp // 2
            params = gate_to_visual(gate, cp)
            r      = max(3, cp // 3)

            # Quadrat fuer Gate-Knoten
            surf = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
            surf.fill((*params['color'], params['alpha']))
            screen.blit(surf, (cx_px - r - 1, cy_px - r - 1))

            # Label (A/O/N/L/W)
            if cp >= 16 and font:
                _draw_glyph(screen, font, params['label'],
                            cx_px, cy_px, (255,255,255), 220)


# ---------------------------------------------------------------------------
# Patch-Funktion: haengt alle Methoden in bestehenden Renderer
# ---------------------------------------------------------------------------
def patch_renderer(renderer):
    """
    Haengt alle neuen draw-Methoden in einen bestehenden Renderer.
    Veraendert renderer.py NICHT.

    Verwendung:
        from artificial_society.visualization.renderer_ext import patch_renderer
        patch_renderer(renderer_instance)
    """
    import types
    renderer.draw_emergent_materials  = types.MethodType(_draw_emergent_materials,  renderer)
    renderer.draw_composite_objects   = types.MethodType(_draw_composite_objects,   renderer)
    renderer.draw_tokens              = types.MethodType(_draw_tokens,              renderer)
    renderer.draw_conductivity_net    = types.MethodType(_draw_conductivity_net,    renderer)
    renderer.draw_fermentation        = types.MethodType(_draw_fermentation,        renderer)
    renderer.draw_growth              = types.MethodType(_draw_growth,              renderer)
    renderer.draw_logic_gates         = types.MethodType(_draw_logic_gates,         renderer)
    return renderer
