import math
import random
import pygame
from artificial_society.visualization.overlays import draw_dashboard
from artificial_society.environment.biomes import BIOME_BASE_COLOR, BIOME_DETAIL_COLOR

# ---------------------------------------------------------------------------
# Disease ring colors per disease type
# ---------------------------------------------------------------------------
DISEASE_COLORS = {
    'malaria':      (200, 230,  80),
    'dysentery':    (160, 110,  50),
    'tuberculosis': (200, 200, 200),
    'typhoid':      (240, 180,  40),
    'scurvy':       (220, 110,  30),
    'wound_fever':  (220,  40,  40),
}
DEFAULT_SICK_COLOR = (210, 60, 60)

# Action mode icon colors
MODE_COLORS = {
    'gather':      (120, 245, 120),
    'build:camp':  (240, 210, 120),
    'build:farm':  (200, 240, 100),
    'build:well':  (120, 200, 255),
    'signal':      (255, 240,  80),
    'attack':      (255,  70,  70),
    'mate':        (255, 150, 200),
    'forage_herb': (120, 255, 160),
    'experiment':  (180, 140, 255),
    'sick':        (210,  60,  60),
}

# Deterministic per-cell noise seed
_NOISE_CACHE: dict = {}


def _tile_noise(x: int, y: int, amp: int = 10) -> int:
    key = (x, y)
    if key not in _NOISE_CACHE:
        random.seed(x * 9973 + y * 1009)
        _NOISE_CACHE[key] = random.randint(-amp, amp)
        random.seed()
    return _NOISE_CACHE[key]


def _clamp_color(c):
    return tuple(max(0, min(255, v)) for v in c)


def _shift(color, d):
    return _clamp_color(c + d for c in color)


def _blend(a, b, t):
    return _clamp_color(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Renderer:
    def __init__(self, width, height, cell_px):
        self.width = width
        self.height = height
        self.cell_px = cell_px
        self.font      = pygame.font.SysFont('segoeui', 15)
        self.font_bold = pygame.font.SysFont('segoeui', 15, bold=True)
        self.small     = pygame.font.SysFont('segoeui', 12)
        self.dashboard_x = width - 300
        self._surf_cache: dict = {}
        self._tick = 0

    # ------------------------------------------------------------------
    # Alpha surface cache (avoids constant Surface allocation)
    # ------------------------------------------------------------------
    def _alpha(self, w, h, rgba):
        key = (w, h, rgba)
        if key not in self._surf_cache:
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill(rgba)
            self._surf_cache[key] = s
        return self._surf_cache[key]

    # ------------------------------------------------------------------
    # World rendering
    # ------------------------------------------------------------------
    def draw_world(self, screen, world):
        cp = self.cell_px
        for y in range(world.height):
            for x in range(world.width):
                biome = world.get_biome(x, y)
                cell  = world.get_cell(x, y)
                base  = world.color_at(x, y)          # from biomes.biome_color()
                n     = _tile_noise(x, y, 9)
                color = _clamp_color(c + n for c in base)
                rx, ry = x * cp, y * cp
                rect = (rx, ry, cp, cp)

                # ---- Base tile ----
                pygame.draw.rect(screen, color, rect)

                # ---- Biome-specific texture detail ----
                self._draw_biome_texture(screen, biome, cell, x, y, cp, color)

                # ---- Environmental overlays ----
                dis = cell['disease']
                pol = cell['pollution']
                dist = cell['disturbance']
                ash  = cell['ash']

                # Disease: sickly purple-green mist
                if dis > 10:
                    alpha = min(80, int(dis * 1.2))
                    screen.blit(self._alpha(cp, cp, (120, 40, 140, alpha)), (rx, ry))

                # Pollution: dark brownish haze
                if pol > 15:
                    alpha = min(70, int(pol * 0.9))
                    screen.blit(self._alpha(cp, cp, (80, 55, 20, alpha)), (rx, ry))

                # Disturbance: orange shimmer (fire/storm)
                if dist > 20:
                    alpha = min(100, int(dist * 1.3))
                    screen.blit(self._alpha(cp, cp, (240, 120, 30, alpha)), (rx, ry))

                # Ash: grey veil after fire
                if ash > 12:
                    alpha = min(90, int(ash * 1.5))
                    screen.blit(self._alpha(cp, cp, (100, 100, 100, alpha)), (rx, ry))

                # ---- Structures ----
                if cell['structures'].get('camp'):  self._draw_camp(screen, rx, ry, cp)
                if cell['structures'].get('farm'):  self._draw_farm(screen, rx, ry, cp)
                if cell['structures'].get('well'):  self._draw_well(screen, rx, ry, cp)

                # ---- Subtle tile border ----
                if cp >= 10:
                    pygame.draw.rect(screen, _shift(color, -18), rect, 1)

    # ------------------------------------------------------------------
    # Biome texture details (drawn on top of base color)
    # ------------------------------------------------------------------
    def _draw_biome_texture(self, screen, biome, cell, gx, gy, cp, base_color):
        rx, ry = gx * cp, gy * cp
        detail = BIOME_DETAIL_COLOR.get(biome, base_color)

        if biome == 'forest' and cp >= 8:
            # Tree canopy dots — 3 per cell in fixed positions
            offsets = [(cp // 4, cp // 4), (cp * 3 // 4, cp // 3), (cp // 2, cp * 2 // 3)]
            r = max(2, cp // 5)
            for ox, oy in offsets:
                pygame.draw.circle(screen, detail, (rx + ox, ry + oy), r)

        elif biome == 'grassland' and cp >= 8:
            # Grass tufts — short vertical strokes
            tuft_color = BIOME_DETAIL_COLOR['grassland']
            for ox in (cp // 4, cp // 2, cp * 3 // 4):
                h_tuft = max(2, cp // 5)
                pygame.draw.line(screen, tuft_color,
                                 (rx + ox, ry + cp - 2),
                                 (rx + ox, ry + cp - 2 - h_tuft), 1)

        elif biome == 'desert' and cp >= 8:
            # Dune ripple — single horizontal arc
            cx_d = rx + cp // 2
            cy_d = ry + cp * 2 // 3
            pygame.draw.arc(screen, BIOME_DETAIL_COLOR['desert'],
                            (cx_d - cp // 3, cy_d - cp // 6, cp * 2 // 3, cp // 4),
                            0, math.pi, 1)

        elif biome == 'mountain' and cp >= 8:
            # Mountain peak triangle
            peak   = (rx + cp // 2, ry + 2)
            left   = (rx + 2,       ry + cp - 3)
            right  = (rx + cp - 2,  ry + cp - 3)
            pygame.draw.polygon(screen, BIOME_DETAIL_COLOR['mountain'],
                                [peak, left, right], 1)
            # Snow cap (white filled small triangle)
            snow_l = (rx + cp // 2 - cp // 6, ry + cp // 3)
            snow_r = (rx + cp // 2 + cp // 6, ry + cp // 3)
            pygame.draw.polygon(screen, (235, 240, 245), [peak, snow_l, snow_r])

        elif biome == 'swamp' and cp >= 8:
            # Small water puddle ellipse
            puddle_rect = (rx + cp // 4, ry + cp // 3, cp // 2, cp // 4)
            pygame.draw.ellipse(screen, (45, 75, 90), puddle_rect)

        elif biome == 'water' and cp >= 8:
            # Wave line
            wave_y = ry + cp // 2
            wave_color = BIOME_DETAIL_COLOR['water']
            for i in range(0, cp - 3, max(3, cp // 4)):
                pygame.draw.arc(screen, wave_color,
                                (rx + i, wave_y - cp // 8, max(3, cp // 4), cp // 6),
                                0, math.pi, 1)

    # ------------------------------------------------------------------
    # Structure symbols (recognisable real-world glyphs)
    # ------------------------------------------------------------------
    def _draw_camp(self, screen, rx, ry, cp):
        """Tent: two triangles side by side."""
        # Left tent
        mid = cp // 2
        q   = cp // 4
        pygame.draw.polygon(screen, (215, 185, 100),
                            [(rx + q,     ry + cp - 3),
                             (rx + mid,   ry + 3),
                             (rx + 3,     ry + cp - 3)], 1)
        # Right tent
        pygame.draw.polygon(screen, (215, 185, 100),
                            [(rx + mid,   ry + cp - 3),
                             (rx + cp-3,  ry + cp - 3),
                             (rx + cp*3//4, ry + 4)], 1)

    def _draw_farm(self, screen, rx, ry, cp):
        """Ploughed field: grid of short lines."""
        col = (160, 130, 60)
        step = max(3, cp // 3)
        for i in range(rx + 2, rx + cp - 1, step):
            pygame.draw.line(screen, col, (i, ry + 2), (i, ry + cp - 2), 1)
        for j in range(ry + 2, ry + cp - 1, step):
            pygame.draw.line(screen, col, (rx + 2, j), (rx + cp - 2, j), 1)

    def _draw_well(self, screen, rx, ry, cp):
        """Well: circle with cross on top (bucket symbol)."""
        cx, cy = rx + cp // 2, ry + cp // 2
        r = max(2, cp // 5)
        pygame.draw.circle(screen, (170, 210, 235), (cx, cy), r, 1)
        # Roof cross bar
        pygame.draw.line(screen, (170, 210, 235), (cx - r, cy - r - 2), (cx + r, cy - r - 2), 1)
        # Vertical support
        pygame.draw.line(screen, (170, 210, 235), (cx, cy - r - 4), (cx, cy - r), 1)

    # ------------------------------------------------------------------
    # Social links
    # ------------------------------------------------------------------
    def draw_social_links(self, screen, agents):
        cp = self.cell_px
        for agent in agents:
            ax = int(agent.pos[0] * cp + cp / 2)
            ay = int(agent.pos[1] * cp + cp / 2)
            for other in agents:
                if other.id <= agent.id:
                    continue
                if abs(other.pos[0]-agent.pos[0]) <= 2 and abs(other.pos[1]-agent.pos[1]) <= 2:
                    trust = agent.trust.get(other.id, 0.0)
                    if trust > 0.30:
                        ox = int(other.pos[0] * cp + cp / 2)
                        oy = int(other.pos[1] * cp + cp / 2)
                        pygame.draw.line(screen, (200, 200, 200), (ax, ay), (ox, oy), 1)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def draw_agents(self, screen, agents, tribes):
        cp = self.cell_px
        radius = max(5, cp // 2 - 1)
        for agent in agents:
            px = int(agent.pos[0] * cp + cp / 2)
            py = int(agent.pos[1] * cp + cp / 2)

            color = tribes.color_for(agent.tribe_id) if agent.tribe_id is not None else agent.display_color()

            # Drop shadow
            pygame.draw.circle(screen, (0, 0, 0), (px + 1, py + 2), radius)
            # Body
            pygame.draw.circle(screen, color, (px, py), radius)
            # Bright highlight (top-left)
            hi = _clamp_color(c + 60 for c in color)
            pygame.draw.circle(screen, hi, (px - radius // 3, py - radius // 3), max(1, radius // 3))
            # Dark outline
            pygame.draw.circle(screen, _shift(color, -50), (px, py), radius, 1)

            # Disease ring (disease-type color)
            sick = getattr(agent, 'sick', 0.0)
            disease_id = getattr(agent, 'disease_id', None)
            if sick > 8 or disease_id:
                ring_col = DISEASE_COLORS.get(disease_id, DEFAULT_SICK_COLOR)
                pygame.draw.circle(screen, ring_col, (px, py), radius + 3, 2)

            # Pregnancy ring (pink)
            if agent.pregnant:
                pygame.draw.circle(screen, (255, 110, 170), (px, py), radius + 5, 1)

            # Tool ring (white)
            if agent.tool is not None:
                pygame.draw.circle(screen, (255, 255, 255), (px, py), radius + 6, 1)

            # Sex indicator dot
            dot_col = (255, 140, 180) if agent.sex == 'f' else (100, 180, 255)
            pygame.draw.circle(screen, dot_col, (px + radius - 1, py - radius + 1), 2)

            # Action mode icon
            mode = getattr(agent, 'last_action_mode', 'idle')
            icon_col = MODE_COLORS.get(mode)
            if icon_col:
                self._draw_mode_icon(screen, px, py, radius, mode, icon_col)

            # Communication aura
            mv = getattr(agent, 'message_vector', [0,0,0,0]) or [0,0,0,0]
            intensity = min(120, int(sum(abs(v) for v in mv) * 22))
            if intensity > 20:
                aura = pygame.Surface((radius*6, radius*6), pygame.SRCALPHA)
                pygame.draw.circle(aura, (255, 230, 70, intensity),
                                   (radius*3, radius*3), radius + 8, 2)
                screen.blit(aura, (px - radius*3, py - radius*3))

    def _draw_mode_icon(self, screen, px, py, radius, mode, color):
        top = py - radius - 8
        if mode == 'gather':
            # Green dot = collecting food
            pygame.draw.circle(screen, color, (px, top), 3)
        elif mode.startswith('build'):
            # Small square = building
            pygame.draw.rect(screen, color, (px - 3, top - 3, 7, 7))
        elif mode == 'signal':
            # Triangle = communicating
            pygame.draw.polygon(screen, color, [(px, top-5),(px-4, top),(px+4, top)])
        elif mode == 'attack':
            # X = fighting
            pygame.draw.line(screen, color, (px-4, top-3), (px+4, top+3), 2)
            pygame.draw.line(screen, color, (px+4, top-3), (px-4, top+3), 2)
        elif mode == 'sick':
            # Cross = sick
            pygame.draw.line(screen, color, (px, top-4), (px, top+4), 2)
            pygame.draw.line(screen, color, (px-4, top), (px+4, top), 2)
        elif mode == 'mate':
            # Heart-ish: two circles
            pygame.draw.circle(screen, color, (px - 2, top), 2)
            pygame.draw.circle(screen, color, (px + 2, top), 2)
        elif mode == 'forage_herb':
            # Leaf: small circle + stem
            pygame.draw.circle(screen, color, (px, top), 3)
            pygame.draw.line(screen, color, (px, top+2), (px, top+6), 1)
        elif mode == 'experiment':
            # Question mark shape: circle + dot
            pygame.draw.circle(screen, color, (px, top), 4, 1)
            pygame.draw.circle(screen, color, (px, top), 1)

    # ------------------------------------------------------------------
    # Hotspot outlines
    # ------------------------------------------------------------------
    def draw_hotspots(self, screen, world):
        cp = self.cell_px
        for x, y, cell in world.hotspots():
            if cell['disease'] > 25:
                # Pulsing purple disease hotspot ring
                pygame.draw.circle(screen, (160, 40, 160),
                                   (x*cp + cp//2, y*cp + cp//2), max(3, cp//2), 1)
            if cell['disturbance'] > 30:
                # Orange disturbance border
                pygame.draw.rect(screen, (240, 150, 40),
                                 (x*cp+1, y*cp+1, cp-2, cp-2), 1)

    # ------------------------------------------------------------------
    # Vignette
    # ------------------------------------------------------------------
    def _draw_vignette(self, screen):
        w = self.dashboard_x
        h = self.height
        depth = 22
        for i in range(depth):
            alpha = int(100 * ((1 - i / depth) ** 2))
            edges = [
                ((0, 0),       (w, i + 1)),
                ((0, h-i-1),   (w, 1)),
                ((0, 0),       (i + 1, h)),
                ((w-i-1, 0),   (1, h)),
            ]
            for (ex, ey), (ew, eh) in edges:
                s = pygame.Surface((ew, eh), pygame.SRCALPHA)
                s.fill((0, 0, 0, alpha))
                screen.blit(s, (ex, ey))

    # ------------------------------------------------------------------
    # Master draw
    # ------------------------------------------------------------------
    def draw(self, screen, world, agents, stats, tribes, technology):
        self._tick += 1
        screen.fill((10, 10, 14))
        self.draw_world(screen, world)
        self.draw_hotspots(screen, world)
        self.draw_social_links(screen, agents)
        self.draw_agents(screen, agents, tribes)
        self._draw_vignette(screen)
        draw_dashboard(
            screen, self.dashboard_x, 0, 300, self.height,
            stats, tribes, technology,
            self.font, self.font_bold, self.small,
            tick=self._tick,
        )
        pygame.display.flip()
