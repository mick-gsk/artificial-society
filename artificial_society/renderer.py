import math
import random
import pygame
from artificial_society.visualization.overlays import draw_dashboard
from artificial_society.systems.remedy import REMEDY_REGISTRY

# ---------------------------------------------------------------------------
# Biome base colors  —  richer, more painterly palette
# ---------------------------------------------------------------------------
BIOME_COLORS = {
    'grassland':  (78,  138,  68),
    'forest':     (34,   90,  45),
    'desert':     (194, 165,  96),
    'mountain':   (130, 118, 108),
    'swamp':      (62,   98,  72),
    'water':      (42,   98, 165),
    'tundra':     (170, 182, 175),
    'savanna':    (165, 148,  68),
    'jungle':     (22,   80,  42),
    'wetland':    (58,  110,  85),
}

# Disease ring colors per disease
DISEASE_COLORS = {
    'malaria':      (200, 230,  80),   # sickly yellow-green
    'dysentery':    (150, 100,  50),   # muddy brown
    'tuberculosis': (200, 200, 200),   # pallor grey
    'typhoid':      (240, 180,  40),   # fever gold
    'scurvy':       (220, 100,  30),   # vitamin-orange
    'wound_fever':  (220,  40,  40),   # blood red
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
    'forage_herb': (160, 255, 180),
    'experiment':  (180, 140, 255),
    'sick':        (210,  60,  60),
}

_NOISE_CACHE: dict = {}


def _tile_noise(x: int, y: int) -> int:
    """Deterministic per-cell brightness jitter ±8."""
    key = (x, y)
    if key not in _NOISE_CACHE:
        random.seed(x * 1000 + y)
        _NOISE_CACHE[key] = random.randint(-8, 8)
        random.seed()  # restore
    return _NOISE_CACHE[key]


def _blend(c, amount, target=(0, 0, 0)):
    """Blend color c toward target by amount [0..1]."""
    return tuple(max(0, min(255, int(c[i] + (target[i] - c[i]) * amount))) for i in range(3))


def _shift(color, delta):
    return tuple(max(0, min(255, c + delta)) for c in color)


class Renderer:
    def __init__(self, width, height, cell_px):
        self.width = width
        self.height = height
        self.cell_px = cell_px
        self.font = pygame.font.SysFont('segoeui', 15)
        self.font_bold = pygame.font.SysFont('segoeui', 15, bold=True)
        self.small = pygame.font.SysFont('segoeui', 12)
        self.dashboard_x = width - 300
        self._surf_cache: dict = {}   # (w, h, color_key) -> Surface
        self._tick = 0

    def _alpha_surf(self, w: int, h: int, color_rgba: tuple) -> pygame.Surface:
        key = (w, h, color_rgba)
        if key not in self._surf_cache:
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill(color_rgba)
            self._surf_cache[key] = s
        return self._surf_cache[key]

    # ------------------------------------------------------------------
    # World tiles
    # ------------------------------------------------------------------
    def draw_world(self, screen, world):
        cp = self.cell_px
        for y in range(world.height):
            for x in range(world.width):
                biome = world.get_biome(x, y)
                base = BIOME_COLORS.get(biome, (100, 100, 100))
                n = _tile_noise(x, y)
                color = _shift(base, n)
                rect = (x * cp, y * cp, cp, cp)
                pygame.draw.rect(screen, color, rect)

                cell = world.get_cell(x, y)

                # --- Food richness: subtle green tint ---
                food_ratio = cell['food'] / 180.0
                if food_ratio > 0.4 and biome not in ('water',):
                    surf = self._alpha_surf(cp, cp, (60, 180, 60, int(30 * food_ratio)))
                    screen.blit(surf, rect[:2])

                # --- Disease fog: purple-green tint ---
                dis = cell['disease']
                pol = cell['pollution']
                if dis > 12 or pol > 18:
                    alpha = min(100, int(dis * 1.0 + pol * 0.7))
                    screen.blit(self._alpha_surf(cp, cp, (140, 40, 120, alpha)), rect[:2])

                # --- Disturbance shimmer: orange ---
                if cell['disturbance'] > 18:
                    alpha = min(110, int(cell['disturbance'] * 1.1))
                    screen.blit(self._alpha_surf(cp, cp, (240, 140, 45, alpha)), rect[:2])

                # --- Ash: grey cross-hatch lines ---
                if cell['ash'] > 12:
                    pygame.draw.line(screen, (70, 70, 70),
                                     (x*cp+1, y*cp+1), (x*cp+cp-2, y*cp+cp-2), 1)

                # --- High moisture dot ---
                if cell['moisture'] > 68 and biome != 'water':
                    pygame.draw.circle(screen, (100, 170, 255),
                                       (x*cp + cp//2, y*cp + cp//2), max(1, cp//9))

                # --- Structures ---
                if cell['structures'].get('camp'):
                    self._draw_camp(screen, x, y, cp)
                if cell['structures'].get('farm'):
                    self._draw_farm(screen, x, y, cp)
                if cell['structures'].get('well'):
                    self._draw_well(screen, x, y, cp)

                # --- Tile border (very subtle) ---
                if cp >= 10:
                    pygame.draw.rect(screen, _shift(color, -14), rect, 1)

    def _draw_camp(self, screen, x, y, cp):
        cx, cy = x*cp + cp//2, y*cp + cp//2
        s = max(3, cp//3)
        pts = [(cx, cy - s), (cx - s, cy + s//2), (cx + s, cy + s//2)]
        pygame.draw.polygon(screen, (215, 190, 110), pts, 1)

    def _draw_farm(self, screen, x, y, cp):
        for i in range(x*cp + 2, x*cp + cp - 2, max(3, cp//4)):
            pygame.draw.line(screen, (175, 145, 60),
                             (i, y*cp + 2), (i, y*cp + cp - 2), 1)

    def _draw_well(self, screen, x, y, cp):
        cx, cy = x*cp + cp//2, y*cp + cp//2
        pygame.draw.circle(screen, (180, 215, 235), (cx, cy), max(2, cp//5), 1)
        pygame.draw.line(screen, (180, 215, 235),
                         (cx - cp//5, cy), (cx + cp//5, cy), 1)

    # ------------------------------------------------------------------
    # Social links
    # ------------------------------------------------------------------
    def draw_social_links(self, screen, agents):
        cp = self.cell_px
        for agent in agents:
            ax = int(agent.pos[0]*cp + cp/2)
            ay = int(agent.pos[1]*cp + cp/2)
            for other in agents:
                if other.id <= agent.id:
                    continue
                if abs(other.pos[0]-agent.pos[0]) <= 2 and abs(other.pos[1]-agent.pos[1]) <= 2:
                    trust = agent.trust.get(other.id, 0.0)
                    if trust > 0.25:
                        ox = int(other.pos[0]*cp + cp/2)
                        oy = int(other.pos[1]*cp + cp/2)
                        alpha = min(180, int(trust * 180))
                        pygame.draw.line(screen, (220, 220, 220), (ax, ay), (ox, oy), 1)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def draw_agents(self, screen, agents, tribes):
        cp = self.cell_px
        radius = max(4, cp // 3)
        for agent in agents:
            px = int(agent.pos[0]*cp + cp/2)
            py = int(agent.pos[1]*cp + cp/2)

            # Body color
            color = tribes.color_for(agent.tribe_id) if agent.tribe_id is not None else agent.display_color()

            # Shadow
            pygame.draw.circle(screen, (0, 0, 0), (px+1, py+1), radius)
            # Body
            pygame.draw.circle(screen, color, (px, py), radius)
            # Inner highlight
            hi = _shift(color, 55)
            pygame.draw.circle(screen, hi, (px - radius//3, py - radius//3), max(1, radius//3))

            # Disease ring (colored per disease type)
            sick = getattr(agent, 'sick', 0.0)
            disease_id = getattr(agent, 'disease_id', None)
            if sick > 10 or disease_id:
                ring_color = DISEASE_COLORS.get(disease_id, DEFAULT_SICK_COLOR)
                pygame.draw.circle(screen, ring_color, (px, py), radius + 2, 2)

            # Pregnancy ring
            if agent.pregnant:
                pygame.draw.circle(screen, (255, 110, 170), (px, py), radius + 4, 1)

            # Tool ring (white)
            if agent.tool is not None:
                pygame.draw.circle(screen, (255, 255, 255), (px, py), radius + 5, 1)

            # Sex dot
            dot_color = (255, 140, 180) if agent.sex == 'f' else (100, 180, 255)
            pygame.draw.circle(screen, dot_color, (px + radius - 2, py - radius + 2), 2)

            # Action mode icon above agent
            mode = getattr(agent, 'last_action_mode', 'idle')
            icon_color = MODE_COLORS.get(mode)
            if icon_color:
                self._draw_mode_icon(screen, px, py, radius, mode, icon_color)

            # Communication aura
            mv = getattr(agent, 'message_vector', [0,0,0,0]) or [0,0,0,0]
            intensity = min(130, int(sum(abs(v) for v in mv) * 22))
            if intensity > 25:
                aura = pygame.Surface((radius*6, radius*6), pygame.SRCALPHA)
                pygame.draw.circle(aura, (255, 235, 80, intensity),
                                   (radius*3, radius*3), radius + 7, 2)
                screen.blit(aura, (px - radius*3, py - radius*3))

    def _draw_mode_icon(self, screen, px, py, radius, mode, color):
        top = py - radius - 7
        if mode == 'gather':
            pygame.draw.circle(screen, color, (px, top), 3)
        elif mode.startswith('build'):
            pygame.draw.rect(screen, color, (px - 3, top - 3, 6, 6))
        elif mode == 'signal':
            pts = [(px, top - 5), (px - 4, top), (px + 4, top)]
            pygame.draw.polygon(screen, color, pts)
        elif mode == 'attack':
            pygame.draw.line(screen, color, (px - 4, top - 3), (px + 4, top + 3), 2)
            pygame.draw.line(screen, color, (px + 4, top - 3), (px - 4, top + 3), 2)
        elif mode in ('sick', 'wound_fever'):
            pygame.draw.line(screen, color, (px - 3, top - 4), (px + 3, top), 1)
            pygame.draw.line(screen, color, (px - 3, top), (px + 3, top - 4), 1)
        elif mode == 'mate':
            pygame.draw.circle(screen, color, (px, top), 3, 1)
        elif mode == 'forage_herb':
            pygame.draw.circle(screen, color, (px, top), 3)
            pygame.draw.line(screen, color, (px, top + 2), (px, top + 6), 1)
        elif mode == 'experiment':
            pygame.draw.circle(screen, color, (px, top), 4, 1)
            pygame.draw.circle(screen, color, (px, top), 1)

    # ------------------------------------------------------------------
    # Hotspot outlines (disease / pollution / disturbance)
    # ------------------------------------------------------------------
    def draw_hotspots(self, screen, world):
        cp = self.cell_px
        for x, y, cell in world.hotspots():
            if cell['disease'] > 25:
                pygame.draw.circle(screen, (180, 40, 150),
                                   (x*cp + cp//2, y*cp + cp//2), max(3, cp//2), 1)
            if cell['disturbance'] > 30:
                pygame.draw.rect(screen, (245, 165, 60),
                                 (x*cp+2, y*cp+2, cp-4, cp-4), 1)

    # ------------------------------------------------------------------
    # Master draw
    # ------------------------------------------------------------------
    def draw(self, screen, world, agents, stats, tribes, technology):
        self._tick += 1
        screen.fill((12, 12, 16))
        self.draw_world(screen, world)
        self.draw_hotspots(screen, world)
        self.draw_social_links(screen, agents)
        self.draw_agents(screen, agents, tribes)
        # Vignette on world area
        self._draw_vignette(screen)
        draw_dashboard(screen, self.dashboard_x, 0, 300, self.height,
                       stats, tribes, technology, self.font, self.font_bold, self.small,
                       tick=self._tick)
        pygame.display.flip()

    def _draw_vignette(self, screen):
        """Subtle dark border vignette on the world viewport."""
        w = self.dashboard_x
        h = self.height
        v = 28
        for i in range(v):
            alpha = int(120 * (1 - i / v) ** 2)
            for rect, size in [
                ((0, 0, w, i+1), (w, 1)),
                ((0, h-i-1, w, 1), (w, 1)),
                ((0, 0, i+1, h), (1, h)),
                ((w-i-1, 0, 1, h), (1, h)),
            ]:
                s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
                s.fill((0, 0, 0, alpha))
                screen.blit(s, rect[:2])
