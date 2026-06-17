import pygame
from artificial_society.visualization.overlays import draw_dashboard, draw_heatmap_legend


class Renderer:
    def __init__(self, width, height, cell_px):
        self.width = width
        self.height = height
        self.cell_px = cell_px
        self.font = pygame.font.SysFont('consolas', 15)
        self.small = pygame.font.SysFont('consolas', 12)
        self.dashboard_x = width - 250

    def draw_world(self, screen, world):
        for y in range(world.height):
            for x in range(world.width):
                color = world.color_at(x, y)
                rect = (x * self.cell_px, y * self.cell_px, self.cell_px, self.cell_px)
                pygame.draw.rect(screen, color, rect)
                cell = world.get_cell(x, y)
                if cell['disturbance'] > 22:
                    alpha = min(125, int(cell['disturbance'] * 1.2))
                    surf = pygame.Surface((self.cell_px, self.cell_px), pygame.SRCALPHA)
                    surf.fill((240, 140, 45, alpha))
                    screen.blit(surf, rect[:2])
                if cell['pollution'] > 24 or cell['disease'] > 18:
                    alpha = min(110, int(cell['pollution'] * 0.8 + cell['disease'] * 0.8))
                    surf = pygame.Surface((self.cell_px, self.cell_px), pygame.SRCALPHA)
                    surf.fill((170, 45, 50, alpha))
                    screen.blit(surf, rect[:2])
                if cell['moisture'] > 70 and world.get_biome(x, y) != 'water':
                    pygame.draw.circle(screen, (120, 185, 255), (x * self.cell_px + self.cell_px // 2, y * self.cell_px + self.cell_px // 2), max(1, self.cell_px // 8))
                if cell['ash'] > 15:
                    pygame.draw.line(screen, (80, 80, 80), (x * self.cell_px + 2, y * self.cell_px + 2), (x * self.cell_px + self.cell_px - 2, y * self.cell_px + self.cell_px - 2), 1)
                if cell['structures']['camp']:
                    pygame.draw.rect(screen, (215, 190, 125), (x * self.cell_px + self.cell_px // 4, y * self.cell_px + self.cell_px // 4, self.cell_px // 2, self.cell_px // 2), 1)
                if cell['structures']['farm']:
                    for i in range(2, self.cell_px, max(3, self.cell_px // 3)):
                        pygame.draw.line(screen, (180, 150, 70), (x * self.cell_px + i, y * self.cell_px + 2), (x * self.cell_px + i, y * self.cell_px + self.cell_px - 2), 1)
                if cell['structures']['well']:
                    pygame.draw.circle(screen, (200, 220, 235), (x * self.cell_px + self.cell_px // 2, y * self.cell_px + self.cell_px // 2), max(2, self.cell_px // 5), 1)

    def draw_social_links(self, screen, agents):
        for agent in agents:
            ax = int(agent.pos[0] * self.cell_px + self.cell_px / 2)
            ay = int(agent.pos[1] * self.cell_px + self.cell_px / 2)
            for other in agents:
                if other.id <= agent.id:
                    continue
                if abs(other.pos[0] - agent.pos[0]) <= 1 and abs(other.pos[1] - agent.pos[1]) <= 1:
                    trust = agent.trust.get(other.id, 0.0)
                    if trust > 0.15:
                        ox = int(other.pos[0] * self.cell_px + self.cell_px / 2)
                        oy = int(other.pos[1] * self.cell_px + self.cell_px / 2)
                        pygame.draw.line(screen, (230, 230, 230), (ax, ay), (ox, oy), 1)

    def draw_agents(self, screen, agents, tribes):
        for agent in agents:
            x = int(agent.pos[0] * self.cell_px + self.cell_px / 2)
            y = int(agent.pos[1] * self.cell_px + self.cell_px / 2)
            color = tribes.color_for(agent.tribe_id) if agent.tribe_id is not None else agent.display_color()
            radius = max(3, self.cell_px // 3)
            pygame.draw.circle(screen, color, (x, y), radius)
            if agent.pregnant:
                pygame.draw.circle(screen, (255, 110, 170), (x, y), radius + 2, 1)
            if agent.tool is not None:
                pygame.draw.circle(screen, (255, 255, 255), (x, y), radius + 4, 1)
            if getattr(agent, 'sick', 0.0) > 12:
                pygame.draw.circle(screen, (205, 55, 55), (x, y), radius + 1, 1)
            if agent.sex == 'f':
                pygame.draw.circle(screen, (255, 140, 180), (x - radius // 2, y - radius // 2), 1)
            else:
                pygame.draw.circle(screen, (100, 180, 255), (x - radius // 2, y - radius // 2), 1)
            mode = getattr(agent, 'last_action_mode', 'idle')
            if mode.startswith('build'):
                pygame.draw.rect(screen, (240, 215, 140), (x - 2, y - radius - 7, 5, 5))
            elif mode == 'signal':
                pts = [(x, y - radius - 6), (x - 4, y - radius - 1), (x + 4, y - radius - 1)]
                pygame.draw.polygon(screen, (255, 245, 120), pts)
            elif mode == 'attack':
                pygame.draw.line(screen, (255, 90, 90), (x - 4, y - radius - 4), (x + 4, y - radius), 2)
            elif mode == 'gather':
                pygame.draw.circle(screen, (140, 255, 140), (x, y - radius - 3), 2)
            elif mode == 'sick':
                pygame.draw.line(screen, (210, 60, 60), (x - 3, y - radius - 6), (x + 3, y - radius), 1)
                pygame.draw.line(screen, (210, 60, 60), (x - 3, y - radius), (x + 3, y - radius - 6), 1)
            mv = agent.message_vector if getattr(agent, 'message_vector', None) else [0, 0, 0, 0]
            intensity = min(140, int(sum(abs(v) for v in mv) * 20))
            if intensity > 20:
                surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255, 240, 90, intensity), (radius * 2, radius * 2), radius + 5, 1)
                screen.blit(surf, (x - radius * 2, y - radius * 2))

    def draw_hotspots(self, screen, world):
        for x, y, cell in world.hotspots():
            cx = x * self.cell_px + self.cell_px // 2
            cy = y * self.cell_px + self.cell_px // 2
            if cell['disease'] > 25:
                pygame.draw.circle(screen, (190, 50, 50), (cx, cy), max(3, self.cell_px // 2), 1)
            if cell['pollution'] > 30:
                pygame.draw.rect(screen, (120, 40, 40), (x * self.cell_px + 1, y * self.cell_px + 1, self.cell_px - 2, self.cell_px - 2), 1)
            if cell['disturbance'] > 30:
                pygame.draw.rect(screen, (245, 170, 70), (x * self.cell_px + 2, y * self.cell_px + 2, self.cell_px - 4, self.cell_px - 4), 1)

    def draw(self, screen, world, agents, stats, tribes, technology):
        screen.fill((18, 18, 20))
        self.draw_world(screen, world)
        self.draw_hotspots(screen, world)
        self.draw_social_links(screen, agents)
        self.draw_agents(screen, agents, tribes)
        draw_dashboard(screen, self.dashboard_x, 0, 250, self.height, stats, tribes, technology, self.font, self.small)
        draw_heatmap_legend(screen, self.dashboard_x + 10, self.height - 90, self.small)
        pygame.display.flip()
