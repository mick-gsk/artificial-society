import pygame
from artificial_society.simulation import Simulation


def main():
    pygame.init()
    sim = Simulation(width=1200, height=800, grid_w=60, grid_h=40, initial_population=36)
    sim.run()
    pygame.quit()


if __name__ == '__main__':
    main()
