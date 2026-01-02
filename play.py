# test_pygame.py
import pygame, sys
pygame.init()
screen = pygame.display.set_mode((480, 360))
pygame.display.set_caption("Pygame test")
clock = pygame.time.Clock()
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
    screen.fill((50, 120, 200))
    pygame.draw.circle(screen, (255, 255, 0), (240, 180), 50)
    pygame.display.flip()
    clock.tick(30)
pygame.quit()