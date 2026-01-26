import pygame
from acoustic_world import AcousticWorld
import numpy as np

CLOCK_RATE = 30

env = AcousticWorld(render=True)
obs = env.reset()

pygame.init()
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(30)  # slow down to 10 Hz
    steer, accel = 0, 0

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        steer = -2
    if keys[pygame.K_RIGHT]:
        steer = 2
    if keys[pygame.K_UP]:
        accel = 50
    if keys[pygame.K_DOWN]:
        accel = -50

    # Quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    obs, collision = env.step(np.array([steer, accel]))
    if collision:
        print("Collision!")
        obs = env.reset()


