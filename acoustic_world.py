import pygame
import numpy as np
import math
import random
import pickle

# ---------------- CONFIG ----------------
W, H = 800, 600
NUM_RAYS = 32
MAX_RANGE = 200
DT = 0.05                   # Recommend 0.1
SAVE_DATA = True
CLOCK_TICK = 10             # Recommend 30

# --------------------------------------

class Obstacle:
    def __init__(self):
        self.x = random.uniform(100, W-100)
        self.y = random.uniform(100, H-100)
        self.vx = random.uniform(-50, 50)
        self.vy = random.uniform(-50, 50)
        self.r = random.uniform(20, 40)

    def update(self):
        self.x += self.vx * DT
        self.y += self.vy * DT

        if self.x < self.r or self.x > W-self.r: self.vx *= -1
        if self.y < self.r or self.y > H-self.r: self.vy *= -1

class Agent:
    def __init__(self):
        self.x = W/2
        self.y = H/2
        self.theta = 0
        self.v = 0

    def step(self, steer, accel):
        self.theta += steer * DT
        self.v += accel * DT
        self.v = np.clip(self.v, -100, 200)

        self.x += self.v * math.cos(self.theta) * DT
        self.y += self.v * math.sin(self.theta) * DT

        self.x = np.clip(self.x, 0, W)
        self.y = np.clip(self.y, 0, H)

def raycast(agent, obstacles):
    angles = np.linspace(-np.pi/2, np.pi/2, NUM_RAYS)
    readings = []

    for a in angles:
        ray_theta = agent.theta + a
        min_dist = MAX_RANGE

        for d in np.linspace(0, MAX_RANGE, 100):
            rx = agent.x + d * math.cos(ray_theta)
            ry = agent.y + d * math.sin(ray_theta)

            for ob in obstacles:
                if (rx-ob.x)**2 + (ry-ob.y)**2 < ob.r**2:
                    min_dist = d
                    break
            if min_dist < MAX_RANGE:
                break

        readings.append(min_dist / MAX_RANGE)
    return np.array(readings)

# --------------------------------------

pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

agent = Agent()
obstacles = [Obstacle() for _ in range(5)]

dataset = []

running = True
while running:
    clock.tick(CLOCK_TICK)
    screen.fill((0,0,0))

    # ----- Keyboard control -----
    keys = pygame.key.get_pressed()
    steer = 0
    accel = 0
    if keys[pygame.K_LEFT]: steer = -2
    if keys[pygame.K_RIGHT]: steer = 2
    if keys[pygame.K_UP]: accel = 50
    if keys[pygame.K_DOWN]: accel = -50

    # Quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ----- Update -----
    agent.step(steer, accel)
    for ob in obstacles:
        ob.update()

    sonar = raycast(agent, obstacles)

    # Save expert data
    if SAVE_DATA:
        obs = np.concatenate([sonar, [agent.x/W, agent.y/H, agent.theta, agent.v/200]])
        act = np.array([steer, accel])
        dataset.append((obs, act))
        print(sonar)

    # ----- Draw agent -----
    pygame.draw.circle(screen, (0,255,0), (int(agent.x), int(agent.y)), 10)
    hx = agent.x + 20 * math.cos(agent.theta)
    hy = agent.y + 20 * math.sin(agent.theta)
    pygame.draw.line(screen, (255,0,0), (agent.x, agent.y), (hx, hy), 2)

    # ----- Draw obstacles -----
    for ob in obstacles:
        pygame.draw.circle(screen, (0,0,255), (int(ob.x), int(ob.y)), int(ob.r))

    # ----- Draw sonar rays -----
    angles = np.linspace(-np.pi/2, np.pi/2, NUM_RAYS)
    for d, a in zip(sonar, angles):
        dist = d * MAX_RANGE
        rx = agent.x + dist * math.cos(agent.theta + a)
        ry = agent.y + dist * math.sin(agent.theta + a)
        pygame.draw.line(screen, (255,255,0), (agent.x, agent.y), (rx, ry), 1)

    pygame.display.flip()

pygame.quit()

# Save dataset
if SAVE_DATA:
    with open("expert_demos.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} samples")
