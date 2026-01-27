import pygame
import numpy as np
import math
import random
import pickle

WIDTH, HEIGHT = 800, 600
NUM_OBSTACLES = 7
NUM_RAYS = 100
MAX_RANGE = 200
DT = 0.1
BG_COLOR = (37, 57, 69)
OBSTACLE_COLOR = (122, 13, 13)
AGENT_COLOR = (92, 12, 92)
RAY_COLOR = (0,100,255)
AGENT_RADIUS = 8

class Agent:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.theta = 0.0
        self.v = 0.0
        self.sonar = SonarSensor(self)

    def update(self, steer, accel):
        self.theta += steer * DT
        self.v += accel * DT
        self.v = np.clip(self.v, -100, 200)

        self.x += self.v * math.cos(self.theta) * DT
        self.y += self.v * math.sin(self.theta) * DT

        self.x = np.clip(self.x, 0, WIDTH)
        self.y = np.clip(self.y, 0, HEIGHT)


class Obstacle:
    def __init__(self):
        self.x = np.random.randint(0, WIDTH)
        self.y = np.random.randint(0, HEIGHT)
        self.vx = np.random.uniform(-50, 50)
        self.vy = np.random.uniform(-50, 50)
        self.r = random.uniform(20, 40)

    def update(self):
        self.x += self.vx * DT
        self.y += self.vy * DT
        if self.x < self.r or self.x > WIDTH-self.r: self.vx *= -1
        if self.y < self.r or self.y > HEIGHT-self.r: self.vy *= -1

class SonarSensor:
    def __init__(self, agent, n_rays=16, max_dist=200):
        self.agent = agent
        self.n_rays = n_rays
        self.max_dist = max_dist
        self.last_rays = []

    def sense(self, obstacles):
        obs = np.zeros(self.n_rays)
        self.last_rays = []

        base_theta = self.agent.theta
        angles = np.linspace(-np.pi/2, np.pi/2, self.n_rays)

        for i, a in enumerate(angles):
            angle = base_theta + a
            d = self._raycast(angle, obstacles)
            obs[i] = d
            self.last_rays.append((angle, d))
        return obs


    def _raycast(self, angle, obstacles):
        x, y = self.agent.x, self.agent.y
        dx, dy = np.cos(angle), np.sin(angle)

        for d in np.linspace(0, self.max_dist, 50):
            px = x + dx * d
            py = y + dy * d
            for o in obstacles:
                if np.hypot(px - o.x, py - o.y) < o.r:
                    return d / self.max_dist
        return 1.0
        

class AcousticWorld:
    def __init__(self, n_obstacles=NUM_OBSTACLES, render=False):
        self.render_mode = render
        self.n_obstacles = n_obstacles
        self.reset()

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # ------------------------
    # Reset environment
    # ------------------------
    def reset(self):
        self.agent = Agent()
        self.obstacles = [Obstacle() for _ in range(self.n_obstacles)]
        self.t = 0
        return self._get_obs()

    # ------------------------
    # Step environment
    # action = [steer, accel]
    # ------------------------
    def step(self, action):
        steer, accel = action

        self.agent.update(steer, accel)

        # update obstacles
        for o in self.obstacles:
            o.update()

        # collision check
        collision = self._check_collision()

        obs = self._get_obs()
        self.t += DT

        if self.render_mode:
            self.render()

        return obs, collision

    # ------------------------
    def _check_collision(self):
        for o in self.obstacles:
            d = np.hypot(self.agent.x - o.x, self.agent.y - o.y)
            if d < (AGENT_RADIUS + o.r):
                return True
        return False

    def _get_obs(self):
        sonar_measurements = self.agent.sonar.sense(self.obstacles)

        return np.concatenate([sonar_measurements, [self.agent.x/WIDTH, self.agent.y/HEIGHT, self.agent.theta, self.agent.v/200]])


    # ------------------------
    # Render
    # ------------------------
    def render(self):
        self.screen.fill(BG_COLOR)

        # agent
        pygame.draw.circle(self.screen, AGENT_COLOR, (int(self.agent.x), int(self.agent.y)), AGENT_RADIUS)
        hx = self.agent.x + 20 * math.cos(self.agent.theta)
        hy = self.agent.y + 20 * math.sin(self.agent.theta)
        pygame.draw.line(self.screen, (255,0,0), (self.agent.x, self.agent.y), (hx, hy), 2)

        # obstacles
        for o in self.obstacles:
            pygame.draw.circle(self.screen, OBSTACLE_COLOR, (int(o.x), int(o.y)), int(o.r))

            # sonar rays
        for angle, d in self.agent.sonar.last_rays:
            dx = np.cos(angle) * d * self.agent.sonar.max_dist
            dy = np.sin(angle) * d * self.agent.sonar.max_dist
            pygame.draw.line(self.screen, RAY_COLOR, (self.agent.x, self.agent.y), (self.agent.x + dx, self.agent.y + dy), 1)

        pygame.display.flip()
