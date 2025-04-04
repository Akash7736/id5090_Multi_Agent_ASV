import pygame
import numpy as np
import math
import random

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Velocity Obstacle with COLREGs (Teleporting + Variable Speeds)")
clock = pygame.time.Clock()

# Constants
NUM_AGENTS = 100
RADIUS = 300
AGENT_RADIUS = 7
Rf = 50  # Region of influence
GOAL_THRESHOLD = 10 # Distance to goal before teleporting

# Colors
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
BLUE = (50, 100, 255)

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    det = np.cross(v1, v2)
    return math.degrees(math.atan2(det, dot))

class Agent:
    def __init__(self, idx, start, goal):
        self.idx = idx
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.pos = np.array(start, dtype=float)
        self.speed = random.uniform(1,3)
        # self.speed= 3
        self.velocity = self.compute_desired_velocity()
        self.color = BLUE
        
        self.original_start=self.start
        self.original_goal=self.goal

    def compute_desired_velocity(self):
        direction = self.goal - self.pos
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros(2)
        return self.speed * direction / norm

    def update(self, agents):
        # If agent is near the goal, teleport back to the original start and swap goal
        if np.linalg.norm(self.goal - self.pos) < GOAL_THRESHOLD:
            # Swap start and goal
            # self., self.goal = self.goal.copy(), self.start.copy()
            self.pos = self.start.copy()  # Teleport to start
            self.velocity = self.compute_desired_velocity()
            return  # ← KEY FIX: Don't compute velocity this frame

        desired = self.compute_desired_velocity()
        avoidance = np.zeros(2)

        for other in agents:
            if other.idx == self.idx:
                continue
            rel_pos = other.pos - self.pos
            dist = np.linalg.norm(rel_pos)
            if dist < Rf:
                angle = angle_between(desired, rel_pos)
                if np.abs(angle) < 15:
                    # Head-on: COLREG Rule 1
                    avoidance += np.array([desired[1], -desired[0]])
                else:
                    cross = np.cross(desired, rel_pos)
                    if cross < 0:
                        # Crossing: COLREG Rule 2
                        avoidance += np.array([desired[1], -desired[0]])

        # Blend avoidance with desired direction
        if np.linalg.norm(avoidance) > 0:
            avoidance = self.speed * avoidance / np.linalg.norm(avoidance)
            self.velocity = 0.7 * desired + 0.3 * avoidance
        else:
            self.velocity = desired

        self.pos += self.velocity


    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos.astype(int), AGENT_RADIUS)
        pygame.draw.circle(screen, GREY, self.goal.astype(int), 4)

# Generate circular positions
center = np.array([WIDTH // 2, HEIGHT // 2])
angles = np.linspace(0, 2 * np.pi, NUM_AGENTS, endpoint=False)
starts = [center + RADIUS * np.array([np.cos(a), np.sin(a)]) for a in angles]
goals = [center + RADIUS * np.array([np.cos(a + np.pi), np.sin(a + np.pi)]) for a in angles]

# Create agents
agents = [Agent(i, s, g) for i, (s, g) in enumerate(zip(starts, goals))]

# Main loop
running = True
while running:
    clock.tick(60)
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for agent in agents:
        agent.update(agents)
        agent.draw(screen)

    pygame.display.flip()

pygame.quit()
