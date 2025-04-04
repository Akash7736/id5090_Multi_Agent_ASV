import pygame
import numpy as np
import math
import random

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Velocity Obstacle with Metrics")
clock = pygame.time.Clock()

# Constants
NUM_AGENTS = 100
RADIUS = 300
AGENT_RADIUS = 7
Rf = 50  # Region of influence
GOAL_THRESHOLD = 10  # Distance to goal before teleporting

# Colors
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
BLUE = (50, 100, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)

# Helper function
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
        self.speed = random.uniform(2, 5)
        # self.speed=3
        self.velocity = self.compute_desired_velocity()
        self.path_length = 0  # Track path length for efficiency
        self.collided = False  # Track collisions

    def compute_desired_velocity(self):
        direction = self.goal - self.pos
        norm = np.linalg.norm(direction)
        return (self.speed * direction / norm) if norm > 0 else np.zeros(2)

    def update(self, agents):
        if np.linalg.norm(self.goal - self.pos) < GOAL_THRESHOLD:
            self.pos = self.start.copy()
            self.path_length = 0  # Reset path length
            self.velocity = self.compute_desired_velocity()
            return
        
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
                    avoidance += np.array([desired[1], -desired[0]])
                else:
                    cross = np.cross(desired, rel_pos)
                    if cross < 0:
                        avoidance += np.array([desired[1], -desired[0]])

        if np.linalg.norm(avoidance) > 0:
            avoidance = self.speed * avoidance / np.linalg.norm(avoidance)
            self.velocity = 0.7 * desired + 0.3 * avoidance
        else:
            self.velocity = desired
        
        self.pos += self.velocity
        self.path_length += np.linalg.norm(self.velocity)

    def check_collision(self, agents):
        for other in agents:
            if other.idx != self.idx and np.linalg.norm(self.pos - other.pos) <1.5*AGENT_RADIUS:
                self.collided = True
                return
        self.collided = False

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, self.pos.astype(int), AGENT_RADIUS)
        pygame.draw.circle(screen, GREY, self.goal.astype(int), 4)

# Initialize agents
center = np.array([WIDTH // 2, HEIGHT // 2])
angles = np.linspace(0, 2 * np.pi, NUM_AGENTS, endpoint=False)
starts = [center + RADIUS * np.array([np.cos(a), np.sin(a)]) for a in angles]
goals = [center + RADIUS * np.array([np.cos(a + np.pi), np.sin(a + np.pi)]) for a in angles]
agents = [Agent(i, s, g) for i, (s, g) in enumerate(zip(starts, goals))]

# Metrics calculation
def compute_metrics(agents):
    total_alpha, collision_count = 0, 0
    for agent in agents:
        Lo = np.linalg.norm(agent.start - agent.goal)
        L = agent.path_length if agent.path_length > 0 else Lo  # Avoid division by zero
        total_alpha += Lo / L
        if agent.collided:
            collision_count += 1
    
    avg_alpha = total_alpha / len(agents)
    beta = collision_count / len(agents)
    return avg_alpha, beta

# Main loop
running = True
time_step=0
while running:
    clock.tick(60)
    screen.fill(WHITE)
    time_step+=1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for agent in agents:
        agent.update(agents)
        agent.check_collision(agents)
        agent.draw(screen)
    
    # Compute and display metrics
    avg_alpha, beta = compute_metrics(agents)
    font = pygame.font.SysFont(None, 24)
    text_alpha = font.render(f'Path Efficiency (α): {avg_alpha:.3f}', True, BLACK)
    text_beta = font.render(f'Collision Rate (β): {beta:.3f}', True, BLACK)
    text_time = font.render(f"Time Step: {time_step}", True, RED)
    screen.blit(text_alpha, (20, 20))
    screen.blit(text_beta, (20, 50))
    screen.blit(text_time, (20, 70))
    pygame.display.flip()

pygame.quit()
