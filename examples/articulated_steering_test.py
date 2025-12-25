##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: articulated_steering_test.py
# @Description: Visualization test for articulated vehicle steering kinematics.
# @Author: Tactics2D Team
# @Version: 1.0.0

import math
import sys
import pygame
import numpy as np

# Constants
L1 = 1.2  # Front section length (axle to hinge)
L2 = 1.2  # Rear section length (hinge to axle)
WIDTH = 1.0  # Vehicle width
LENGTH = 2.0  # Section length (visual)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

class ArticulatedVehicle:
    def __init__(self, x, y, heading_r=0.0, heading_f=0.0):
        self.hinge_x = x
        self.hinge_y = y
        self.theta1 = heading_f  # Front heading
        self.theta2 = heading_r  # Rear heading
        self.phi = self.theta1 - self.theta2  # Articulation angle
        
        # Target phi for animation
        self.target_phi = math.radians(30)
        self.animation_state = 0 # 0: 0->30, 1: 30->0
        self.timer = 0

    def update(self, dt):
        # Simulate motor driving phi
        omega_mag = math.radians(10) # 10 degrees per second
        
        omega = 0.0
        if self.animation_state == 0:
            if self.phi < self.target_phi:
                omega = omega_mag
            else:
                self.animation_state = 1
                self.timer = 0
        elif self.animation_state == 1:
            if self.phi > 0:
                omega = -omega_mag
            else:
                self.animation_state = 0
                self.timer = 0
        
        # Kinematic equations for static steering (v=0)
        # theta1_dot = (l2 * omega) / (l1 * cos(phi) + l2)
        # theta2_dot = -(l2 * omega * cos(phi)) / (l1 * cos(phi) + l2)
        # Note: Using l1 in numerator for theta2_dot based on derivation consistency check,
        # but user said l2. Since l1=l2, it doesn't matter numerically.
        # I will use l2 as requested by user.
        
        denom = L1 * math.cos(self.phi) + L2
        theta1_dot = (L2 * omega) / denom
        theta2_dot = -(L2 * omega * math.cos(self.phi)) / denom
        
        self.theta1 += theta1_dot * dt
        self.theta2 += theta2_dot * dt
        self.phi = self.theta1 - self.theta2
        
        # Update positions based on Hinge fixed at (hinge_x, hinge_y)
        # O1 = O + l1 * [cos(theta1), sin(theta1)]
        # O2 = O - l2 * [cos(theta2), sin(theta2)] (Standard rear axle position)
        # User formula: O2 = O + l2... which implies O2 is ahead of O?
        # Let's stick to standard visual representation: Rear axle is behind hinge.
        # So vector from Hinge to Rear Axle is -[cos(theta2), sin(theta2)] * L2
        pass

    def draw(self, screen):
        # Calculate axle positions
        # Front Axle
        o1_x = self.hinge_x + L1 * math.cos(self.theta1)
        o1_y = self.hinge_y + L1 * math.sin(self.theta1) # Y is down in pygame? No, let's handle transform.
        
        # Rear Axle
        # Assuming rear axle is behind hinge along theta2
        o2_x = self.hinge_x - L2 * math.cos(self.theta2)
        o2_y = self.hinge_y - L2 * math.sin(self.theta2)
        
        # Transform to screen coords
        def to_screen(x, y):
            return int(x), int(SCREEN_HEIGHT - y)

        hx, hy = to_screen(self.hinge_x, self.hinge_y)
        ax1_x, ax1_y = to_screen(o1_x, o1_y)
        ax2_x, ax2_y = to_screen(o2_x, o2_y)
        
        # Draw Hinge
        pygame.draw.circle(screen, RED, (hx, hy), 5)
        
        # Draw Axles
        pygame.draw.circle(screen, BLUE, (ax1_x, ax1_y), 5)
        pygame.draw.circle(screen, BLUE, (ax2_x, ax2_y), 5)
        
        # Draw Bodies (Rectangles)
        # Front Body
        self.draw_rect(screen, o1_x, o1_y, self.theta1, GREEN)
        # Rear Body
        self.draw_rect(screen, o2_x, o2_y, self.theta2, GRAY)
        
        # Draw Text
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Phi: {math.degrees(self.phi):.1f} deg", True, BLACK)
        screen.blit(text, (10, 10))
        text2 = font.render(f"Theta1: {math.degrees(self.theta1):.1f}", True, BLACK)
        screen.blit(text2, (10, 30))
        text3 = font.render(f"Theta2: {math.degrees(self.theta2):.1f}", True, BLACK)
        screen.blit(text3, (10, 50))

    def draw_rect(self, screen, cx, cy, angle, color):
        # Draw a rectangle centered at cx, cy with orientation angle
        # Length 2m, Width 1m. Scale: 1m = 100px
        scale = 100
        l = LENGTH * scale
        w = WIDTH * scale
        
        # Corners relative to center
        corners = [
            (l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)
        ]
        
        # Rotate and translate
        rotated_corners = []
        for x, y in corners:
            rx = x * math.cos(angle) - y * math.sin(angle)
            ry = x * math.sin(angle) + y * math.cos(angle)
            
            # Screen transform
            sx = int(cx * scale + rx)
            sy = int(SCREEN_HEIGHT - (cy * scale + ry)) # Invert Y
            rotated_corners.append((sx, sy))
            
        pygame.draw.polygon(screen, color, rotated_corners, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Articulated Steering Kinematics Test")
    clock = pygame.time.Clock()
    
    # Initialize vehicle at center of screen (in meters)
    # Screen center is (4, 3) meters approx
    vehicle = ArticulatedVehicle(4.0, 3.0)
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        vehicle.update(dt)
        
        screen.fill(WHITE)
        vehicle.draw(screen)
        pygame.display.flip()
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
