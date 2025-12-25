##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: generate_wheel_loader_scenario.py
# @Description: Generate a scenario for wheel loader testing.
# @Author: Tactics2D Team
# @Version: 1.0.0

import numpy as np
from shapely.geometry import Point, Polygon

from tactics2d.map.element.area import Area
from tactics2d.map.element.map import Map


class WheelLoaderScenarioGenerator:
    """Generate a scenario for wheel loader testing.

    The scenario consists of:
    - A 50m × 50m square boundary
    - Multiple circular obstacles randomly distributed inside

    Attributes:
        width (float): The width of the scenario. Defaults to 50.0 m.
        height (float): The height of the scenario. Defaults to 50.0 m.
        num_obstacles (int): The number of obstacles. Defaults to 8.
        obstacle_radius_range (Tuple[float, float]): The range of obstacle radii. Defaults to (1.0, 3.0) m.
        min_obstacle_spacing (float): The minimum spacing between obstacles. Defaults to 5.0 m.
    """

    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        num_obstacles: int = 8,
        obstacle_radius_range: tuple = (1.0, 3.0),
        min_obstacle_spacing: float = 5.0,
    ):
        """Initialize the scenario generator.

        Args:
            width (float, optional): The width of the scenario. Defaults to 50.0 m.
            height (float, optional): The height of the scenario. Defaults to 50.0 m.
            num_obstacles (int, optional): The number of obstacles. Defaults to 8.
            obstacle_radius_range (tuple, optional): The range of obstacle radii. Defaults to (1.0, 3.0) m.
            min_obstacle_spacing (float, optional): The minimum spacing between obstacles. Defaults to 5.0 m.
        """
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.min_obstacle_spacing = min_obstacle_spacing

    def generate(self, map_: Map, seed: int = None):
        """Generate the scenario and add elements to the map.

        Args:
            map_ (Map): The map to add elements to.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)

        # Create boundary area (50m × 50m square)
        boundary_coords = [
            (0, 0),
            (self.width, 0),
            (self.width, self.height),
            (0, self.height),
        ]
        boundary = Polygon(boundary_coords)
        boundary_area = Area(
            id_="boundary",
            geometry=boundary,
            type_="area",
            subtype="freespace",
        )
        map_.add_area(boundary_area)

        # Generate circular obstacles
        obstacles = []
        max_attempts = 1000
        attempt = 0

        while len(obstacles) < self.num_obstacles and attempt < max_attempts:
            attempt += 1

            # Random position
            x = np.random.uniform(
                self.min_obstacle_spacing, 
                self.width - self.min_obstacle_spacing
            )
            y = np.random.uniform(
                self.min_obstacle_spacing, 
                self.height - self.min_obstacle_spacing
            )

            # Random radius
            radius = np.random.uniform(*self.obstacle_radius_range)

            # Check spacing from existing obstacles
            too_close = False
            for obs in obstacles:
                obs_center = obs.geometry.centroid
                distance = np.sqrt(
                    (x - obs_center.x)**2 + (y - obs_center.y)**2
                )
                if distance < self.min_obstacle_spacing + radius:
                    too_close = True
                    break

            if not too_close:
                # Create circular obstacle
                circle = Point(x, y).buffer(radius)
                obstacle = Area(
                    id_=f"obstacle_{len(obstacles):03d}",
                    geometry=circle,
                    type_="area",
                    subtype="obstacle",
                )
                obstacles.append(obstacle)
                map_.add_area(obstacle)

        if len(obstacles) < self.num_obstacles:
            print(
                f"Warning: Only {len(obstacles)} obstacles were placed "
                f"(requested {self.num_obstacles})"
            )

