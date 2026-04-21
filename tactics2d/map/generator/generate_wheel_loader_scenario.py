##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: generate_wheel_loader_scenario.py
# @Description: Generate wheel loader scenarios with either a legacy random map or PPO-aligned geometry.
# @Author: Tactics2D Team
# @Version: 1.0.0

import numpy as np
from shapely.geometry import Point, Polygon

from tactics2d.map.element.area import Area
from tactics2d.map.element.map import Map
from tactics2d.utils.ppo_articulated_defaults import PPO_DEFAULT_MAP_LEVEL

from .generate_ppo_parking_map import PPOParkingMapGenerator


class _LegacyWheelLoaderScenarioGenerator:
    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        num_obstacles: int = 8,
        obstacle_radius_range: tuple = (1.0, 3.0),
        min_obstacle_spacing: float = 5.0,
    ):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.min_obstacle_spacing = min_obstacle_spacing

    def generate(self, map_: Map, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        boundary = Polygon(
            [
                (0, 0),
                (self.width, 0),
                (self.width, self.height),
                (0, self.height),
            ]
        )
        map_.reset()
        map_.add_area(
            Area(
                id_=1,
                geometry=boundary,
                type_="area",
                subtype="freespace",
            )
        )

        obstacles = []
        max_attempts = 1000
        attempt = 0
        area_id = 2

        while len(obstacles) < self.num_obstacles and attempt < max_attempts:
            attempt += 1
            x = np.random.uniform(self.min_obstacle_spacing, self.width - self.min_obstacle_spacing)
            y = np.random.uniform(self.min_obstacle_spacing, self.height - self.min_obstacle_spacing)
            radius = np.random.uniform(*self.obstacle_radius_range)

            too_close = False
            for obstacle in obstacles:
                center = obstacle.geometry.centroid
                distance = np.hypot(x - center.x, y - center.y)
                if distance < self.min_obstacle_spacing + radius:
                    too_close = True
                    break

            if too_close:
                continue

            obstacle = Area(
                id_=area_id,
                geometry=Point(x, y).buffer(radius),
                type_="area",
                subtype="obstacle",
            )
            area_id += 1
            obstacles.append(obstacle)
            map_.add_area(obstacle)

        map_.set_boundary((0.0, self.width, 0.0, self.height))
        return None


class WheelLoaderScenarioGenerator:
    """Generate wheel loader scenes.

    The default backend follows PPO_articulated_vehicle's parking_map_normal geometry.
    Use ``backend="legacy"`` to keep the previous random circular-obstacle scene.
    """

    def __init__(
        self,
        backend: str = "ppo",
        scene_type: str = "navigation",
        map_level: str = PPO_DEFAULT_MAP_LEVEL,
        ppo_root: str = None,
        width: float = 50.0,
        height: float = 50.0,
        num_obstacles: int = 8,
        obstacle_radius_range: tuple = (1.0, 3.0),
        min_obstacle_spacing: float = 5.0,
    ):
        self.backend = backend
        self.scene_type = scene_type
        self.map_level = map_level

        if self.backend == "legacy":
            self._generator = _LegacyWheelLoaderScenarioGenerator(
                width=width,
                height=height,
                num_obstacles=num_obstacles,
                obstacle_radius_range=obstacle_radius_range,
                min_obstacle_spacing=min_obstacle_spacing,
            )
        elif self.backend == "ppo":
            self._generator = PPOParkingMapGenerator(
                scene_type=scene_type,
                map_level=map_level,
                ppo_root=ppo_root,
            )
        else:
            raise ValueError("backend must be either 'ppo' or 'legacy'.")

        self.width = getattr(self._generator, "width", width)
        self.height = getattr(self._generator, "height", height)
        self.num_obstacles = getattr(self._generator, "num_obstacles", num_obstacles)

    def generate(self, map_: Map, seed: int = None):
        result = self._generator.generate(map_, seed=seed)
        boundary = map_.boundary
        self.width = float(boundary[1] - boundary[0])
        self.height = float(boundary[3] - boundary[2])
        self.num_obstacles = len(
            [area for area in map_.areas.values() if getattr(area, "subtype", None) == "obstacle"]
        )
        return result

