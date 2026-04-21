import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from shapely.geometry import GeometryCollection, LinearRing, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from tactics2d.map.element import Area, Map
from tactics2d.participant.trajectory import ArticulatedState
from tactics2d.utils.ppo_articulated_defaults import (
    PPO_DEFAULT_MAP_LEVEL,
    PPO_FRONT_OVERHANG,
    PPO_HITCH_OFFSET,
    PPO_NAVIGATION_BOUNDARY,
    PPO_REAR_OVERHANG,
    PPO_SCENE_MARGIN,
    PPO_START_AREA_COLOR,
    PPO_START_DEST_ARTICULATION_RANGE,
    PPO_TARGET_AREA_COLOR,
    PPO_TRAILER_LENGTH,
    PPO_WIDTH,
    build_front_vehicle_box,
    build_rear_vehicle_box,
)


_PPO_MODULE_CACHE = {}


def _discover_ppo_root(explicit_root: Optional[str] = None) -> Path:
    candidate_roots = []
    if explicit_root:
        candidate_roots.append(Path(explicit_root).expanduser())

    for parent in Path(__file__).resolve().parents:
        candidate_roots.append(parent / "PPO_articulated_vehicle")

    for candidate in candidate_roots:
        if (candidate / "src" / "env" / "parking_map_normal.py").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Cannot locate PPO_articulated_vehicle. Provide ppo_root explicitly or place the repo next to tactics2d-articulated_Vehicle."
    )


def _load_ppo_parking_module(ppo_root: Optional[str] = None):
    root = _discover_ppo_root(ppo_root)
    cache_key = str(root)
    if cache_key in _PPO_MODULE_CACHE:
        return _PPO_MODULE_CACHE[cache_key]

    src_root = root / "src"
    inserted = False
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
        inserted = True

    original_sdl_video_driver = os.environ.get("SDL_VIDEODRIVER")

    try:
        module = importlib.import_module("env.parking_map_normal")
    finally:
        if original_sdl_video_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = original_sdl_video_driver
        if inserted:
            sys.path.pop(0)

    _PPO_MODULE_CACHE[cache_key] = module
    return module


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        polygon = geometry.buffer(0)
        if not polygon.is_empty:
            yield polygon
        return

    if isinstance(geometry, LinearRing):
        polygon = Polygon(geometry).buffer(0)
        if not polygon.is_empty:
            yield polygon
        return

    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            yield from _iter_polygons(poly)
        return

    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            yield from _iter_polygons(geom)
        return

    buffered = geometry.buffer(0)
    yield from _iter_polygons(buffered)


class PPOParkingMapGenerator:
    """Generate tactics2d maps from PPO_articulated_vehicle scene geometry.

    The underlying geometry source remains PPO's parking_map_normal implementation,
    while the returned objects are converted into tactics2d map areas and
    articulated states.
    """

    _supported_scene_types = {"bay", "parallel", "navigation"}

    def __init__(
        self,
        scene_type: str = "navigation",
        map_level: str = PPO_DEFAULT_MAP_LEVEL,
        ppo_root: str = None,
        sample_start_dest_articulation: bool = None,
        articulation_range: Tuple[float, float] = PPO_START_DEST_ARTICULATION_RANGE,
        width: float = PPO_WIDTH,
        hitch_offset: float = PPO_HITCH_OFFSET,
        trailer_length: float = PPO_TRAILER_LENGTH,
        front_overhang: float = PPO_FRONT_OVERHANG,
        rear_overhang: float = PPO_REAR_OVERHANG,
    ):
        if scene_type not in self._supported_scene_types:
            raise ValueError(
                f"Unsupported scene_type {scene_type}. Expected one of {sorted(self._supported_scene_types)}."
            )

        self.scene_type = scene_type
        self.map_level = map_level
        self.ppo_root = None if ppo_root is None else str(Path(ppo_root).expanduser())
        self.sample_start_dest_articulation = (
            self.scene_type == "navigation"
            if sample_start_dest_articulation is None
            else bool(sample_start_dest_articulation)
        )
        self.articulation_range = articulation_range

        self.width = float(width)
        self.height = float(PPO_NAVIGATION_BOUNDARY[3] - PPO_NAVIGATION_BOUNDARY[2])
        self.hitch_offset = float(hitch_offset)
        self.trailer_length = float(trailer_length)
        self.front_overhang = float(front_overhang)
        self.rear_overhang = float(rear_overhang)
        self.num_obstacles = 0
        self.last_scene_meta = None

        self._front_vehicle_box = build_front_vehicle_box(
            width=self.width,
            hitch_offset=self.hitch_offset,
            front_overhang=self.front_overhang,
        )
        self._rear_vehicle_box = build_rear_vehicle_box(
            width=self.width,
            trailer_length=self.trailer_length,
            rear_overhang=self.rear_overhang,
        )

    def _get_source_module(self):
        return _load_ppo_parking_module(self.ppo_root)

    def _generate_source_scene(self):
        module = self._get_source_module()

        if self.scene_type == "bay":
            start_pose, dest_pose, obstacles = module.generate_bay_parking_case(self.map_level)
            scene_meta = None
        elif self.scene_type == "parallel":
            start_pose, dest_pose, obstacles = module.generate_parallel_parking_case(self.map_level)
            scene_meta = None
        else:
            start_pose, dest_pose, obstacles, scene_meta = module.generate_navigation_case(
                self.map_level, return_regions=True
            )

        return start_pose, dest_pose, obstacles, scene_meta

    def _sample_articulation(self) -> float:
        if not self.sample_start_dest_articulation:
            return 0.0
        module = self._get_source_module()
        if hasattr(module, "random_uniform_num"):
            return float(module.random_uniform_num(*self.articulation_range))
        return float(np.random.uniform(*self.articulation_range))

    def _build_state(self, pose, articulation: float) -> ArticulatedState:
        state = ArticulatedState(
            frame=0,
            x=float(pose[0]),
            y=float(pose[1]),
            heading=float(pose[2]),
            speed=0.0,
            accel=0.0,
            steering=0.0,
            rear_heading=float(pose[2]) - float(articulation),
        )
        state.update_trailer_loc(self.hitch_offset, self.trailer_length)
        return state

    def _build_vehicle_polygons(self, state: ArticulatedState):
        front_ring, rear_ring = state.create_boxes(
            self._front_vehicle_box,
            self._rear_vehicle_box,
            self.hitch_offset,
            self.trailer_length,
        )
        return Polygon(front_ring), Polygon(rear_ring)

    def _compute_boundary(self, polygons) -> Tuple[float, float, float, float]:
        if self.scene_type == "navigation":
            return PPO_NAVIGATION_BOUNDARY

        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        for polygon in polygons:
            px_min, py_min, px_max, py_max = polygon.bounds
            min_x = min(min_x, px_min)
            max_x = max(max_x, px_max)
            min_y = min(min_y, py_min)
            max_y = max(max_y, py_max)

        return (
            float(np.floor(min_x - PPO_SCENE_MARGIN)),
            float(np.ceil(max_x + PPO_SCENE_MARGIN)),
            float(np.floor(min_y - PPO_SCENE_MARGIN)),
            float(np.ceil(max_y + PPO_SCENE_MARGIN)),
        )

    def generate(self, map_: Map, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        map_.reset()
        if map_.name is None:
            map_.name = f"ppo_{self.scene_type}_scene"
        if map_.scenario_type is None:
            map_.scenario_type = self.scene_type

        start_pose, dest_pose, obstacles, scene_meta = self._generate_source_scene()
        start_state = self._build_state(start_pose, self._sample_articulation())
        dest_state = self._build_state(dest_pose, self._sample_articulation())
        start_front_poly, start_rear_poly = self._build_vehicle_polygons(start_state)
        target_front_poly, target_rear_poly = self._build_vehicle_polygons(dest_state)

        obstacle_polygons = []
        for obstacle in obstacles:
            obstacle_polygons.extend(list(_iter_polygons(obstacle)))

        scene_polygons = [start_front_poly, start_rear_poly, target_front_poly, target_rear_poly]
        scene_polygons.extend(obstacle_polygons)
        boundary = self._compute_boundary(scene_polygons)
        world_polygon = box(boundary[0], boundary[2], boundary[1], boundary[3])
        obstacle_union = unary_union(obstacle_polygons) if obstacle_polygons else GeometryCollection()
        freespace_geometry = world_polygon.difference(obstacle_union).buffer(0)

        area_id = 1
        for polygon in _iter_polygons(freespace_geometry):
            map_.add_area(
                Area(
                    id_=area_id,
                    geometry=polygon,
                    type_="area",
                    subtype="freespace",
                )
            )
            area_id += 1

        start_front_area = Area(
            id_=area_id,
            geometry=start_front_poly,
            type_="area",
            subtype="start_area",
            color=PPO_START_AREA_COLOR,
        )
        map_.add_area(start_front_area)
        area_id += 1
        map_.add_area(
            Area(
                id_=area_id,
                geometry=start_rear_poly,
                type_="area",
                subtype="start_rear_area",
                color=PPO_START_AREA_COLOR,
            )
        )
        area_id += 1

        target_area = Area(
            id_=area_id,
            geometry=target_front_poly,
            type_="area",
            subtype="target_area",
            color=PPO_TARGET_AREA_COLOR,
        )
        map_.add_area(target_area)
        area_id += 1
        map_.add_area(
            Area(
                id_=area_id,
                geometry=target_rear_poly,
                type_="area",
                subtype="target_rear_area",
                color=PPO_TARGET_AREA_COLOR,
            )
        )
        area_id += 1

        for polygon in obstacle_polygons:
            map_.add_area(
                Area(
                    id_=area_id,
                    geometry=polygon,
                    type_="area",
                    subtype="obstacle",
                )
            )
            area_id += 1

        map_.set_boundary(boundary)
        map_.customs["source"] = "ppo_articulated_vehicle"
        map_.customs["scene_type"] = self.scene_type
        map_.customs["map_level"] = self.map_level
        map_.customs["start_state"] = start_state
        map_.customs["dest_state"] = dest_state
        map_.customs["target_area"] = target_area
        map_.customs["target_heading"] = float(dest_state.heading)
        map_.customs["start_boxes"] = (start_front_poly, start_rear_poly)
        map_.customs["target_boxes"] = (target_front_poly, target_rear_poly)
        map_.customs["scene_meta"] = scene_meta
        map_.customs["source_start_pose"] = tuple(map(float, start_pose))
        map_.customs["source_dest_pose"] = tuple(map(float, dest_pose))

        self.last_scene_meta = scene_meta
        self.num_obstacles = len(obstacle_polygons)
        self.width = float(boundary[1] - boundary[0])
        self.height = float(boundary[3] - boundary[2])

        return start_state, target_area, float(dest_state.heading)