from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pygame
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, LineString, Point, Polygon

from tactics2d.controller import (
    ArticulatedPurePursuitController,
    ArticulatedReferenceTrajectory,
    PurePursuitController,
    build_articulated_reference_trajectory,
)
from tactics2d.participant.element import Vehicle, WheelLoader
from tactics2d.participant.trajectory import ArticulatedState, State


class _FallbackPurePursuitController:
    def __init__(self, min_pre_aiming_distance: float = 6.0, target_speed: float = 1.5):
        self.min_pre_aiming_distance = float(min_pre_aiming_distance)
        self.target_speed = float(target_speed)
        self.kp_speed = 0.8

    def step(self, ego_state: State, waypoints: LineString, wheel_base: float = 2.637):
        speed = 0.0 if ego_state.speed is None else float(ego_state.speed)
        preview_distance = max(self.min_pre_aiming_distance, speed * 1.5)
        closest_distance = float(waypoints.project(Point(ego_state.x, ego_state.y)))
        preview_point = waypoints.interpolate(min(closest_distance + preview_distance, waypoints.length))
        dx = preview_point.x - ego_state.x
        dy = preview_point.y - ego_state.y
        alpha = float(np.arctan2(dy, dx) - ego_state.heading)
        alpha = float(np.arctan2(np.sin(alpha), np.cos(alpha)))
        distance = max(np.hypot(dx, dy), 1e-6)
        steering = float(np.arctan2(2.0 * wheel_base * np.sin(alpha), distance))
        accel = float(np.clip((self.target_speed - speed) * self.kp_speed, -2.0, 2.0))
        return steering, accel


class _FallbackArticulatedPurePursuitController:
    def __init__(self, min_pre_aiming_distance: float = 3.0, target_speed: float = 1.2):
        self.min_pre_aiming_distance = float(min_pre_aiming_distance)
        self.target_speed = float(target_speed)
        self.kp_speed = 0.8

    def step(
        self,
        rear_axle_state: State,
        front_axle_state: Tuple[float, float, float],
        waypoints: LineString,
        axle_distance: float = 3.0,
        is_forward: bool = True,
    ):
        speed = 0.0 if rear_axle_state.speed is None else abs(float(rear_axle_state.speed))
        preview_distance = max(self.min_pre_aiming_distance, speed * 1.5)
        ref_x, ref_y = (front_axle_state[0], front_axle_state[1]) if is_forward else rear_axle_state.location
        closest_distance = float(waypoints.project(Point(ref_x, ref_y)))
        signed_preview = preview_distance if is_forward else -preview_distance
        preview_s = float(np.clip(closest_distance + signed_preview, 0.0, waypoints.length))
        preview_point = waypoints.interpolate(preview_s)

        if is_forward:
            base_heading = rear_axle_state.heading
        else:
            base_heading = rear_axle_state.heading + np.pi

        target_angle = np.arctan2(preview_point.y - ref_y, preview_point.x - ref_x)
        heading_error = float(np.arctan2(np.sin(target_angle - base_heading), np.cos(target_angle - base_heading)))
        distance = max(np.hypot(preview_point.x - ref_x, preview_point.y - ref_y), 1e-6)
        curvature = 2.0 * np.sin(heading_error) / distance
        articulation = float(np.arctan(curvature * axle_distance))
        if not is_forward:
            articulation = -articulation

        signed_speed = (1.0 if is_forward else -1.0) * (0.0 if rear_axle_state.speed is None else float(rear_axle_state.speed))
        target_speed = self.target_speed if is_forward else -self.target_speed
        accel = float(np.clip((target_speed - signed_speed) * self.kp_speed, -1.5, 1.5))
        return articulation, accel


def _point_xy(point_like) -> Tuple[float, float]:
    if isinstance(point_like, Point):
        return float(point_like.x), float(point_like.y)
    return float(point_like[0]), float(point_like[1])


def _polyline_heading(path: LineString, distance: float = 0.0) -> float:
    path_length = max(float(path.length), 1e-6)
    left = float(np.clip(distance, 0.0, path_length))
    right = float(np.clip(distance + min(1.0, path_length), 0.0, path_length))
    if abs(right - left) < 1e-9:
        right = float(np.clip(path_length, 0.0, path_length))
        left = float(np.clip(path_length - 1.0, 0.0, path_length))
    start = path.interpolate(left)
    end = path.interpolate(right)
    return float(np.arctan2(end.y - start.y, end.x - start.x))


def _state_from_reference_path(path: LineString, participant_kind: str) -> State:
    start_point = path.interpolate(0.0)
    heading = _polyline_heading(path)
    if participant_kind == "wheel_loader":
        state = ArticulatedState(
            frame=0,
            x=float(start_point.x),
            y=float(start_point.y),
            heading=heading,
            speed=0.0,
            accel=0.0,
            rear_heading=heading,
            steering=0.0,
        )
        return state
    return State(
        frame=0,
        x=float(start_point.x),
        y=float(start_point.y),
        heading=heading,
        speed=0.0,
        accel=0.0,
    )


def _goal_point(map_, target_area=None) -> Optional[Tuple[float, float]]:
    if target_area is not None:
        centroid = target_area.geometry.centroid
        return float(centroid.x), float(centroid.y)
    dest_state = map_.customs.get("dest_state")
    if dest_state is not None:
        return float(dest_state.x), float(dest_state.y)
    if "center_line" in map_.roadlines:
        end_point = Point(map_.roadlines["center_line"].geometry.coords[-1])
        return float(end_point.x), float(end_point.y)
    return None


def _synthesize_reference_path(boundary: Tuple[float, float, float, float], participant_kind: str) -> LineString:
    min_x, max_x, min_y, max_y = boundary
    width = max_x - min_x
    height = max_y - min_y
    if participant_kind == "wheel_loader":
        points = [
            (min_x + 0.10 * width, min_y + 0.10 * height),
            (min_x + 0.35 * width, min_y + 0.15 * height),
            (max_x - 0.10 * width, min_y + 0.45 * height),
            (max_x - 0.25 * width, max_y - 0.18 * height),
            (min_x + 0.12 * width, max_y - 0.12 * height),
        ]
    else:
        points = [
            (min_x + 0.15 * width, min_y + 0.20 * height),
            (min_x + 0.45 * width, min_y + 0.25 * height),
            (max_x - 0.18 * width, min_y + 0.50 * height),
            (max_x - 0.28 * width, max_y - 0.18 * height),
        ]
    return LineString(points)


def _parking_reference_path(start_state: State, target_area, target_heading: Optional[float]) -> LineString:
    start_xy = (float(start_state.x), float(start_state.y))
    target_center = target_area.geometry.centroid
    center_xy = (float(target_center.x), float(target_center.y))
    if target_heading is None:
        return LineString([start_xy, center_xy])

    approach_distance = max(target_area.geometry.length / 10.0, 3.0)
    approach_xy = (
        float(target_center.x - np.cos(target_heading) * approach_distance),
        float(target_center.y - np.sin(target_heading) * approach_distance),
    )
    if np.linalg.norm(np.array(start_xy) - np.array(approach_xy)) < 1.0:
        return LineString([start_xy, center_xy])
    return LineString([start_xy, approach_xy, center_xy])


def _resolve_reference_path(
    map_,
    participant_kind: str,
    start_state: Optional[State],
    target_area=None,
    target_heading: Optional[float] = None,
):
    metadata: Dict[str, Any] = {}

    if participant_kind == "wheel_loader" and map_.customs.get("dest_state") is not None:
        reference = build_articulated_reference_trajectory(
            map_,
            step_interval_ms=200,
            nominal_speed=1.2,
        )
        metadata["reference_trajectory"] = reference
        metadata["reference_path_source"] = "articulated_reference_trajectory"
        return reference.path, metadata

    if "center_line" in map_.roadlines:
        metadata["reference_path_source"] = "roadline:center_line"
        return map_.roadlines["center_line"].geometry, metadata

    if start_state is not None and target_area is not None:
        metadata["reference_path_source"] = "target_area_projection"
        return _parking_reference_path(start_state, target_area, target_heading), metadata

    boundary = map_.boundary
    metadata["reference_path_source"] = "synthetic_boundary_path"
    return _synthesize_reference_path(boundary, participant_kind), metadata


@dataclass
class SceneDescription:
    map_: Any
    participant_kind: str
    start_state: State
    reference_path: LineString
    base_reference_path: LineString
    title: str
    target_area: Any = None
    target_heading: Optional[float] = None
    goal_point: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def adapt_generated_scene(
    map_,
    generator: Any = None,
    generate_result: Any = None,
    title: Optional[str] = None,
) -> SceneDescription:
    start_state = None
    target_area = None
    target_heading = None

    if isinstance(generate_result, tuple):
        if len(generate_result) >= 1:
            start_state = generate_result[0]
        if len(generate_result) >= 2:
            target_area = generate_result[1]
        if len(generate_result) >= 3:
            target_heading = generate_result[2]

    start_state = start_state or map_.customs.get("start_state")
    target_area = target_area or map_.customs.get("target_area")
    target_heading = target_heading if target_heading is not None else map_.customs.get("target_heading")

    generator_name = generator.__class__.__name__ if generator is not None else "MapGenerator"
    participant_kind = "wheel_loader" if isinstance(start_state, ArticulatedState) else "vehicle"

    reference_path, metadata = _resolve_reference_path(
        map_,
        participant_kind=participant_kind,
        start_state=start_state,
        target_area=target_area,
        target_heading=target_heading,
    )
    if start_state is None:
        start_state = _state_from_reference_path(reference_path, participant_kind)

    if participant_kind == "vehicle" and float(getattr(start_state, "heading", 0.0)) == 0.0:
        start_state = State(
            frame=start_state.frame,
            x=start_state.x,
            y=start_state.y,
            heading=_polyline_heading(reference_path),
            speed=0.0 if start_state.speed is None else start_state.speed,
            accel=0.0 if start_state.accel is None else start_state.accel,
        )

    metadata.update(
        {
            "generator_name": generator_name,
            "scenario_type": map_.scenario_type,
            "source_customs": dict(map_.customs),
        }
    )

    return SceneDescription(
        map_=map_,
        participant_kind=participant_kind,
        start_state=start_state,
        reference_path=reference_path,
        base_reference_path=LineString(reference_path.coords),
        title=title or f"{generator_name} - {map_.scenario_type or 'scene'}",
        target_area=target_area,
        target_heading=target_heading,
        goal_point=_goal_point(map_, target_area=target_area),
        metadata=metadata,
    )


def create_default_participant(scene: SceneDescription, id_: int = 0):
    if scene.participant_kind == "wheel_loader":
        participant = WheelLoader(id_=id_, type_="wheel_loader", verify=True)
        participant.verify = False
        participant.add_state(scene.start_state)
        participant.current_articulation = participant.current_state.articulation_angle
        return participant

    participant = Vehicle(id_=id_, type_="medium_car", verify=False)
    participant.load_from_template("medium_car")
    participant.verify = False
    participant._auto_construct_physics_model()
    participant.add_state(scene.start_state)
    return participant


class PygameSceneRenderer:
    def __init__(
        self,
        boundary: Tuple[float, float, float, float],
        window_size: Tuple[int, int] = (960, 960),
        fps: int = 30,
        title: str = "Tactics2D Pygame Runtime",
        padding: int = 40,
        headless: bool = False,
    ):
        pygame.init()
        pygame.font.init()

        self.boundary = boundary
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.fps = int(fps)
        self.padding = int(padding)
        self.headless = bool(headless)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.palette = {
            "background": (244, 243, 239),
            "lane": (44, 50, 65),
            "freespace": (228, 231, 235),
            "obstacle": (171, 180, 186),
            "target": (36, 186, 97, 110),
            "start": (71, 122, 212, 90),
            "reference_base": (136, 150, 160),
            "reference": (255, 51, 102),
            "reference_node": (255, 179, 0),
            "trajectory": (31, 78, 255),
            "roadline": (255, 255, 255),
            "centerline": (245, 184, 36),
            "vehicle": (255, 92, 92),
            "vehicle_front": (255, 154, 59),
            "text": (22, 24, 29),
        }

        if self.headless:
            self.screen = pygame.Surface(self.window_size)
        else:
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption(title)

        width = max(boundary[1] - boundary[0], 1e-6)
        height = max(boundary[3] - boundary[2], 1e-6)
        available_width = max(self.window_size[0] - 2 * self.padding, 1)
        available_height = max(self.window_size[1] - 2 * self.padding, 1)
        self.scale = min(available_width / width, available_height / height)
        self.offset_x = self.padding - boundary[0] * self.scale
        self.offset_y = self.window_size[1] - self.padding + boundary[2] * self.scale

    def close(self):
        pygame.display.quit()
        pygame.font.quit()
        pygame.quit()

    def transform_point(self, x: float, y: float) -> Tuple[int, int]:
        screen_x = int(round(self.offset_x + x * self.scale))
        screen_y = int(round(self.offset_y - y * self.scale))
        return screen_x, screen_y

    def _draw_shape(self, geometry, color, width: int = 0):
        if geometry is None:
            return

        if isinstance(geometry, Polygon):
            points = [self.transform_point(x, y) for x, y in geometry.exterior.coords]
            if len(points) >= 3:
                pygame.draw.polygon(self.screen, color, points, width)
            return

        if isinstance(geometry, LinearRing):
            points = [self.transform_point(x, y) for x, y in geometry.coords]
            if len(points) >= 3:
                pygame.draw.polygon(self.screen, color, points, width)
            return

        if isinstance(geometry, LineString):
            points = [self.transform_point(x, y) for x, y in geometry.coords]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, max(width, 1))

    def _draw_map(self, scene: SceneDescription):
        self.screen.fill(self.palette["background"])

        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for area in scene.map_.areas.values():
            subtype = getattr(area, "subtype", None)
            if subtype == "freespace":
                color = self.palette["freespace"]
                self._draw_shape(area.geometry, color, 0)
            elif subtype in {"obstacle", "wall"}:
                self._draw_shape(area.geometry, self.palette["obstacle"], 0)
            elif subtype and subtype.startswith("target"):
                points = [self.transform_point(x, y) for x, y in area.geometry.exterior.coords]
                if len(points) >= 3:
                    pygame.draw.polygon(overlay, self.palette["target"], points, 0)
                    pygame.draw.polygon(self.screen, self.palette["text"], points, 2)
            elif subtype and subtype.startswith("start"):
                points = [self.transform_point(x, y) for x, y in area.geometry.exterior.coords]
                if len(points) >= 3:
                    pygame.draw.polygon(overlay, self.palette["start"], points, 0)
                    pygame.draw.polygon(self.screen, self.palette["text"], points, 1)

        for lane in scene.map_.lanes.values():
            self._draw_shape(lane.geometry, self.palette["lane"], 0)

        for roadline in scene.map_.roadlines.values():
            subtype = getattr(roadline, "subtype", "") or ""
            color = self.palette["centerline"] if "center" in str(roadline.id_).lower() else self.palette["roadline"]
            if subtype in {"solid", "solid_solid", "wall", "guard_rail"}:
                self._draw_shape(roadline.geometry, color, 3)
            elif subtype in {"dashed", "solid_dashed", "dashed_solid"}:
                points = [self.transform_point(x, y) for x, y in roadline.geometry.coords]
                for start, end in zip(points[::2], points[1::2]):
                    pygame.draw.line(self.screen, color, start, end, 2)
            else:
                self._draw_shape(roadline.geometry, color, 2)

        self.screen.blit(overlay, (0, 0))

    def _draw_reference(self, scene: SceneDescription):
        reference_source = str(scene.metadata.get("reference_path_source", ""))

        if scene.base_reference_path is not None and scene.base_reference_path.length > 0.0:
            base_width = 1 if reference_source == "ppo_primitive_global_plan" else 2
            self._draw_shape(scene.base_reference_path, self.palette["reference_base"], base_width)

        reference_width = 5 if reference_source == "ppo_primitive_global_plan" else 3
        self._draw_shape(scene.reference_path, self.palette["reference"], reference_width)

        if reference_source == "ppo_primitive_global_plan":
            coords = list(scene.reference_path.coords)
            if len(coords) >= 2:
                stride = max(1, len(coords) // 18)
                for x_coord, y_coord in coords[::stride]:
                    pygame.draw.circle(
                        self.screen,
                        self.palette["reference_node"],
                        self.transform_point(x_coord, y_coord),
                        4,
                    )

        if scene.goal_point is not None:
            pygame.draw.circle(self.screen, self.palette["reference"], self.transform_point(*scene.goal_point), 6)

    def _draw_trajectory(self, participant):
        points = [
            self.transform_point(state.x, state.y)
            for state in (participant.trajectory.get_state(frame) for frame in participant.trajectory.frames)
        ]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, self.palette["trajectory"], False, points, 2)

    def _draw_participant(self, participant):
        pose = participant.get_pose()
        if isinstance(pose, tuple):
            rear_pose, front_pose = pose
            self._draw_shape(rear_pose, self.palette["vehicle"], 0)
            self._draw_shape(front_pose, self.palette["vehicle_front"], 0)
        else:
            self._draw_shape(pose, self.palette["vehicle"], 0)

        state = participant.current_state
        center = self.transform_point(state.x, state.y)
        pygame.draw.circle(self.screen, self.palette["text"], center, 4)
        arrow_tip = (
            int(round(center[0] + 16 * np.cos(state.heading))),
            int(round(center[1] - 16 * np.sin(state.heading))),
        )
        pygame.draw.line(self.screen, self.palette["text"], center, arrow_tip, 2)

    def _draw_hud(self, lines: Sequence[str]):
        top = 12
        for line in lines:
            text = self.font.render(line, True, self.palette["text"])
            self.screen.blit(text, (12, top))
            top += text.get_height() + 4

    def render(self, scene: SceneDescription, participant, hud_lines: Optional[Sequence[str]] = None):
        self._draw_map(scene)
        self._draw_reference(scene)
        self._draw_trajectory(participant)
        self._draw_participant(participant)
        if hud_lines:
            self._draw_hud(hud_lines)
        if not self.headless:
            pygame.display.flip()
            self.clock.tick(self.fps)


class SimulationRunner:
    def __init__(
        self,
        scene: SceneDescription,
        participant,
        renderer: Optional[PygameSceneRenderer] = None,
        dt_ms: int = 100,
        max_steps: int = 1500,
        is_forward: bool = True,
        wheel_loader_planner: Optional[Dict[str, Any]] = None,
    ):
        self.scene = scene
        self.participant = participant
        self.renderer = renderer
        self.dt_ms = int(dt_ms)
        self.max_steps = int(max_steps)
        self.is_forward = bool(is_forward)
        self.paused = False
        self.finished = False
        self.current_step = 0
        self.wheel_loader_planner = None
        self.active_reference_trajectory = None
        self.last_planning_error = None
        self.last_planning_result = None
        self.default_reference_trajectory = scene.metadata.get("reference_trajectory")
        self.fallback_controller = None
        self.last_status = None
        self.last_planning_step = -1
        self.pending_primitive_controls = []
        self.pending_primitive_control_index = 0

        if scene.participant_kind == "wheel_loader":
            controller_cls = ArticulatedPurePursuitController or _FallbackArticulatedPurePursuitController
            self.fallback_controller = controller_cls(min_pre_aiming_distance=3.0, target_speed=1.2)
            planner_mode = None if wheel_loader_planner is None else wheel_loader_planner.get("mode")
            if planner_mode == "ppo":
                from .ppo_primitive_bridge import PPOPrimitivePathPlanner

                self.wheel_loader_planner = PPOPrimitivePathPlanner(
                    checkpoint_path=wheel_loader_planner["checkpoint_path"],
                    ppo_root=wheel_loader_planner.get("ppo_root"),
                    control_interval_ms=self.dt_ms,
                    replan_every_steps=wheel_loader_planner.get("replan_every_steps", 1),
                    deterministic=wheel_loader_planner.get("deterministic", True),
                )
                self.controller = None
                self.active_reference_trajectory = None
                self._initialize_planned_reference()
            else:
                self.controller = self.fallback_controller
        else:
            controller_cls = PurePursuitController or _FallbackPurePursuitController
            self.controller = controller_cls(
                min_pre_aiming_distance=6.0,
                target_speed=3.0 if scene.map_.scenario_type == "racing" else 1.5,
            )

    def _initialize_planned_reference(self):
        if self.wheel_loader_planner is None:
            return

        try:
            planning_result = self.wheel_loader_planner.plan(self.scene, self.participant)
        except Exception as exc:
            self.last_planning_error = str(exc)
            self.last_planning_result = None
            self.active_reference_trajectory = self.default_reference_trajectory
            if self.default_reference_trajectory is not None:
                self.scene.reference_path = self.default_reference_trajectory.path
                self.scene.metadata["reference_trajectory"] = self.default_reference_trajectory
                self.scene.metadata["reference_path_source"] = "fallback_articulated_reference_trajectory"
            return

        self.last_planning_error = None
        self.last_planning_result = planning_result
        self.active_reference_trajectory = planning_result.reference
        self.pending_primitive_controls = [tuple(map(float, control)) for control in planning_result.control_actions]
        self.pending_primitive_control_index = 0
        self.scene.reference_path = planning_result.reference.path
        self.scene.metadata["reference_trajectory"] = planning_result.reference
        self.scene.metadata["reference_path_source"] = planning_result.metadata.get(
            "reference_path_source",
            "ppo_primitive_global_plan",
        )
        self.last_planning_step = self.current_step

    def _maybe_replan_reference(self):
        if self.wheel_loader_planner is None:
            return

        if self.last_planning_step < 0:
            self._initialize_planned_reference()
            return

        if (self.current_step - self.last_planning_step) >= int(self.wheel_loader_planner.replan_every_steps):
            self._initialize_planned_reference()

    def _candidate_geometries(self, state) -> Sequence[Polygon]:
        if isinstance(self.participant, WheelLoader):
            articulated_state = self.participant._coerce_state(state)
            front_pose, rear_pose = articulated_state.create_boxes(
                self.participant._front_bbox,
                self.participant._rear_bbox,
                self.participant.hitch_offset,
                self.participant.trailer_length,
            )
            return (Polygon(front_pose), Polygon(rear_pose))

        if isinstance(self.participant, Vehicle):
            transform_matrix = [
                np.cos(state.heading),
                -np.sin(state.heading),
                np.sin(state.heading),
                np.cos(state.heading),
                state.location[0],
                state.location[1],
            ]
            return (Polygon(affine_transform(self.participant._bbox, transform_matrix)),)

        pose = self.participant.get_pose()
        if isinstance(pose, tuple):
            return tuple(Polygon(geom) if not isinstance(geom, Polygon) else geom for geom in pose)
        if isinstance(pose, Polygon):
            return (pose,)
        return (Polygon(pose),)

    def _hits_static_obstacle(self, state) -> bool:
        obstacles = [
            area.geometry
            for area in self.scene.map_.areas.values()
            if getattr(area, "subtype", None) in {"obstacle", "wall"} and getattr(area, "geometry", None) is not None
        ]
        if not obstacles:
            return False

        for candidate in self._candidate_geometries(state):
            if any(candidate.intersects(obstacle) for obstacle in obstacles):
                return True
        return False

    def _is_out_of_bounds(self, state) -> bool:
        min_x, max_x, min_y, max_y = self.scene.map_.boundary
        return bool(state.x < min_x or state.x > max_x or state.y < min_y or state.y > max_y)

    def _commit_state(self, next_state, articulation_angle: Optional[float] = None) -> bool:
        if self._hits_static_obstacle(next_state):
            self.finished = True
            self.last_status = "collision"
            return False
        if self._is_out_of_bounds(next_state):
            self.finished = True
            self.last_status = "out_of_bounds"
            return False

        if isinstance(self.participant, WheelLoader):
            self.participant.add_state(next_state, articulation_angle=articulation_angle)
        else:
            self.participant.add_state(next_state)
        return True

    def _step_wheel_loader_with_primitive_control(self):
        self._maybe_replan_reference()
        if self.last_planning_result is None:
            return False

        if self.pending_primitive_control_index >= len(self.pending_primitive_controls):
            if self.pending_primitive_controls:
                self.pending_primitive_control_index = len(self.pending_primitive_controls) - 1
            else:
                return False

        current_state = self.participant.current_state
        steering_rate, speed = self.pending_primitive_controls[self.pending_primitive_control_index]
        next_state, _, _ = self.participant.physics_model.step(
            state=current_state,
            steering=float(steering_rate),
            speed=float(speed),
            interval=self.dt_ms,
        )
        self.pending_primitive_control_index += 1
        self._commit_state(next_state)
        return True

    def _distance_to_goal(self) -> Optional[float]:
        if self.scene.goal_point is None:
            return None
        state = self.participant.current_state
        return float(np.hypot(state.x - self.scene.goal_point[0], state.y - self.scene.goal_point[1]))

    def _handle_events(self) -> bool:
        if self.renderer is None or self.renderer.headless:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def step_once(self) -> bool:
        if self.finished:
            return False

        if self.scene.participant_kind == "wheel_loader":
            if self.wheel_loader_planner is not None:
                stepped = self._step_wheel_loader_with_primitive_control()
                if not stepped:
                    self.wheel_loader_planner = None
                    self.controller = self.fallback_controller
            if self.wheel_loader_planner is None:
                current_state = self.participant.current_state
                rear_axle_state = self.participant.get_rear_axle_state()
                front_axle_pose = self.participant.get_front_axle_position(
                    articulation_angle=self.participant.current_articulation
                )
                articulation_angle, accel = self.controller.step(
                    rear_axle_state=rear_axle_state,
                    front_axle_state=front_axle_pose,
                    waypoints=self.scene.reference_path,
                    axle_distance=self.participant.axle_distance,
                    is_forward=self.is_forward,
                )
                next_state, _, applied_articulation = self.participant.physics_model.step(
                    state=current_state,
                    accel=accel,
                    articulation_angle=articulation_angle,
                    interval=self.dt_ms,
                    current_articulation=self.participant.current_articulation,
                )
                self._commit_state(next_state, articulation_angle=applied_articulation)
        else:
            current_state = self.participant.current_state
            steering, accel = self.controller.step(
                ego_state=current_state,
                waypoints=self.scene.reference_path,
                wheel_base=self.participant.wheel_base,
            )
            next_state, _, _ = self.participant.physics_model.step(
                current_state,
                accel,
                steering,
                interval=self.dt_ms,
            )
            self._commit_state(next_state)

        self.current_step += 1
        distance_to_goal = self._distance_to_goal()
        if self.current_step >= self.max_steps:
            self.finished = True
            self.last_status = "max_steps"
        elif distance_to_goal is not None and distance_to_goal < 2.0:
            self.finished = True
            self.last_status = "goal_reached"
        return not self.finished

    def snapshot_lines(self) -> Sequence[str]:
        state = self.participant.current_state
        goal_distance = self._distance_to_goal()
        lines = [
            self.scene.title,
            f"step={self.current_step}/{self.max_steps}",
            f"pos=({state.x:.2f}, {state.y:.2f}) heading={np.degrees(state.heading):.1f}deg",
            f"speed={(0.0 if state.speed is None else state.speed):.2f} m/s",
        ]
        if self.scene.participant_kind == "wheel_loader":
            lines.append(
                f"articulation={np.degrees(self.participant.current_articulation):.1f}deg"
            )
            if self.last_planning_result is not None:
                lines.append(f"primitive={self.last_planning_result.primitive_id}")
                primitive_sequence = self.last_planning_result.metadata.get("primitive_sequence", [])
                lines.append(f"planned_primitives={len(primitive_sequence)}")
            if self.last_planning_error:
                lines.append(f"planner_error={self.last_planning_error}")
            lines.append(f"reference={self.scene.metadata.get('reference_path_source', 'unknown')}")
        if goal_distance is not None:
            lines.append(f"goal_distance={goal_distance:.2f} m")
        if self.last_status:
            lines.append(f"status={self.last_status}")
        if self.paused:
            lines.append("paused")
        return lines

    def render(self):
        if self.renderer is not None:
            self.renderer.render(self.scene, self.participant, self.snapshot_lines())

    def run(self):
        running = True
        while running and not self.finished:
            running = self._handle_events()
            if not self.paused:
                self.step_once()
            self.render()
        return self.participant