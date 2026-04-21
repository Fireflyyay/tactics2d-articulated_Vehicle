from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:
    minimize = None

from tactics2d.participant.trajectory import ArticulatedState
from tactics2d.participant.trajectory.articulated_state import wrap_angle
from tactics2d.utils.ppo_articulated_defaults import (
    PPO_HITCH_OFFSET,
    PPO_TRAILER_LENGTH,
    PPO_WIDTH,
)


HAS_SCIPY = minimize is not None


def _angle_difference(lhs: float, rhs: float) -> float:
    return wrap_angle(float(lhs) - float(rhs))


def _interpolate_angle(start: float, end: float, weight: float) -> float:
    weight = float(np.clip(weight, 0.0, 1.0))
    return wrap_angle(float(start) + weight * _angle_difference(end, start))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _coerce_xy(point_like) -> Tuple[float, float]:
    if isinstance(point_like, Point):
        return (float(point_like.x), float(point_like.y))

    if hasattr(point_like, "__len__") and len(point_like) >= 2:
        return (float(point_like[0]), float(point_like[1]))

    raise TypeError(f"Cannot convert {point_like!r} to an (x, y) point.")


def _dedupe_points(points: Sequence[Tuple[float, float]], tolerance: float = 1e-6):
    deduped = []
    for point in points:
        xy = _coerce_xy(point)
        if deduped and np.linalg.norm(np.array(xy) - np.array(deduped[-1])) <= tolerance:
            continue
        deduped.append(xy)
    return deduped


def _coerce_guidance_points(raw_points) -> List[Tuple[float, float]]:
    if raw_points is None:
        return []

    return _dedupe_points([_coerce_xy(point) for point in raw_points])


def _support_anchor(edge_meta: Optional[dict], inward_offset: float) -> Optional[Tuple[float, float]]:
    if edge_meta is None:
        return None

    p1 = np.asarray(edge_meta.get("p1"), dtype=float)
    p2 = np.asarray(edge_meta.get("p2"), dtype=float)
    if p1.shape != (2,) or p2.shape != (2,):
        return None

    segment = p2 - p1
    segment_length = float(edge_meta.get("len", np.linalg.norm(segment)))
    if segment_length <= 1e-6:
        return None

    s_value = float(np.clip(edge_meta.get("s", segment_length * 0.5), 0.0, segment_length))
    edge_point = p1 + segment * (s_value / segment_length)

    inward = np.asarray(edge_meta.get("inward", [0.0, 0.0]), dtype=float)
    inward_norm = float(np.linalg.norm(inward))
    if inward_norm > 1e-6:
        edge_point = edge_point + inward * (float(inward_offset) / inward_norm)

    return (float(edge_point[0]), float(edge_point[1]))


def _sample_heading(path: LineString, distance: float, epsilon: float) -> float:
    path_length = max(float(path.length), 1e-6)
    left = max(0.0, float(distance) - epsilon)
    right = min(path_length, float(distance) + epsilon)
    if abs(right - left) <= 1e-9:
        left = max(0.0, float(distance) - epsilon * 0.5)
        right = min(path_length, float(distance) + epsilon * 0.5 + 1e-6)

    p_left = path.interpolate(left)
    p_right = path.interpolate(right)
    return float(np.arctan2(p_right.y - p_left.y, p_right.x - p_left.x))


def _largest_polygon(geometry):
    if geometry is None or geometry.is_empty:
        return None

    if isinstance(geometry, Polygon):
        return geometry

    if isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
        return max(polygons, key=lambda polygon: float(polygon.area)) if polygons else None

    buffered = geometry.buffer(0)
    if isinstance(buffered, Polygon):
        return buffered
    if isinstance(buffered, MultiPolygon):
        polygons = list(buffered.geoms)
        return max(polygons, key=lambda polygon: float(polygon.area)) if polygons else None
    return None


def _collect_obstacle_geometry(map_) -> Optional[Polygon]:
    obstacle_geometries = [
        area.geometry
        for area in map_.areas.values()
        if getattr(area, "subtype", None) == "obstacle" and area.geometry is not None
    ]
    if not obstacle_geometries:
        return None
    return unary_union(obstacle_geometries).buffer(0)


@dataclass
class ArticulatedReferenceTrajectory:
    states: List[ArticulatedState]
    path: LineString
    anchors: List[Tuple[float, float]]
    guidance_points: List[Tuple[float, float]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self._positions = np.array([(state.x, state.y) for state in self.states], dtype=float)

    def __len__(self):
        return len(self.states)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    def closest_index(self, state_or_xy) -> int:
        if len(self.states) == 0:
            raise ValueError("Reference trajectory is empty.")

        if isinstance(state_or_xy, ArticulatedState):
            query = np.array([state_or_xy.x, state_or_xy.y], dtype=float)
        elif hasattr(state_or_xy, "x") and hasattr(state_or_xy, "y"):
            query = np.array([float(state_or_xy.x), float(state_or_xy.y)], dtype=float)
        else:
            query = np.array(_coerce_xy(state_or_xy), dtype=float)

        distances = np.linalg.norm(self._positions - query[None, :], axis=1)
        return int(np.argmin(distances))

    def window(self, start_idx: int, length: int) -> List[ArticulatedState]:
        if len(self.states) == 0:
            return []

        start_idx = int(np.clip(start_idx, 0, len(self.states) - 1))
        end_idx = min(len(self.states), start_idx + int(length))
        window = list(self.states[start_idx:end_idx])
        if len(window) < length:
            window.extend([self.states[-1]] * (length - len(window)))
        return window


@dataclass
class MPCSolveResult:
    steering_rate: float
    speed: float
    predicted_states: List[ArticulatedState]
    reference_states: List[ArticulatedState]
    reference_index: int
    objective: float
    success: bool
    iterations: int
    message: str


def build_articulated_reference_trajectory(
    map_,
    num_samples: Optional[int] = None,
    sample_spacing: float = 1.0,
    step_interval_ms: int = 200,
    nominal_speed: float = 1.0,
    hitch_offset: float = PPO_HITCH_OFFSET,
    trailer_length: float = PPO_TRAILER_LENGTH,
    support_edge_offset: float = PPO_WIDTH * 0.8,
) -> ArticulatedReferenceTrajectory:
    customs = map_.customs
    start_state = customs.get("start_state")
    dest_state = customs.get("dest_state")
    target_boxes = customs.get("target_boxes")
    scene_meta = customs.get("scene_meta") or {}

    if start_state is None or dest_state is None:
        raise ValueError("map.customs must contain start_state and dest_state.")

    start_xy = (float(start_state.x), float(start_state.y))
    dest_xy = (float(dest_state.x), float(dest_state.y))
    guidance_points = _coerce_guidance_points(scene_meta.get("guidance_path_points"))
    start_anchor = _support_anchor(scene_meta.get("start_support_edge"), support_edge_offset)
    dest_anchor = _support_anchor(scene_meta.get("dest_support_edge"), support_edge_offset)

    anchors = [start_xy]
    if start_anchor is not None and np.linalg.norm(np.array(start_anchor) - np.array(start_xy)) > 0.5:
        anchors.append(start_anchor)
    anchors.extend(guidance_points)
    if dest_anchor is not None:
        last_point = anchors[-1] if anchors else start_xy
        if np.linalg.norm(np.array(dest_anchor) - np.array(last_point)) > 0.5:
            anchors.append(dest_anchor)
    anchors.append(dest_xy)
    anchors = _dedupe_points(anchors)

    if len(anchors) < 2:
        heading_vector = np.array([np.cos(start_state.heading), np.sin(start_state.heading)], dtype=float)
        anchors = [start_xy, tuple(np.array(start_xy) + heading_vector * max(sample_spacing, 1.0))]

    path = LineString(anchors)
    if path.length <= 1e-9:
        path = LineString([start_xy, dest_xy])

    target_centroid = None
    if target_boxes:
        target_union = unary_union(list(target_boxes)).buffer(0)
        target_centroid = (float(target_union.centroid.x), float(target_union.centroid.y))

    path_length = max(float(path.length), 1e-6)
    if num_samples is None:
        num_samples = max(int(np.ceil(path_length / max(sample_spacing, 0.1))) + 1, 24)
    num_samples = max(int(num_samples), 2)

    distances = np.linspace(0.0, path_length, num_samples)
    epsilon = max(path_length / max(num_samples - 1, 1), 0.5)
    start_phi = float(start_state.articulation_angle)
    dest_phi = float(dest_state.articulation_angle)
    transition_span = max(2, int(round(0.18 * num_samples)))
    states = []

    for index, distance in enumerate(distances):
        point = path.interpolate(float(distance))

        if index == 0:
            x_coord, y_coord = start_xy
            heading = float(start_state.heading)
            articulation = start_phi
        elif index == num_samples - 1:
            x_coord, y_coord = dest_xy
            heading = float(dest_state.heading)
            articulation = dest_phi
        else:
            x_coord = float(point.x)
            y_coord = float(point.y)
            heading = _sample_heading(path, float(distance), epsilon)
            start_blend = 1.0 - _smoothstep(index / transition_span)
            end_blend = _smoothstep((index - (num_samples - transition_span - 1)) / transition_span)
            heading = _interpolate_angle(heading, float(start_state.heading), start_blend)
            heading = _interpolate_angle(heading, float(dest_state.heading), end_blend)
            articulation = start_phi * start_blend + dest_phi * end_blend

        rear_heading = wrap_angle(heading - articulation)
        speed = float(nominal_speed)
        ramp_window = max(2, int(round(0.12 * num_samples)))
        if index < ramp_window:
            speed *= max(0.35, index / ramp_window)
        remaining = num_samples - index - 1
        if remaining < ramp_window:
            speed *= max(0.05, remaining / ramp_window)
        if index == num_samples - 1:
            speed = 0.0

        state = ArticulatedState(
            frame=index * int(step_interval_ms),
            x=x_coord,
            y=y_coord,
            heading=heading,
            speed=speed,
            steering=0.0,
            rear_heading=rear_heading,
        )
        state.update_trailer_loc(hitch_offset, trailer_length)
        states.append(state)

    obstacle_geometry = _collect_obstacle_geometry(map_)
    metadata = {
        "scene_type": customs.get("scene_type", map_.scenario_type),
        "scene_meta": scene_meta,
        "start_state": start_state,
        "dest_state": dest_state,
        "target_boxes": target_boxes,
        "target_centroid": target_centroid,
        "used_guidance_path": bool(guidance_points),
        "obstacle_geometry": obstacle_geometry,
    }
    return ArticulatedReferenceTrajectory(
        states=states,
        path=path,
        anchors=anchors,
        guidance_points=guidance_points,
        metadata=metadata,
    )


class ArticulatedMPCController:
    def __init__(
        self,
        physics_model,
        horizon_steps: int = 10,
        step_interval_ms: int = 200,
        nominal_speed: float = 1.0,
        position_weight: float = 14.0,
        heading_weight: float = 3.0,
        articulation_weight: float = 2.0,
        speed_weight: float = 0.8,
        control_weight: float = 0.05,
        smoothness_weight: float = 0.25,
        terminal_weight_multiplier: float = 3.0,
        terminal_box_weight: float = 8.0,
        obstacle_margin: float = PPO_WIDTH * 0.75,
        obstacle_weight: float = 10.0,
        max_iterations: int = 40,
        obstacle_geometry=None,
    ):
        self.physics_model = physics_model
        self.horizon_steps = max(int(horizon_steps), 1)
        self.step_interval_ms = int(step_interval_ms)
        self.nominal_speed = float(nominal_speed)
        self.position_weight = float(position_weight)
        self.heading_weight = float(heading_weight)
        self.articulation_weight = float(articulation_weight)
        self.speed_weight = float(speed_weight)
        self.control_weight = float(control_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.terminal_weight_multiplier = float(terminal_weight_multiplier)
        self.terminal_box_weight = float(terminal_box_weight)
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_weight = float(obstacle_weight)
        self.max_iterations = max(int(max_iterations), 1)
        self.obstacle_geometry = obstacle_geometry
        self._last_reference_index = 0
        self._previous_controls = None

    def _control_bounds(self):
        steering_bounds = self.physics_model.steering_rate_range or (-1.0, 1.0)
        speed_bounds = self.physics_model.speed_range or (-self.nominal_speed, self.nominal_speed)
        return [steering_bounds, speed_bounds] * self.horizon_steps

    def _closest_reference_index(
        self, current_state: ArticulatedState, reference: ArticulatedReferenceTrajectory
    ) -> int:
        reference_index = reference.closest_index(current_state)
        reference_index = max(reference_index, self._last_reference_index)
        self._last_reference_index = reference_index
        return reference_index

    def _seed_controls(
        self, current_state: ArticulatedState, reference_window: Sequence[ArticulatedState]
    ) -> np.ndarray:
        if self._previous_controls is not None and self._previous_controls.shape == (self.horizon_steps, 2):
            return np.array(self._previous_controls, dtype=float)

        dt = max(self.step_interval_ms / 1000.0, 1e-6)
        seed = np.zeros((self.horizon_steps, 2), dtype=float)
        previous_phi = float(current_state.articulation_angle)
        for index, reference_state in enumerate(reference_window):
            target_phi = float(reference_state.articulation_angle)
            seed[index, 0] = (target_phi - previous_phi) / dt
            seed[index, 1] = (
                float(reference_state.speed)
                if reference_state.speed is not None
                else self.nominal_speed
            )
            previous_phi = target_phi

        steering_bounds = self.physics_model.steering_rate_range or (-1.0, 1.0)
        speed_bounds = self.physics_model.speed_range or (-self.nominal_speed, self.nominal_speed)
        seed[:, 0] = np.clip(seed[:, 0], steering_bounds[0], steering_bounds[1])
        seed[:, 1] = np.clip(seed[:, 1], speed_bounds[0], speed_bounds[1])
        return seed

    def _rollout(
        self, current_state: ArticulatedState, controls: np.ndarray
    ) -> List[ArticulatedState]:
        predicted_states = []
        rollout_state = self.physics_model.ensure_articulated_state(current_state)
        for steering_rate, speed in controls:
            rollout_state, _, _ = self.physics_model.step(
                rollout_state,
                steering=float(steering_rate),
                speed=float(speed),
                interval=self.step_interval_ms,
            )
            predicted_states.append(rollout_state)
        return predicted_states

    def _state_cost(
        self,
        predicted_state: ArticulatedState,
        reference_state: ArticulatedState,
        terminal_weight: float,
    ) -> float:
        position_delta = np.array(
            [predicted_state.x - reference_state.x, predicted_state.y - reference_state.y],
            dtype=float,
        )
        heading_error = _angle_difference(predicted_state.heading, reference_state.heading)
        articulation_error = _angle_difference(
            predicted_state.articulation_angle, reference_state.articulation_angle
        )
        predicted_speed = 0.0 if predicted_state.speed is None else float(predicted_state.speed)
        reference_speed = self.nominal_speed
        if reference_state.speed is not None:
            reference_speed = float(reference_state.speed)

        cost = terminal_weight * self.position_weight * float(position_delta @ position_delta)
        cost += terminal_weight * self.heading_weight * float(heading_error**2)
        cost += terminal_weight * self.articulation_weight * float(articulation_error**2)
        cost += self.speed_weight * float((predicted_speed - reference_speed) ** 2)
        return float(cost)

    def _obstacle_cost(self, predicted_state: ArticulatedState) -> float:
        if self.obstacle_geometry is None or self.obstacle_geometry.is_empty:
            return 0.0

        front_clearance = float(self.obstacle_geometry.distance(Point(predicted_state.x, predicted_state.y)))
        rear_x, rear_y, _ = predicted_state.get_rear_axle_position(
            self.physics_model.hitch_offset,
            self.physics_model.trailer_length,
        )
        rear_clearance = float(self.obstacle_geometry.distance(Point(rear_x, rear_y)))
        clearance = min(front_clearance, rear_clearance)
        if clearance >= self.obstacle_margin:
            return 0.0
        return self.obstacle_weight * float((self.obstacle_margin - clearance) ** 2)

    def _objective(
        self,
        flat_controls: np.ndarray,
        current_state: ArticulatedState,
        reference_window: Sequence[ArticulatedState],
        target_centroid: Optional[Tuple[float, float]],
    ) -> float:
        controls = np.asarray(flat_controls, dtype=float).reshape(self.horizon_steps, 2)
        predicted_states = self._rollout(current_state, controls)

        cost = 0.0
        previous_control = None
        for index, (predicted_state, reference_state) in enumerate(zip(predicted_states, reference_window)):
            terminal_weight = self.terminal_weight_multiplier if index == len(reference_window) - 1 else 1.0
            cost += self._state_cost(predicted_state, reference_state, terminal_weight)
            cost += self._obstacle_cost(predicted_state)

            control = controls[index]
            cost += self.control_weight * float(control @ control)
            if previous_control is not None:
                delta_control = control - previous_control
                cost += self.smoothness_weight * float(delta_control @ delta_control)
            previous_control = control

        if target_centroid is not None and predicted_states:
            final_state = predicted_states[-1]
            rear_x, rear_y, _ = final_state.get_rear_axle_position(
                self.physics_model.hitch_offset,
                self.physics_model.trailer_length,
            )
            vehicle_center = 0.5 * np.array([final_state.x + rear_x, final_state.y + rear_y], dtype=float)
            centroid_error = vehicle_center - np.asarray(target_centroid, dtype=float)
            cost += self.terminal_box_weight * float(centroid_error @ centroid_error)

        return float(cost)

    def solve(
        self,
        current_state: ArticulatedState,
        reference: ArticulatedReferenceTrajectory,
    ) -> MPCSolveResult:
        current_state = self.physics_model.ensure_articulated_state(current_state)
        reference_index = self._closest_reference_index(current_state, reference)
        reference_window = reference.window(reference_index + 1, self.horizon_steps)
        initial_controls = self._seed_controls(current_state, reference_window)
        target_centroid = reference.metadata.get("target_centroid")

        if not HAS_SCIPY:
            predicted_states = self._rollout(current_state, initial_controls)
            objective = self._objective(
                initial_controls.reshape(-1),
                current_state,
                reference_window,
                target_centroid,
            )
            self._previous_controls = np.vstack([initial_controls[1:], initial_controls[-1]])
            return MPCSolveResult(
                steering_rate=float(initial_controls[0, 0]),
                speed=float(initial_controls[0, 1]),
                predicted_states=predicted_states,
                reference_states=list(reference_window),
                reference_index=reference_index,
                objective=float(objective),
                success=False,
                iterations=0,
                message="scipy.optimize is unavailable; returned warm-start controls.",
            )

        result = minimize(
            self._objective,
            initial_controls.reshape(-1),
            args=(current_state, reference_window, target_centroid),
            method="L-BFGS-B",
            bounds=self._control_bounds(),
            options={"maxiter": self.max_iterations, "maxfun": self.max_iterations * 20},
        )

        optimized_controls = np.asarray(result.x, dtype=float).reshape(self.horizon_steps, 2)
        predicted_states = self._rollout(current_state, optimized_controls)
        self._previous_controls = np.vstack([optimized_controls[1:], optimized_controls[-1]])

        return MPCSolveResult(
            steering_rate=float(optimized_controls[0, 0]),
            speed=float(optimized_controls[0, 1]),
            predicted_states=predicted_states,
            reference_states=list(reference_window),
            reference_index=reference_index,
            objective=float(result.fun),
            success=bool(result.success),
            iterations=int(getattr(result, "nit", 0)),
            message=str(result.message),
        )


__all__ = [
    "ArticulatedMPCController",
    "ArticulatedReferenceTrajectory",
    "MPCSolveResult",
    "build_articulated_reference_trajectory",
    "HAS_SCIPY",
]