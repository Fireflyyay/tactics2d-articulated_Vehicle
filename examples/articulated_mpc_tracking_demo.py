import argparse
from pathlib import Path
from typing import Iterable, Optional
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactics2d.controller import (
    HAS_SCIPY,
    ArticulatedMPCController,
    build_articulated_reference_trajectory,
)
from tactics2d.map.element import Map
from tactics2d.map.generator import PPOParkingMapGenerator
from tactics2d.participant.element import WheelLoader
from tactics2d.participant.trajectory.articulated_state import wrap_angle


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        yield geometry
        return

    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield polygon
        return

    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            yield from _iter_polygons(geom)
        return

    buffered = geometry.buffer(0)
    yield from _iter_polygons(buffered)


def _plot_geometry(ax, geometry, **kwargs):
    for polygon in _iter_polygons(geometry):
        x_coord, y_coord = polygon.exterior.xy
        ax.fill(x_coord, y_coord, **kwargs)


def _plot_outline(ax, geometry, **kwargs):
    for polygon in _iter_polygons(geometry):
        x_coord, y_coord = polygon.exterior.xy
        ax.plot(x_coord, y_coord, **kwargs)


def run_tracking_demo(
    scene_type: str = "navigation",
    map_level: str = "Normal",
    seed: int = 42,
    max_steps: int = 45,
    horizon_steps: int = 8,
    step_interval_ms: int = 200,
    nominal_speed: float = 1.0,
):
    map_ = Map(name=f"ppo_{scene_type}_mpc", scenario_type=scene_type)
    generator = PPOParkingMapGenerator(scene_type=scene_type, map_level=map_level)
    start_state, _, _ = generator.generate(map_, seed=seed)

    wheel_loader = WheelLoader(id_=0, verify=True)
    wheel_loader.trajectory.add_state(start_state)
    wheel_loader.current_articulation = start_state.articulation_angle

    reference = build_articulated_reference_trajectory(
        map_,
        step_interval_ms=step_interval_ms,
        nominal_speed=nominal_speed,
    )
    controller = ArticulatedMPCController(
        physics_model=wheel_loader.physics_model,
        horizon_steps=horizon_steps,
        step_interval_ms=step_interval_ms,
        nominal_speed=nominal_speed,
        obstacle_geometry=reference.metadata.get("obstacle_geometry"),
    )

    control_history = []
    last_result = None
    dest_state = map_.customs["dest_state"]
    stop_reason = "max_steps"

    for _ in range(max_steps):
        last_result = controller.solve(wheel_loader.current_state, reference)
        next_state, _, _ = wheel_loader.physics_model.step(
            wheel_loader.current_state,
            steering=last_result.steering_rate,
            speed=last_result.speed,
            interval=step_interval_ms,
        )
        wheel_loader.add_state(next_state)
        control_history.append(last_result)

        position_error = np.linalg.norm(
            np.array([next_state.x - dest_state.x, next_state.y - dest_state.y], dtype=float)
        )
        heading_error = abs(wrap_angle(next_state.heading - dest_state.heading))
        articulation_error = abs(
            wrap_angle(next_state.articulation_angle - dest_state.articulation_angle)
        )
        if position_error < 1.0 and heading_error < 0.25 and articulation_error < 0.25:
            stop_reason = "goal_reached"
            break

    return {
        "map": map_,
        "generator": generator,
        "wheel_loader": wheel_loader,
        "reference": reference,
        "controller": controller,
        "control_history": control_history,
        "last_result": last_result,
        "stop_reason": stop_reason,
        "seed": seed,
        "scene_type": scene_type,
        "map_level": map_level,
        "step_interval_ms": step_interval_ms,
    }


def render_tracking_report(result: dict, output_path: Optional[Path] = None, show: bool = False):
    map_ = result["map"]
    reference = result["reference"]
    wheel_loader = result["wheel_loader"]
    control_history = result["control_history"]
    scene_meta = map_.customs.get("scene_meta") or {}
    start_boxes = map_.customs.get("start_boxes") or ()
    target_boxes = map_.customs.get("target_boxes") or ()

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    map_axis, control_axis = axes

    for area in map_.areas.values():
        subtype = getattr(area, "subtype", None)
        if subtype == "freespace":
            _plot_geometry(map_axis, area.geometry, color="#f3f5f7", alpha=0.65)
        elif subtype == "obstacle":
            _plot_geometry(map_axis, area.geometry, color="#3c4658", alpha=0.9)

    drivable = scene_meta.get("drivable")
    if drivable is not None:
        _plot_outline(map_axis, drivable, color="#b8c4d0", linewidth=1.0, alpha=0.9)

    for corridor in scene_meta.get("corridors", []) or []:
        _plot_outline(map_axis, corridor, color="#94a9bf", linewidth=0.8, alpha=0.7, linestyle="--")

    plaza = scene_meta.get("plaza")
    if plaza is not None:
        _plot_outline(map_axis, plaza, color="#6c8197", linewidth=1.1, alpha=0.8)

    if reference.guidance_points:
        guidance_array = np.asarray(reference.guidance_points, dtype=float)
        map_axis.plot(
            guidance_array[:, 0],
            guidance_array[:, 1],
            color="#737373",
            linewidth=1.2,
            linestyle=":",
            label="Guidance Path",
        )

    reference_positions = reference.positions
    map_axis.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        color="#ff8c00",
        linewidth=2.0,
        linestyle="--",
        label="MPC Reference",
    )

    front_trajectory = np.array(
        [
            (wheel_loader.trajectory.get_state(frame).x, wheel_loader.trajectory.get_state(frame).y)
            for frame in wheel_loader.trajectory.frames
        ],
        dtype=float,
    )
    rear_trajectory = np.array(
        [
            (wheel_loader.get_rear_axle_state(frame).x, wheel_loader.get_rear_axle_state(frame).y)
            for frame in wheel_loader.trajectory.frames
        ],
        dtype=float,
    )
    map_axis.plot(front_trajectory[:, 0], front_trajectory[:, 1], color="#165dff", linewidth=2.0, label="Front Axle")
    map_axis.plot(rear_trajectory[:, 0], rear_trajectory[:, 1], color="#f77234", linewidth=1.5, label="Rear Axle")

    for polygon in start_boxes:
        _plot_outline(map_axis, polygon, color="#6495ed", linewidth=2.0)
    for polygon in target_boxes:
        _plot_outline(map_axis, polygon, color="#458b00", linewidth=2.0)

    pose_stride = max(1, len(wheel_loader.trajectory.frames) // 6)
    for frame in wheel_loader.trajectory.frames[::pose_stride]:
        rear_pose, front_pose = wheel_loader.get_pose(frame)
        _plot_outline(map_axis, rear_pose, color="#f77234", linewidth=1.0, alpha=0.4)
        _plot_outline(map_axis, front_pose, color="#165dff", linewidth=1.0, alpha=0.4)

    boundary = map_.boundary
    map_axis.set_xlim(boundary[0] - 2.0, boundary[1] + 2.0)
    map_axis.set_ylim(boundary[2] - 2.0, boundary[3] + 2.0)
    map_axis.set_aspect("equal")
    map_axis.set_title(
        f"PPO Scene + MPC Tracking\nscene={result['scene_type']} seed={result['seed']} stop={result['stop_reason']}"
    )
    map_axis.legend(loc="upper right")

    if control_history:
        steering_series = [solve_result.steering_rate for solve_result in control_history]
        speed_series = [solve_result.speed for solve_result in control_history]
        objective_series = [solve_result.objective for solve_result in control_history]
        control_axis.plot(steering_series, color="#165dff", linewidth=1.8, label="Steering Rate")
        control_axis.plot(speed_series, color="#14b8a6", linewidth=1.8, label="Speed")
        control_axis.plot(objective_series, color="#f59e0b", linewidth=1.2, label="Objective")
        control_axis.set_title("Control / Objective History")
        control_axis.set_xlabel("MPC Step")
        control_axis.grid(alpha=0.25)
        control_axis.legend(loc="upper right")
    else:
        control_axis.set_title("No MPC iterations executed")
        control_axis.axis("off")

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO scene articulated MPC tracking demo.")
    parser.add_argument("--scene-type", default="navigation", choices=["navigation", "bay", "parallel"])
    parser.add_argument("--map-level", default="Normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--step-ms", type=int, default=200)
    parser.add_argument("--nominal-speed", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="examples/runtime/articulated_mpc_tracking_demo.png",
        help="Path to save the rendered report.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_tracking_demo(
        scene_type=args.scene_type,
        map_level=args.map_level,
        seed=args.seed,
        max_steps=args.max_steps,
        horizon_steps=args.horizon,
        step_interval_ms=args.step_ms,
        nominal_speed=args.nominal_speed,
    )
    render_tracking_report(result, output_path=Path(args.output), show=args.show)


if __name__ == "__main__":
    main()