import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import pytest

from tactics2d.controller import (
    HAS_SCIPY,
    ArticulatedMPCController,
    build_articulated_reference_trajectory,
)
from tactics2d.map.element import Map
from tactics2d.map.generator import PPOParkingMapGenerator
from tactics2d.participant.element import WheelLoader


def _build_navigation_map(seed: int = 11):
    map_ = Map(name="ppo_navigation_test", scenario_type="navigation")
    generator = PPOParkingMapGenerator(scene_type="navigation", map_level="Normal")
    generator.generate(map_, seed=seed)
    return map_


@pytest.mark.map_generator
@pytest.mark.math
def test_reference_trajectory_uses_ppo_scene_customs():
    map_ = _build_navigation_map()
    reference = build_articulated_reference_trajectory(
        map_,
        num_samples=32,
        step_interval_ms=200,
        nominal_speed=1.0,
    )

    start_state = map_.customs["start_state"]
    dest_state = map_.customs["dest_state"]

    assert len(reference.states) == 32
    assert reference.metadata["scene_type"] == "navigation"
    assert reference.metadata["target_boxes"] is map_.customs["target_boxes"]
    assert reference.metadata["target_centroid"] is not None
    assert reference.metadata["scene_meta"] is map_.customs["scene_meta"]
    assert reference.positions.shape == (32, 2)
    assert np.linalg.norm(reference.positions[0] - np.array([start_state.x, start_state.y])) < 1e-9
    assert np.linalg.norm(reference.positions[-1] - np.array([dest_state.x, dest_state.y])) < 1e-9
    assert reference.states[-1].heading == pytest.approx(dest_state.heading)
    assert reference.states[-1].articulation_angle == pytest.approx(dest_state.articulation_angle)
    assert reference.path.length > 0.0


@pytest.mark.physics
def test_mpc_controller_returns_bounded_control():
    map_ = _build_navigation_map(seed=7)
    reference = build_articulated_reference_trajectory(
        map_,
        num_samples=24,
        step_interval_ms=200,
        nominal_speed=1.0,
    )

    wheel_loader = WheelLoader(id_=0, verify=True)
    start_state = map_.customs["start_state"]
    wheel_loader.trajectory.add_state(start_state)
    wheel_loader.current_articulation = start_state.articulation_angle

    controller = ArticulatedMPCController(
        physics_model=wheel_loader.physics_model,
        horizon_steps=4,
        step_interval_ms=200,
        nominal_speed=1.0,
        max_iterations=8,
        obstacle_geometry=reference.metadata.get("obstacle_geometry"),
    )
    solve_result = controller.solve(wheel_loader.current_state, reference)

    steer_low, steer_high = wheel_loader.physics_model.steering_rate_range
    speed_low, speed_high = wheel_loader.physics_model.speed_range
    assert steer_low <= solve_result.steering_rate <= steer_high
    assert speed_low <= solve_result.speed <= speed_high
    assert len(solve_result.predicted_states) == 4
    assert len(solve_result.reference_states) == 4
    assert np.isfinite(solve_result.objective)
    if not HAS_SCIPY:
        assert "unavailable" in solve_result.message