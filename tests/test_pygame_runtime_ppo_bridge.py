from pathlib import Path

import numpy as np
import pytest

from tactics2d.map.element import Map
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator
from tactics2d.renderer import SimulationRunner, adapt_generated_scene, create_default_participant
from tactics2d.renderer.ppo_primitive_bridge import PPOPrimitivePathPlanner


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ppo_assets():
    workspace_root = _workspace_root()
    checkpoint_path = workspace_root / "BestCheckPoint" / "PPO_best.pt"
    ppo_root = workspace_root / "PPO_articulated_vehicle"
    if not checkpoint_path.exists() or not ppo_root.exists():
        pytest.skip("PPO checkpoint or PPO_articulated_vehicle repo is unavailable in this workspace.")
    return checkpoint_path, ppo_root


def _build_navigation_scene():
    _, ppo_root = _ppo_assets()
    generator = WheelLoaderScenarioGenerator(
        backend="ppo",
        scene_type="navigation",
        map_level="Normal",
        ppo_root=str(ppo_root),
    )
    map_ = Map(name="ppo_bridge_test", scenario_type="wheel_loader")
    generate_result = generator.generate(map_, seed=42)
    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)
    participant = create_default_participant(scene)
    return scene, participant


@pytest.mark.render
def test_ppo_primitive_planner_builds_reference():
    checkpoint_path, ppo_root = _ppo_assets()
    scene, participant = _build_navigation_scene()

    planner = PPOPrimitivePathPlanner(
        checkpoint_path=str(checkpoint_path),
        ppo_root=str(ppo_root),
        control_interval_ms=100,
        replan_every_steps=1,
        deterministic=True,
    )
    assert getattr(planner.agent.state_normalize, "n_state", 0) > 1
    result = planner.plan(scene, participant)

    assert result.observation.shape == (133,)
    assert result.observation.shape == (planner.observation_dim,)
    assert result.primitive_actions.ndim == 2
    assert result.control_actions.ndim == 2
    assert len(result.reference.states) >= 2
    assert np.isfinite(result.reference.positions).all()
    assert result.metadata["primitive_id"] == result.primitive_id
    assert result.metadata["reference_path_source"] == "ppo_primitive_global_plan"
    assert result.reference.path.length > 0.0
    assert result.metadata["planning_mode"] == "closed_loop_policy"
    assert result.metadata["action_mask_used"] is True
    assert result.metadata["action_mask_feasible_count"] > 0
    assert len(result.metadata["primitive_sequence"]) == 1
    assert result.metadata["control_actions_shape"] == result.control_actions.shape
    assert result.metadata["primitive_selected_count"] == 1
    assert result.metadata["primitive_selection_counts"][result.primitive_id] == 1
    assert result.metadata["adaptive_selected_count_total"] == sum(result.metadata["adaptive_primitive_selection_counts"].values())
    assert result.metadata["primitive_origin"] in {"adaptive", "base", "unknown"}
    if result.metadata["primitive_origin"] == "adaptive":
        assert result.metadata["primitive_added_round"] is not None
        assert result.metadata["adaptive_selected_count_total"] == 1
    else:
        assert result.metadata["primitive_added_round"] is None

    second_result = planner.plan(scene, participant)

    assert second_result.metadata["primitive_selected_count"] >= 1
    assert second_result.metadata["primitive_selection_counts"][second_result.primitive_id] == second_result.metadata["primitive_selected_count"]
    assert sum(second_result.metadata["primitive_selection_counts"].values()) == 2
    assert second_result.metadata["adaptive_selected_count_total"] == sum(second_result.metadata["adaptive_primitive_selection_counts"].values())
    if second_result.metadata["primitive_origin"] == "adaptive":
        assert second_result.metadata["primitive_added_round"] is not None
        assert second_result.metadata["adaptive_round_selection_counts"][second_result.metadata["primitive_added_round"]] >= 1


@pytest.mark.render
def test_simulation_runner_consumes_ppo_reference():
    checkpoint_path, ppo_root = _ppo_assets()
    scene, participant = _build_navigation_scene()

    runner = SimulationRunner(
        scene=scene,
        participant=participant,
        renderer=None,
        dt_ms=100,
        max_steps=3,
        wheel_loader_planner={
            "mode": "ppo",
            "checkpoint_path": str(checkpoint_path),
            "ppo_root": str(ppo_root),
            "replan_every_steps": 1,
            "deterministic": True,
        },
    )

    assert runner.last_planning_result is not None
    assert runner.scene.metadata["reference_path_source"] == "ppo_primitive_global_plan"
    planned_length = runner.scene.reference_path.length
    base_length = runner.scene.base_reference_path.length
    initial_primitive_id = runner.last_planning_result.primitive_id

    runner.step_once()

    assert runner.last_planning_result is not None
    assert runner.scene.metadata["reference_path_source"] == "ppo_primitive_global_plan"
    assert runner.active_reference_trajectory is not None
    assert runner.last_planning_result.metadata["planning_mode"] == "closed_loop_policy"
    assert runner.last_planning_result.metadata["action_mask_used"] is True
    assert runner.last_planning_result.metadata["action_mask_feasible_count"] > 0
    assert runner.last_planning_result.metadata["primitive_selected_count"] >= 1
    assert runner.last_planning_result.metadata["adaptive_selected_count_total"] == sum(
        runner.last_planning_result.metadata["adaptive_primitive_selection_counts"].values()
    )
    assert runner.last_planning_step == 0
    assert runner.scene.reference_path.length > 0.0
    assert planned_length > 0.0
    assert base_length > 0.0
    assert participant.current_state.frame > 0
    assert isinstance(initial_primitive_id, int)
    assert runner.pending_primitive_controls
    assert runner.pending_primitive_control_index == 1

    runner.step_once()

    assert runner.last_planning_step == 1
    assert sum(runner.last_planning_result.metadata["primitive_selection_counts"].values()) == 2
    assert (
        runner.last_planning_result.metadata["primitive_selection_counts"][runner.last_planning_result.primitive_id]
        == runner.last_planning_result.metadata["primitive_selected_count"]
    )


@pytest.mark.render
def test_simulation_runner_stops_before_colliding_obstacle():
    scene, participant = _build_navigation_scene()
    obstacle = next(
        area.geometry
        for area in scene.map_.areas.values()
        if getattr(area, "subtype", None) == "obstacle"
    )
    collision_state = participant.build_state_from_rear_axle(
        frame=participant.current_state.frame + 100,
        x=float(obstacle.centroid.x),
        y=float(obstacle.centroid.y),
        heading=float(participant.get_rear_axle_state().heading),
        speed=0.0,
        accel=0.0,
        articulation_angle=participant.current_articulation,
    )

    class _DummyController:
        def step(self, **kwargs):
            return 0.0, 0.0

    class _StubPhysics:
        def __init__(self, next_state, articulation):
            self.next_state = next_state
            self.articulation = articulation

        def step(self, *args, **kwargs):
            return self.next_state, None, self.articulation

    runner = SimulationRunner(scene=scene, participant=participant, renderer=None, dt_ms=100, max_steps=3)
    runner.controller = _DummyController()
    runner.participant.physics_model = _StubPhysics(collision_state, participant.current_articulation)

    active = runner.step_once()

    assert active is False
    assert runner.last_status == "collision"
    assert participant.current_state.frame == 0