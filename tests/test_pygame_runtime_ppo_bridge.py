import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import LineString

from tactics2d.controller import ArticulatedReferenceTrajectory
from tactics2d.map.element import Map
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator
from tactics2d.participant.trajectory import ArticulatedState
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


def _make_directional_guard_planner():
    planner = PPOPrimitivePathPlanner.__new__(PPOPrimitivePathPlanner)
    planner.ppo_configs = SimpleNamespace(LIDAR_NUM=8, LIDAR_RANGE=10.0)
    planner.safety_stop_distance_m = 2.0
    planner.safety_forward_sector_half_angle = math.radians(20.0)
    planner.safety_forward_sector_max_half_angle = math.radians(55.0)
    planner.safety_steering_sector_center_gain = 0.5
    planner.safety_steering_sector_half_angle_gain = 1.0
    planner.safety_collision_buffer_m = 0.25
    planner.primitive_interval_ms = 500
    planner.replan_on_emergency_stop = True
    planner._last_safety_stop_stats = {}
    return planner


class _StubPrimitiveLibrary:
    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.size = int(self.actions.shape[0])
        self.horizon = int(self.actions.shape[1])

    def get_actions(self, primitive_id):
        return self.actions[int(primitive_id)]


def _make_prefix_safe_planner(
    actions: np.ndarray,
    dist_star: np.ndarray,
    probs: np.ndarray,
    *,
    front_box,
    rear_box,
):
    planner = PPOPrimitivePathPlanner.__new__(PPOPrimitivePathPlanner)
    planner.ppo_configs = SimpleNamespace(
        LIDAR_NUM=int(dist_star.shape[2]),
        LIDAR_RANGE=10.0,
        VALID_SPEED=[-2.5, 2.5],
        VALID_STEER=[-np.radians(36.0), np.radians(36.0)],
    )
    planner.control_interval_ms = 100
    planner.primitive_interval_ms = 200
    planner.replan_every_steps = 4
    planner.deterministic = True
    planner.use_action_mask = True
    planner.action_mask_update_every_k = 1
    planner.action_mask_mode = "soft_ray"
    planner.mask_use_fast_prune = False
    planner.soft_mask_gamma = 1.5
    planner.soft_mask_eps = 0.01
    planner.soft_mask_terminal_gamma = 0.5
    planner.soft_mask_terminal_eps = 0.05
    planner.soft_mask_terminal_radius = 0.5
    planner.soft_mask_min_action_count = 1
    planner.soft_mask_terminal_heading_scale = np.radians(35.0)
    planner.soft_mask_terminal_articulation_scale = np.radians(35.0)
    planner.soft_mask_terminal_weight_min = 0.60
    planner.soft_mask_terminal_weight_max = 1.25
    planner.safety_stop_distance_m = 0.5
    planner.safety_forward_sector_half_angle = math.radians(18.0)
    planner.safety_forward_sector_max_half_angle = math.radians(55.0)
    planner.safety_steering_sector_center_gain = 0.5
    planner.safety_steering_sector_half_angle_gain = 1.0
    planner.safety_collision_buffer_m = 0.25
    planner.replan_on_emergency_stop = True
    planner.max_guard_candidates = 2
    planner.max_candidate_primitives = 2
    planner.emergency_primitive_id = 1
    planner.goal_tolerance_m = 2.0
    planner.min_progress_m = 0.05
    planner.max_stagnation_steps = 4
    planner.primitive_library = _StubPrimitiveLibrary(actions)
    planner._ray_safety_index = SimpleNamespace(dist_star=np.asarray(dist_star, dtype=np.float32))
    planner._safe_prefix_steps_cached = None
    planner._last_safe_prefix_steps = None
    planner._control_prefix_state_cache = None
    planner._control_prefix_cache_repeat_count = 0
    planner._control_prefix_cache_control_steps = 0
    planner._last_guard_stats = {}
    planner._last_safety_stop_stats = {}
    planner._action_mask_cached = None
    planner._action_mask_calls_since_update = 0
    planner._action_mask_index = None
    planner._action_mask_index_source = "none"
    planner._action_mask_inflation_offsets = []
    planner._last_action_mask_stats = {
        "precomputed_available": False,
        "precomputed_used": False,
        "precomputed_candidate_count": None,
        "precomputed_fallback_to_full": False,
        "precomputed_index_kind": None,
        "precomputed_index_source": "none",
        "ray_safety_available": True,
        "soft_mask_ms": None,
        "soft_effective_action_count": None,
    }
    planner.terminal_heading_tolerance_deg = 5.0
    planner.terminal_overlap_target = 0.75
    planner.terminal_front_overlap_target = 0.80
    planner.terminal_rear_overlap_min = 0.45
    planner.parked_stop_speed_mps = 0.20
    planner._front_box = front_box
    planner._rear_box = rear_box

    class _StubAgent:
        def __init__(self, probabilities: np.ndarray):
            self._probs = torch.as_tensor(probabilities, dtype=torch.float32)

        def _actor_forward(self, observation, action_mask=None):
            return SimpleNamespace(probs=self._probs)

    planner.agent = _StubAgent(np.asarray(probs, dtype=np.float32))
    return planner


def _make_stub_planning_result(participant, primitive_id: int, control_actions: np.ndarray, metadata: dict):
    control_actions = np.asarray(control_actions, dtype=np.float64)
    current_state = participant.current_state
    path = LineString(
        [
            (float(current_state.x), float(current_state.y)),
            (float(current_state.x) + 1.0, float(current_state.y)),
        ]
    )
    reference = ArticulatedReferenceTrajectory(
        states=[current_state, current_state],
        path=path,
        anchors=list(path.coords),
        metadata={"reference_path_source": "stub_reference"},
    )
    merged_metadata = {
        "reference_path_source": "stub_reference",
        "planning_mode": "closed_loop_policy",
        "action_mask_used": True,
        "action_mask_feasible_count": 1,
        "control_actions_shape": tuple(int(dim) for dim in control_actions.shape),
    }
    merged_metadata.update(metadata)
    return planner_result(
        primitive_id=int(primitive_id),
        primitive_actions=control_actions.copy(),
        control_actions=control_actions.copy(),
        observation=np.zeros((8,), dtype=np.float64),
        reference=reference,
        metadata=merged_metadata,
    )


def planner_result(**kwargs):
    return SimpleNamespace(**kwargs)


def test_directional_guard_triggers_stop_for_front_obstacle():
    planner = _make_directional_guard_planner()
    scene = SimpleNamespace(map_=SimpleNamespace(boundary=(-10.0, 10.0, -10.0, 10.0)))
    current_state = SimpleNamespace(x=0.0, y=0.0, heading=0.0, steering=0.0)
    observation = np.array([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    primitive_actions = np.array([[0.0, 3.0]], dtype=np.float64)

    stop_stats = planner._evaluate_directional_stop(scene, current_state, observation, primitive_actions)

    assert stop_stats["stop_triggered"] is True
    assert stop_stats["stop_replan_requested"] is True
    assert stop_stats["stop_clearance_distance_m"] == pytest.approx(1.0)
    assert stop_stats["stop_continue_will_collide"] is True


def test_directional_guard_ignores_side_obstacle():
    planner = _make_directional_guard_planner()
    scene = SimpleNamespace(map_=SimpleNamespace(boundary=(-10.0, 10.0, -10.0, 10.0)))
    current_state = SimpleNamespace(x=0.0, y=0.0, heading=0.0, steering=0.0)
    observation = np.array([1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    primitive_actions = np.array([[0.0, 3.0]], dtype=np.float64)

    stop_stats = planner._evaluate_directional_stop(scene, current_state, observation, primitive_actions)

    assert stop_stats["stop_triggered"] is False
    assert stop_stats["stop_replan_requested"] is False
    assert stop_stats["stop_lidar_distance_m"] == pytest.approx(10.0)


def test_policy_selected_primitive_keeps_full_horizon_without_post_decision_truncation():
    scene, participant = _build_navigation_scene()
    actions = np.array(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, -1.0], [0.0, -1.0], [0.0, -1.0]],
        ],
        dtype=np.float64,
    )
    dist_star = np.array(
        [
            [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0], [6.0, 6.0, 6.0, 6.0]],
            [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0]],
        ],
        dtype=np.float32,
    )
    planner = _make_prefix_safe_planner(
        actions,
        dist_star,
        probs=np.array([0.9, 0.1], dtype=np.float32),
        front_box=participant._front_bbox,
        rear_box=participant._rear_bbox,
    )
    current_state = participant.physics_model.ensure_articulated_state(participant.current_state)
    observation = np.full((4,), 0.5, dtype=np.float64)

    primitive_id, primitive_actions, rollout_states, _, selection_info = planner._choose_closed_loop_primitive(
        scene,
        participant,
        current_state,
        observation,
    )

    assert primitive_id == 0
    assert primitive_actions.shape == (3, 2)
    assert selection_info["safe_prefix_primitive_steps"] is None
    assert selection_info["control_prefix_steps"] == 6
    assert selection_info["prefix_truncated"] is False
    assert selection_info["guard_mode"] == "directional_stop_guard"
    assert planner._last_guard_stats["guard_selected_primitive_id"] == 0
    assert planner._last_guard_stats["guard_final_primitive_id"] == 0
    assert planner._last_guard_stats["guard_fallback_used"] is False
    assert planner._last_guard_stats["guard_emergency_used"] is False
    assert len(rollout_states) == 7


def test_policy_selected_primitive_is_not_replaced_by_soft_prefix_fallback():
    scene, participant = _build_navigation_scene()
    actions = np.array(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, -1.0], [0.0, -1.0], [0.0, -1.0]],
        ],
        dtype=np.float64,
    )
    dist_star = np.array(
        [
            [[6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0]],
            [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0], [6.0, 6.0, 6.0, 6.0]],
        ],
        dtype=np.float32,
    )
    planner = _make_prefix_safe_planner(
        actions,
        dist_star,
        probs=np.array([0.9, 0.1], dtype=np.float32),
        front_box=participant._front_bbox,
        rear_box=participant._rear_bbox,
    )
    current_state = participant.physics_model.ensure_articulated_state(participant.current_state)
    observation = np.full((4,), 0.5, dtype=np.float64)

    primitive_id, primitive_actions, rollout_states, _, selection_info = planner._choose_closed_loop_primitive(
        scene,
        participant,
        current_state,
        observation,
    )

    assert primitive_id == 0
    assert primitive_actions.shape == (3, 2)
    assert selection_info["safe_prefix_primitive_steps"] is None
    assert selection_info["control_prefix_steps"] == 6
    assert selection_info["guard_mode"] == "directional_stop_guard"
    assert planner._last_guard_stats["guard_selected_primitive_id"] == 0
    assert planner._last_guard_stats["guard_final_primitive_id"] == 0
    assert planner._last_guard_stats["guard_fallback_used"] is False
    assert planner._last_guard_stats["guard_emergency_used"] is False
    assert len(rollout_states) == 7


@pytest.mark.render
def test_simulation_runner_replans_immediately_after_stop_request():
    scene, participant = _build_navigation_scene()
    runner = SimulationRunner(scene=scene, participant=participant, renderer=None, dt_ms=100, max_steps=3)

    class _StubPlanner:
        def __init__(self):
            self.replan_every_steps = 99
            self.calls = 0

        def plan(self, scene, participant):
            self.calls += 1
            if self.calls == 1:
                return _make_stub_planning_result(
                    participant,
                    primitive_id=7,
                    control_actions=np.zeros((1, 2), dtype=np.float64),
                    metadata={
                        "primitive_id": 7,
                        "stop_triggered": True,
                        "stop_replan_requested": True,
                    },
                )
            return _make_stub_planning_result(
                participant,
                primitive_id=3,
                control_actions=np.array([[0.0, 1.0]], dtype=np.float64),
                metadata={
                    "primitive_id": 3,
                    "stop_triggered": False,
                    "stop_replan_requested": False,
                },
            )

    runner.wheel_loader_planner = _StubPlanner()
    runner.controller = None
    runner._initialize_planned_reference()

    active = runner.step_once()

    assert active is True
    assert runner.wheel_loader_planner.calls == 2
    assert runner.last_planning_result.primitive_id == 3
    assert runner.last_planning_result.metadata["stop_replan_requested"] is False
    assert runner.pending_primitive_controls == [(0.0, 1.0)]
    assert runner.pending_primitive_control_index == 0
    assert runner.last_planning_step == 0
    assert participant.current_state.frame > 0


@pytest.mark.render
def test_simulation_runner_replans_when_prefix_controls_exhaust():
    scene, participant = _build_navigation_scene()
    runner = SimulationRunner(scene=scene, participant=participant, renderer=None, dt_ms=100, max_steps=4)

    class _StubPlanner:
        def __init__(self):
            self.replan_every_steps = 99
            self.calls = 0

        def plan(self, scene, participant):
            self.calls += 1
            if self.calls == 1:
                return _make_stub_planning_result(
                    participant,
                    primitive_id=7,
                    control_actions=np.array([[0.0, 1.0]], dtype=np.float64),
                    metadata={
                        "primitive_id": 7,
                        "stop_triggered": False,
                        "stop_replan_requested": False,
                        "prefix_truncated": True,
                        "replan_when_controls_exhausted": True,
                    },
                )
            return _make_stub_planning_result(
                participant,
                primitive_id=3,
                control_actions=np.array([[0.0, 0.5]], dtype=np.float64),
                metadata={
                    "primitive_id": 3,
                    "stop_triggered": False,
                    "stop_replan_requested": False,
                    "prefix_truncated": False,
                    "replan_when_controls_exhausted": False,
                },
            )

    runner.wheel_loader_planner = _StubPlanner()
    runner.controller = None
    runner._initialize_planned_reference()

    assert runner.wheel_loader_planner.calls == 1
    assert runner.pending_primitive_controls == [(0.0, 1.0)]

    first_active = runner.step_once()
    second_active = runner.step_once()

    assert first_active is True
    assert second_active is True
    assert runner.wheel_loader_planner.calls == 2
    assert runner.last_planning_result.primitive_id == 3
    assert runner.pending_primitive_controls == [(0.0, 0.5)]
    assert runner.pending_primitive_control_index == 1


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


@pytest.mark.render
def test_parking_completion_requires_precise_slot_overlap_not_distance_only():
    scene, participant = _build_navigation_scene()
    runner = SimulationRunner(scene=scene, participant=participant, renderer=None, dt_ms=100, max_steps=3)
    dest_state = scene.map_.customs["dest_state"]

    near_state = ArticulatedState(
        frame=int(dest_state.frame) + 100,
        x=float(dest_state.x) + 1.0,
        y=float(dest_state.y),
        heading=float(dest_state.heading),
        speed=0.0,
        accel=0.0,
        rear_heading=float(dest_state.rear_heading),
        steering=0.0,
    )
    near_distance = float(np.hypot(near_state.x - scene.goal_point[0], near_state.y - scene.goal_point[1]))
    near_complete, near_metrics = runner._parking_completion_status(near_state)
    exact_complete, exact_metrics = runner._parking_completion_status(dest_state)

    assert near_distance < 2.0
    assert near_complete is False
    assert near_metrics["geometry_available"] is True
    assert exact_complete is True
    assert exact_metrics["mean_overlap"] >= 0.75


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