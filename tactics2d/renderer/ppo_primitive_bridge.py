import importlib
import json
import math
import os
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from shapely.geometry import LineString, Point

from tactics2d.controller import ArticulatedReferenceTrajectory
from tactics2d.map.generator.generate_ppo_parking_map import _discover_ppo_root
from tactics2d.participant.trajectory import ArticulatedState
from tactics2d.utils.ppo_articulated_defaults import (
    PPO_FRONT_OVERHANG,
    PPO_HITCH_OFFSET,
    PPO_REAR_OVERHANG,
    PPO_TRAILER_LENGTH,
    PPO_WIDTH,
    build_front_vehicle_box,
    build_rear_vehicle_box,
)


_PPO_IMPORT_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_checkpoint(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return torch.load(path, map_location=map_location)


def _restore_agent_for_inference(agent, checkpoint: object):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint content is not a state-dict dictionary.")

    actor_net = checkpoint.get("actor_net")
    critic_net = checkpoint.get("critic_net")
    critic_target = checkpoint.get("critic_target")
    if actor_net is None:
        raise RuntimeError("Checkpoint does not contain actor_net parameters.")

    agent.actor_net.load_state_dict(actor_net)
    if critic_net is not None:
        agent.critic_net.load_state_dict(critic_net)
    if critic_target is not None:
        agent.critic_target.load_state_dict(critic_target)
    if checkpoint.get("state_norm") is not None:
        agent.state_normalize = deepcopy(checkpoint["state_norm"])


def _extract_checkpoint_configs(checkpoint: object) -> dict:
    out = {}
    if not isinstance(checkpoint, dict):
        return out

    cfg_obj = checkpoint.get("configs")
    if cfg_obj is None:
        return out

    for key in ("discrete", "observation_shape", "action_dim", "gamma", "dist_type", "state_norm"):
        if hasattr(cfg_obj, key):
            out[key] = getattr(cfg_obj, key)
    for key in ("actor_layers", "critic_layers"):
        if hasattr(cfg_obj, key):
            value = getattr(cfg_obj, key)
            out[key] = dict(value) if isinstance(value, dict) else value
    return out


def _infer_actor_output_size(checkpoint: object) -> Optional[int]:
    if not isinstance(checkpoint, dict):
        return None
    actor_sd = checkpoint.get("actor_net")
    if not isinstance(actor_sd, dict):
        return None
    weight = actor_sd.get("net.4.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])

    weight_tensors = [
        value
        for key, value in actor_sd.items()
        if key.endswith("weight") and isinstance(value, torch.Tensor) and value.ndim == 2
    ]
    if not weight_tensors:
        return None
    return int(weight_tensors[-1].shape[0])


def _infer_primitive_size(npz_path: str) -> Optional[int]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        actions = data["actions"]
        if actions.ndim >= 1:
            return int(actions.shape[0])
    except Exception:
        return None
    return None


def _resolve_adaptive_library_from_checkpoint_dir(checkpoint_path: str) -> Optional[str]:
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    active_path = checkpoint_dir / "adaptive_primitives" / "active_version.json"
    if not active_path.exists():
        return None

    try:
        with active_path.open("r", encoding="utf-8") as file_obj:
            version_id = str(json.load(file_obj).get("version_id", "")).strip()
    except Exception:
        return None

    if not version_id:
        return None

    candidate = checkpoint_dir / "adaptive_primitives" / "versions" / f"primitives_v{version_id}.npz"
    if candidate.exists():
        return str(candidate)
    return None


def _find_matching_primitive_library(
    src_dir: str,
    expected_size: int,
    configured_library_path: str,
    preferred_dir: Optional[str] = None,
) -> Optional[str]:
    candidates: List[Path] = []

    if preferred_dir:
        preferred_root = Path(preferred_dir)
        if preferred_root.exists():
            candidates.extend(preferred_root.rglob("*.npz"))

    configured = Path(src_dir) / configured_library_path
    if configured.exists():
        candidates.append(configured)
    else:
        configured = Path(configured_library_path)
        if configured.exists():
            candidates.append(configured)

    log_root = Path(src_dir) / "log" / "exp"
    if log_root.exists():
        candidates.extend(log_root.rglob("*.npz"))

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate.resolve())

    matches: List[Tuple[float, str]] = []
    for candidate in unique_candidates:
        primitive_size = _infer_primitive_size(str(candidate))
        if primitive_size != int(expected_size):
            continue
        try:
            modified_time = candidate.stat().st_mtime
        except OSError:
            modified_time = 0.0
        matches.append((modified_time, str(candidate)))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _load_ppo_modules(ppo_root: Optional[str]) -> Dict[str, Any]:
    root = _discover_ppo_root(ppo_root)
    cache_key = str(root)
    if cache_key in _PPO_IMPORT_CACHE:
        return _PPO_IMPORT_CACHE[cache_key]

    src_root = root / "src"
    inserted = False
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
        inserted = True

    original_sdl_video_driver = os.environ.get("SDL_VIDEODRIVER")

    try:
        modules = {
            "root": root,
            "src_root": src_root,
            "configs": importlib.import_module("configs"),
            "ppo_agent": importlib.import_module("model.agent.ppo_agent"),
            "guidance": importlib.import_module("env.global_guidance"),
            "lidar": importlib.import_module("env.lidar_simulator"),
            "primitives": importlib.import_module("primitives.library"),
        }
    finally:
        if original_sdl_video_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = original_sdl_video_driver
        if inserted:
            sys.path.pop(0)

    _PPO_IMPORT_CACHE[cache_key] = modules
    return modules


def _coerce_path_points(raw_points) -> List[Tuple[float, float]]:
    if raw_points is None:
        return []

    points: List[Tuple[float, float]] = []
    for point_like in raw_points:
        x_coord = float(point_like[0])
        y_coord = float(point_like[1])
        if points:
            prev_x, prev_y = points[-1]
            if math.hypot(x_coord - prev_x, y_coord - prev_y) <= 1e-6:
                continue
        points.append((x_coord, y_coord))
    return points


def _dedupe_rollout_points(states: Sequence[ArticulatedState]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for state in states:
        point = (float(state.x), float(state.y))
        if points:
            prev_x, prev_y = points[-1]
            if math.hypot(point[0] - prev_x, point[1] - prev_y) <= 1e-6:
                continue
        points.append(point)
    return points


@dataclass
class PPOPlanningResult:
    primitive_id: int
    primitive_actions: np.ndarray
    control_actions: np.ndarray
    observation: np.ndarray
    reference: ArticulatedReferenceTrajectory
    metadata: Dict[str, Any]


class PPOPrimitivePathPlanner:
    def __init__(
        self,
        checkpoint_path: str,
        ppo_root: Optional[str] = None,
        control_interval_ms: int = 100,
        replan_every_steps: int = 1,
        deterministic: bool = True,
    ):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.modules = _load_ppo_modules(ppo_root)
        self.ppo_root = str(self.modules["root"])
        self.control_interval_ms = max(int(control_interval_ms), 1)
        self.replan_every_steps = max(int(replan_every_steps), 1)
        self.deterministic = bool(deterministic)
        self._guidance_points_signature = None
        self.max_plan_primitives = 48
        self.goal_tolerance_m = 2.0
        self.max_candidate_primitives = 16
        self.min_progress_m = 0.05
        self.max_stagnation_steps = 4
        self.use_action_mask = bool(getattr(self.modules["configs"], "USE_ACTION_MASK", True))
        self.action_mask_update_every_k = max(
            int(getattr(self.modules["configs"], "ACTION_MASK_UPDATE_EVERY_K", 1)),
            1,
        )
        self.action_mask_mode = self._normalize_action_mask_mode(
            getattr(self.modules["configs"], "ACTION_MASK_MODE", "hybrid")
        )
        self._action_mask_cached: Optional[np.ndarray] = None
        self._action_mask_calls_since_update = 0

        self.ppo_configs = self.modules["configs"]
        self._load_runtime_assets()

    @staticmethod
    def _normalize_action_mask_mode(mode: object) -> str:
        normalized = str(mode).strip().lower()
        if normalized == "hyrbid":
            normalized = "hybrid"
        if normalized not in {"fast_only", "hybrid", "full"}:
            normalized = "hybrid"
        return normalized

    def _load_runtime_assets(self):
        checkpoint = _load_checkpoint(self.checkpoint_path, map_location="cpu")
        checkpoint_configs = _extract_checkpoint_configs(checkpoint)
        actor_output_size = _infer_actor_output_size(checkpoint)
        if actor_output_size is None or actor_output_size <= 0:
            raise RuntimeError("Cannot infer actor output size from PPO checkpoint.")

        preferred_library = _resolve_adaptive_library_from_checkpoint_dir(self.checkpoint_path)
        expected_action_dim = int(checkpoint_configs.get("action_dim", actor_output_size))

        library_path = None
        if preferred_library is not None:
            if _infer_primitive_size(preferred_library) == expected_action_dim:
                library_path = preferred_library

        if library_path is None:
            preferred_dir = str(Path(preferred_library).parent) if preferred_library else None
            library_path = _find_matching_primitive_library(
                src_dir=str(self.modules["src_root"]),
                expected_size=expected_action_dim,
                configured_library_path=str(self.ppo_configs.PRIMITIVE_LIBRARY_PATH),
                preferred_dir=preferred_dir,
            )

        if library_path is None:
            raise RuntimeError(
                "Cannot locate a primitive library whose size matches the PPO actor output."
            )

        primitive_library = self.modules["primitives"].load_library(library_path)
        if int(primitive_library.size) != actor_output_size:
            raise RuntimeError(
                "Checkpoint actor output and primitive library size mismatch: "
                f"actor={actor_output_size}, library={primitive_library.size}."
            )

        primitive_interval_ms = int(round(float(self.ppo_configs.NUM_STEP) * float(self.ppo_configs.STEP_LENGTH) * 1000.0))
        if primitive_interval_ms <= 0:
            primitive_interval_ms = 200

        observation_shape = checkpoint_configs.get("observation_shape")
        if not observation_shape:
            observation_shape = (int(self.ppo_configs.LIDAR_NUM) + 7 + 2 + int(self.ppo_configs.GUIDANCE_FEATURE_DIM),)

        actor_layers = dict(checkpoint_configs.get("actor_layers", self.ppo_configs.ACTOR_CONFIGS))
        critic_layers = dict(checkpoint_configs.get("critic_layers", self.ppo_configs.CRITIC_CONFIGS))
        actor_layers["input_dim"] = int(observation_shape[0])
        actor_layers["output_size"] = int(primitive_library.size)
        actor_layers["use_tanh_output"] = False
        critic_layers["input_dim"] = int(observation_shape[0])

        agent_configs = {
            "discrete": True,
            "observation_shape": tuple(observation_shape),
            "action_dim": int(primitive_library.size),
            "hidden_size": 64,
            "activation": "tanh",
            "dist_type": checkpoint_configs.get("dist_type", "gaussian"),
            "state_norm": bool(checkpoint_configs.get("state_norm", True)),
            "save_params": False,
            "load_params": True,
            "actor_layers": actor_layers,
            "critic_layers": critic_layers,
            "gamma": float(checkpoint_configs.get("gamma", self.ppo_configs.GAMMA_BASE ** primitive_library.horizon)),
        }

        ppo_agent_cls = self.modules["ppo_agent"].PPOAgent
        lidar_cls = self.modules["lidar"].LidarSimlator
        guidance_cls = self.modules["guidance"].SoftGlobalGuidance

        self.agent = ppo_agent_cls(agent_configs, discrete=True, load_params=True)
        _restore_agent_for_inference(self.agent, checkpoint)
        self.primitive_library = primitive_library
        self.primitive_library_path = str(Path(library_path).resolve())
        self.observation_dim = int(observation_shape[0])
        self.primitive_interval_ms = primitive_interval_ms
        self.lidar = lidar_cls(float(self.ppo_configs.LIDAR_RANGE), int(self.ppo_configs.LIDAR_NUM))
        self._front_box = build_front_vehicle_box(
            width=PPO_WIDTH,
            hitch_offset=PPO_HITCH_OFFSET,
            front_overhang=PPO_FRONT_OVERHANG,
        )
        self._rear_box = build_rear_vehicle_box(
            width=PPO_WIDTH,
            trailer_length=PPO_TRAILER_LENGTH,
            rear_overhang=PPO_REAR_OVERHANG,
        )
        self.global_guidance = None
        if bool(getattr(self.ppo_configs, "ENABLE_GLOBAL_SOFT_GUIDANCE", False)):
            self.global_guidance = guidance_cls(
                grid_resolution=float(self.ppo_configs.GUIDANCE_GRID_RESOLUTION),
                obstacle_inflation=float(self.ppo_configs.GUIDANCE_OBS_INFLATION),
                map_margin=float(self.ppo_configs.GUIDANCE_MAP_MARGIN),
                lookahead_base=float(self.ppo_configs.GUIDANCE_LOOKAHEAD_BASE),
                lookahead_speed_gain=float(self.ppo_configs.GUIDANCE_LOOKAHEAD_SPEED_GAIN),
                lookahead_min=float(self.ppo_configs.GUIDANCE_LOOKAHEAD_MIN),
                lookahead_max=float(self.ppo_configs.GUIDANCE_LOOKAHEAD_MAX),
                progress_search_window=int(self.ppo_configs.GUIDANCE_PROGRESS_WINDOW),
                min_clearance_m=float(self.ppo_configs.GUIDANCE_MIN_CLEARANCE_M),
                full_clearance_m=float(self.ppo_configs.GUIDANCE_FULL_CLEARANCE_M),
                near_obs_dist_m=float(self.ppo_configs.GUIDANCE_NEAR_OBS_DIST_M),
                max_dense_ratio=float(self.ppo_configs.GUIDANCE_MAX_DENSE_RATIO),
            )

    def _obstacle_geometries(self, scene) -> List[Any]:
        geometries = []
        for area in scene.map_.areas.values():
            subtype = getattr(area, "subtype", None)
            if subtype in {"obstacle", "wall"} and getattr(area, "geometry", None) is not None:
                geometries.append(area.geometry)
        return geometries

    def _target_features(self, scene, current_state: ArticulatedState) -> np.ndarray:
        dest_state = scene.map_.customs.get("dest_state")
        if dest_state is not None:
            target_x = float(dest_state.x)
            target_y = float(dest_state.y)
            target_heading = float(dest_state.heading)
        elif scene.goal_point is not None:
            target_x = float(scene.goal_point[0])
            target_y = float(scene.goal_point[1])
            target_heading = float(scene.target_heading or current_state.heading)
        else:
            target_x = float(current_state.x)
            target_y = float(current_state.y)
            target_heading = float(current_state.heading)

        dx = target_x - float(current_state.x)
        dy = target_y - float(current_state.y)
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        relative_angle = angle_to_target - float(current_state.heading)
        relative_heading = float(target_heading) - float(current_state.heading)
        articulation = float(current_state.articulation_angle)

        return np.array(
            [
                distance / float(self.ppo_configs.MAX_DIST_TO_DEST),
                math.cos(relative_angle),
                math.sin(relative_angle),
                math.cos(relative_heading),
                math.sin(relative_heading),
                math.cos(articulation),
                math.sin(articulation),
            ],
            dtype=np.float64,
        )

    def _guidance_features(
        self,
        scene,
        current_state: ArticulatedState,
        lidar_obs: np.ndarray,
    ) -> np.ndarray:
        feature_dim = int(getattr(self.ppo_configs, "GUIDANCE_FEATURE_DIM", 0))
        if self.global_guidance is None or feature_dim <= 0:
            return np.zeros((0,), dtype=np.float64)

        scene_meta = scene.map_.customs.get("scene_meta") or {}
        guidance_points = _coerce_path_points(scene_meta.get("guidance_path_points"))
        if guidance_points:
            signature = tuple(guidance_points)
            if signature != self._guidance_points_signature:
                self.global_guidance.set_precomputed_path(guidance_points)
                self._guidance_points_signature = signature

        try:
            return self.global_guidance.get_soft_hint(
                state_x=float(current_state.x),
                state_y=float(current_state.y),
                heading=float(current_state.heading),
                speed=0.0 if current_state.speed is None else float(current_state.speed),
                lidar_norm=lidar_obs,
                lidar_range=float(self.ppo_configs.LIDAR_RANGE),
            )
        except Exception:
            return np.zeros((feature_dim,), dtype=np.float64)

    def build_observation(self, scene, participant, state: Optional[ArticulatedState] = None) -> np.ndarray:
        source_state = participant.current_state if state is None else state
        current_state = participant.physics_model.ensure_articulated_state(source_state)
        lidar_obs = self.lidar.get_observation(current_state, self._obstacle_geometries(scene))
        lidar_obs = np.asarray(lidar_obs, dtype=np.float64) / float(self.ppo_configs.LIDAR_RANGE)
        target_obs = self._target_features(scene, current_state)

        max_speed = max(abs(float(self.ppo_configs.VALID_SPEED[0])), abs(float(self.ppo_configs.VALID_SPEED[1])))
        max_steer = max(abs(float(self.ppo_configs.VALID_STEER[0])), abs(float(self.ppo_configs.VALID_STEER[1])))
        vel_obs = np.array(
            [
                (0.0 if current_state.speed is None else float(current_state.speed)) / max(max_speed, 1e-6),
                float(current_state.steering) / max(max_steer, 1e-6),
            ],
            dtype=np.float64,
        )
        guidance_obs = self._guidance_features(scene, current_state, lidar_obs)
        observation = np.concatenate([lidar_obs, target_obs, vel_obs, guidance_obs]).astype(np.float64)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"PPO observation shape mismatch: expected {(self.observation_dim,)}, got {observation.shape}."
            )
        return observation

    def _goal_distance(self, scene, state: ArticulatedState) -> float:
        dest_state = scene.map_.customs.get("dest_state")
        if dest_state is not None:
            goal_x = float(dest_state.x)
            goal_y = float(dest_state.y)
        elif scene.goal_point is not None:
            goal_x = float(scene.goal_point[0])
            goal_y = float(scene.goal_point[1])
        else:
            return 0.0
        return float(math.hypot(float(state.x) - goal_x, float(state.y) - goal_y))

    def _state_boxes(self, state: ArticulatedState):
        articulated_state = state
        articulated_state.update_trailer_loc(PPO_HITCH_OFFSET, PPO_TRAILER_LENGTH)
        front_ring, rear_ring = articulated_state.create_boxes(
            self._front_box,
            self._rear_box,
            PPO_HITCH_OFFSET,
            PPO_TRAILER_LENGTH,
        )
        return front_ring, rear_ring

    def _state_hits_obstacle(self, scene, state: ArticulatedState) -> bool:
        obstacles = self._obstacle_geometries(scene)
        if not obstacles:
            return False

        front_ring, rear_ring = self._state_boxes(state)
        for obstacle in obstacles:
            if front_ring.intersects(obstacle) or rear_ring.intersects(obstacle):
                return True
        return False

    def _state_out_of_bounds(self, scene, state: ArticulatedState) -> bool:
        min_x, max_x, min_y, max_y = scene.map_.boundary
        return bool(float(state.x) < min_x or float(state.x) > max_x or float(state.y) < min_y or float(state.y) > max_y)

    def _ranked_primitive_ids(self, observation: np.ndarray, action_mask: Optional[np.ndarray] = None) -> np.ndarray:
        action_dist = self.agent._actor_forward(observation, action_mask=action_mask)
        probabilities = action_dist.probs.detach().cpu().numpy().reshape(-1)
        ranked_ids = np.argsort(probabilities)[::-1]
        limit = min(int(self.max_candidate_primitives), int(ranked_ids.shape[0]))
        return ranked_ids[:limit]

    def _is_rollout_feasible(self, scene, rollout_states: Sequence[ArticulatedState]) -> bool:
        for rollout_state in rollout_states[1:]:
            if self._state_out_of_bounds(scene, rollout_state) or self._state_hits_obstacle(scene, rollout_state):
                return False
        return True

    def _compute_action_mask(
        self,
        scene,
        participant,
        current_state: ArticulatedState,
    ) -> np.ndarray:
        if (
            self._action_mask_cached is not None
            and self._action_mask_calls_since_update < (self.action_mask_update_every_k - 1)
        ):
            self._action_mask_calls_since_update += 1
            return self._action_mask_cached.copy()

        mask = np.zeros(int(self.primitive_library.size), dtype=np.int8)
        for primitive_id in range(int(self.primitive_library.size)):
            primitive_actions = np.asarray(self.primitive_library.get_actions(int(primitive_id)), dtype=np.float64)
            rollout_states = self._rollout_primitive(participant, primitive_actions, state=current_state)
            if self._is_rollout_feasible(scene, rollout_states):
                mask[int(primitive_id)] = 1

        if not mask.any():
            mask[:] = 1

        self._action_mask_cached = mask.copy()
        self._action_mask_calls_since_update = 0
        return mask

    def _choose_closed_loop_primitive(
        self,
        scene,
        participant,
        current_state: ArticulatedState,
        observation: np.ndarray,
    ):
        action_mask = self._compute_action_mask(scene, participant, current_state) if self.use_action_mask else None
        primitive_id, _ = self.agent.choose_action(
            observation,
            deterministic=self.deterministic,
            action_mask=action_mask,
        )
        primitive_id = int(primitive_id)
        primitive_actions = np.asarray(self.primitive_library.get_actions(primitive_id), dtype=np.float64)
        rollout_states = self._rollout_primitive(participant, primitive_actions, state=current_state)

        if not self._is_rollout_feasible(scene, rollout_states):
            ranked_ids = self._ranked_primitive_ids(observation, action_mask=action_mask)
            for ranked_primitive_id in ranked_ids:
                primitive_actions = np.asarray(
                    self.primitive_library.get_actions(int(ranked_primitive_id)),
                    dtype=np.float64,
                )
                rollout_states = self._rollout_primitive(participant, primitive_actions, state=current_state)
                if self._is_rollout_feasible(scene, rollout_states):
                    primitive_id = int(ranked_primitive_id)
                    break

        return primitive_id, primitive_actions, rollout_states, action_mask

    def _rollout_primitive(self, participant, primitive_actions: np.ndarray, state: Optional[ArticulatedState] = None) -> List[ArticulatedState]:
        physics_model = participant.physics_model
        rollout_source = participant.current_state if state is None else state
        rollout_state = physics_model.ensure_articulated_state(rollout_source)
        states = [rollout_state]

        for steering_rate, speed in np.asarray(primitive_actions, dtype=np.float64):
            remaining_ms = float(self.primitive_interval_ms)
            while remaining_ms > 1e-6:
                step_interval_ms = min(float(self.control_interval_ms), remaining_ms)
                rollout_state, _, _ = physics_model.step(
                    rollout_state,
                    steering=float(steering_rate),
                    speed=float(speed),
                    interval=step_interval_ms,
                )
                states.append(rollout_state)
                remaining_ms -= step_interval_ms

        return states

    def _expand_primitive_controls(self, primitive_actions: np.ndarray) -> np.ndarray:
        expanded_controls: List[Tuple[float, float]] = []
        repeat_count = max(int(round(float(self.primitive_interval_ms) / float(self.control_interval_ms))), 1)

        for steering_rate, speed in np.asarray(primitive_actions, dtype=np.float64):
            for _ in range(repeat_count):
                expanded_controls.append((float(steering_rate), float(speed)))

        if not expanded_controls:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(expanded_controls, dtype=np.float64)

    def _select_primitive(
        self,
        scene,
        participant,
        current_state: ArticulatedState,
        observation: np.ndarray,
    ):
        current_goal_distance = self._goal_distance(scene, current_state)
        best_candidate = None

        for primitive_id in self._ranked_primitive_ids(observation):
            primitive_actions = np.asarray(self.primitive_library.get_actions(int(primitive_id)), dtype=np.float64)
            rollout_states = self._rollout_primitive(participant, primitive_actions, state=current_state)
            intermediate_states = rollout_states[1:]
            if not intermediate_states:
                continue

            invalid = False
            for rollout_state in intermediate_states:
                if self._state_out_of_bounds(scene, rollout_state) or self._state_hits_obstacle(scene, rollout_state):
                    invalid = True
                    break
            if invalid:
                continue

            end_state = intermediate_states[-1]
            end_goal_distance = self._goal_distance(scene, end_state)
            progress = current_goal_distance - end_goal_distance
            candidate = {
                "primitive_id": int(primitive_id),
                "primitive_actions": primitive_actions,
                "rollout_states": rollout_states,
                "end_goal_distance": float(end_goal_distance),
                "progress": float(progress),
            }

            if best_candidate is None:
                best_candidate = candidate
                continue

            if candidate["progress"] > best_candidate["progress"] + 1e-6:
                best_candidate = candidate
                continue

            if abs(candidate["progress"] - best_candidate["progress"]) <= 1e-6 and candidate["end_goal_distance"] < best_candidate["end_goal_distance"]:
                best_candidate = candidate

        return best_candidate

    def _reference_from_rollout(
        self,
        states: Sequence[ArticulatedState],
        primitive_ids: Sequence[int],
        observation: np.ndarray,
    ) -> ArticulatedReferenceTrajectory:
        points = _dedupe_rollout_points(states)
        if len(points) < 2:
            state = states[0]
            points = [
                (float(state.x), float(state.y)),
                (
                    float(state.x) + math.cos(float(state.heading)) * 0.5,
                    float(state.y) + math.sin(float(state.heading)) * 0.5,
                ),
            ]

        scene_guidance = []
        metadata = {
            "reference_path_source": "ppo_primitive_global_plan",
            "primitive_sequence": [int(primitive_id) for primitive_id in primitive_ids],
            "primitive_horizon": int(self.primitive_library.horizon),
            "checkpoint_path": self.checkpoint_path,
            "primitive_library_path": self.primitive_library_path,
            "ppo_observation_dim": int(observation.shape[0]),
            "plan_num_primitives": int(len(primitive_ids)),
        }
        return ArticulatedReferenceTrajectory(
            states=list(states),
            path=LineString(points),
            anchors=list(points),
            guidance_points=scene_guidance,
            metadata=metadata,
        )

    def plan(self, scene, participant) -> PPOPlanningResult:
        planning_state = participant.physics_model.ensure_articulated_state(participant.current_state)
        observation = self.build_observation(scene, participant, state=planning_state)
        primitive_id, primitive_actions, rollout_states, action_mask = self._choose_closed_loop_primitive(
            scene,
            participant,
            planning_state,
            observation,
        )

        reference = self._reference_from_rollout(rollout_states, [primitive_id], observation)
        metadata = dict(reference.metadata)
        metadata["primitive_actions_shape"] = tuple(int(dim) for dim in primitive_actions.shape)
        metadata["primitive_id"] = int(primitive_id)
        metadata["planning_mode"] = "closed_loop_policy"
        metadata["action_mask_used"] = bool(self.use_action_mask)
        metadata["action_mask_mode"] = self.action_mask_mode
        metadata["action_mask_update_every_k"] = int(self.action_mask_update_every_k)
        metadata["action_mask_feasible_count"] = None if action_mask is None else int(np.count_nonzero(action_mask))
        metadata["replan_every_steps"] = int(self.replan_every_steps)
        control_actions = self._expand_primitive_controls(primitive_actions)
        metadata["control_actions_shape"] = tuple(int(dim) for dim in control_actions.shape)
        return PPOPlanningResult(
            primitive_id=int(primitive_id),
            primitive_actions=primitive_actions,
            control_actions=control_actions,
            observation=observation,
            reference=reference,
            metadata=metadata,
        )