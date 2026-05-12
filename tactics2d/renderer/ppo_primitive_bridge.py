import importlib
import json
import math
import os
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon

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
            "primitive_index": importlib.import_module("primitives.primitive_index"),
            "primitive_ray_safety": importlib.import_module("primitives.primitive_ray_safety"),
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


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


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
        safety_stop_distance_m: Optional[float] = None,
        safety_forward_sector_half_angle_deg: Optional[float] = None,
        safety_collision_buffer_m: Optional[float] = None,
        replan_on_emergency_stop: Optional[bool] = None,
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
        self.mask_use_fast_prune = bool(getattr(self.modules["configs"], "ACTION_MASK_USE_FAST_PRUNE", True))
        self.occupancy_inflation_radius = float(
            getattr(self.modules["configs"], "OCCUPANCY_INFLATION_RADIUS", 1.8)
        )
        self.soft_mask_gamma = float(getattr(self.modules["configs"], "SOFT_MASK_GAMMA", 1.5))
        self.soft_mask_eps = float(getattr(self.modules["configs"], "SOFT_MASK_EPS", 0.01))
        self.soft_mask_terminal_gamma = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_GAMMA", 0.5)
        )
        self.soft_mask_terminal_eps = float(getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_EPS", 0.05))
        self.soft_mask_terminal_radius = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_RADIUS", 4.0)
        )
        self.soft_mask_min_action_count = int(getattr(self.modules["configs"], "SOFT_MASK_MIN_ACTION_COUNT", 6))
        self.soft_mask_terminal_heading_scale = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_HEADING_SCALE", math.radians(35.0))
        )
        self.soft_mask_terminal_articulation_scale = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_ARTICULATION_SCALE", math.radians(35.0))
        )
        self.soft_mask_terminal_weight_min = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_WEIGHT_MIN", 0.60)
        )
        self.soft_mask_terminal_weight_max = float(
            getattr(self.modules["configs"], "SOFT_MASK_TERMINAL_WEIGHT_MAX", 1.25)
        )
        self.safety_stop_distance_m = float(
            getattr(self.modules["configs"], "SAFETY_STOP_DISTANCE_M", 0.5)
            if safety_stop_distance_m is None
            else safety_stop_distance_m
        )
        self.safety_forward_sector_half_angle = math.radians(
            float(
                getattr(self.modules["configs"], "SAFETY_FORWARD_SECTOR_HALF_ANGLE_DEG", 18.0)
                if safety_forward_sector_half_angle_deg is None
                else safety_forward_sector_half_angle_deg
            )
        )
        self.safety_forward_sector_max_half_angle = math.radians(
            float(getattr(self.modules["configs"], "SAFETY_FORWARD_SECTOR_MAX_HALF_ANGLE_DEG", 55.0))
        )
        self.safety_steering_sector_center_gain = float(
            getattr(self.modules["configs"], "SAFETY_STEERING_SECTOR_CENTER_GAIN", 0.5)
        )
        self.safety_steering_sector_half_angle_gain = float(
            getattr(self.modules["configs"], "SAFETY_STEERING_SECTOR_HALF_ANGLE_GAIN", 1.0)
        )
        self.safety_collision_buffer_m = float(
            getattr(self.modules["configs"], "SAFETY_COLLISION_BUFFER_M", 0.25)
            if safety_collision_buffer_m is None
            else safety_collision_buffer_m
        )
        self.replan_on_emergency_stop = bool(
            getattr(self.modules["configs"], "REPLAN_ON_EMERGENCY_STOP", True)
            if replan_on_emergency_stop is None
            else replan_on_emergency_stop
        )
        self.max_guard_candidates = max(
            1,
            int(getattr(self.modules["configs"], "SOFT_MASK_MAX_GUARD_CANDIDATES", 3)),
        )
        self.emergency_primitive_id = getattr(self.modules["configs"], "SOFT_MASK_EMERGENCY_PRIMITIVE_ID", None)
        self._ray_safety_index = None
        self._safe_prefix_steps_cached: Optional[np.ndarray] = None
        self._last_safe_prefix_steps: Optional[np.ndarray] = None
        self._control_prefix_state_cache: Optional[np.ndarray] = None
        self._control_prefix_cache_repeat_count = 0
        self._control_prefix_cache_control_steps = 0
        self.terminal_heading_tolerance_deg = float(
            getattr(self.modules["configs"], "PRIMITIVE_REFINEMENT_FINAL_HEADING_TOL_DEG", 5.0)
        )
        self.terminal_overlap_target = float(
            getattr(self.modules["configs"], "PRIMITIVE_REFINEMENT_FINAL_OVERLAP_TARGET", 0.75)
        )
        self.terminal_front_overlap_target = float(
            getattr(self.modules["configs"], "PRIMITIVE_REFINEMENT_FRONT_TERMINAL_OK_OVERLAP_TARGET", 0.80)
        )
        self.terminal_rear_overlap_min = float(
            getattr(
                self.modules["configs"],
                "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_REAR_OVERLAP_MIN",
                0.45,
            )
        )
        self.parked_stop_speed_mps = float(
            getattr(self.modules["configs"], "TERMINAL_PARKED_STOP_SPEED_MPS", 0.20)
        )
        self._last_guard_stats: Dict[str, Any] = {}
        self._last_safety_stop_stats: Dict[str, Any] = {
            "stop_triggered": False,
            "stop_reason": None,
            "stop_replan_requested": False,
        }
        self._action_mask_cached: Optional[np.ndarray] = None
        self._action_mask_calls_since_update = 0
        self._action_mask_index = None
        self._action_mask_index_source = "none"
        self._action_mask_inflation_offsets: List[Tuple[int, int]] = []
        self._last_action_mask_stats: Dict[str, Any] = {
            "precomputed_available": False,
            "precomputed_used": False,
            "precomputed_candidate_count": None,
            "precomputed_fallback_to_full": False,
            "precomputed_index_kind": None,
            "precomputed_index_source": "none",
            "ray_safety_available": False,
            "soft_mask_ms": None,
            "soft_effective_action_count": None,
        }

        self.ppo_configs = self.modules["configs"]
        self._load_runtime_assets()

    @staticmethod
    def _normalize_action_mask_mode(mode: object) -> str:
        normalized = str(mode).strip().lower()
        if normalized == "hyrbid":
            normalized = "hybrid"
        if normalized in {"soft", "ray_soft"}:
            normalized = "soft_ray"
        if normalized not in {"fast_only", "hybrid", "full", "soft_ray"}:
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

        action_mask_index = getattr(primitive_library, "mask_index", None)
        action_mask_index_source = "library" if action_mask_index is not None else "none"
        if action_mask_index is None and self.mask_use_fast_prune:
            try:
                action_mask_index = self.modules["primitive_index"].build_approx_index_from_deltas(
                    actions=np.asarray(getattr(primitive_library, "actions"), dtype=np.float64),
                    deltas=np.asarray(getattr(primitive_library, "deltas"), dtype=np.float64),
                    grid_resolution=float(getattr(self.ppo_configs, "GRID_RESOLUTION", 0.6)),
                    x_min=-6.0,
                    x_max=12.0,
                    y_min=-9.0,
                    y_max=9.0,
                    group_prefix_steps=max(1, int(round(float(primitive_library.horizon) * 0.3))),
                )
                action_mask_index_source = "approx"
            except Exception:
                action_mask_index = None
                action_mask_index_source = "none"

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
            "soft_mask_logit_lambda": float(getattr(self.ppo_configs, "SOFT_MASK_LOGIT_LAMBDA", 1.0)),
            "soft_mask_small_value": float(getattr(self.ppo_configs, "SOFT_MASK_SMALL_VALUE", 1e-8)),
        }

        ppo_agent_cls = self.modules["ppo_agent"].PPOAgent
        lidar_cls = self.modules["lidar"].LidarSimlator
        guidance_cls = self.modules["guidance"].SoftGlobalGuidance

        self.agent = ppo_agent_cls(agent_configs, discrete=True, load_params=True)
        _restore_agent_for_inference(self.agent, checkpoint)
        self.primitive_library = primitive_library
        self.primitive_library_path = str(Path(library_path).resolve())
        self._action_mask_index = action_mask_index
        self._action_mask_index_source = action_mask_index_source
        self._ray_safety_index = getattr(primitive_library, "ray_safety_index", None)
        self._action_mask_inflation_offsets = self._build_action_mask_inflation_offsets(self._action_mask_index)
        self._last_action_mask_stats = {
            "precomputed_available": self._action_mask_index is not None,
            "precomputed_used": False,
            "precomputed_candidate_count": None,
            "precomputed_fallback_to_full": False,
            "precomputed_index_kind": None
            if self._action_mask_index is None
            else str(getattr(self._action_mask_index, "index_kind", "unknown")),
            "precomputed_index_source": self._action_mask_index_source,
            "ray_safety_available": self._ray_safety_index is not None,
            "soft_mask_ms": None,
            "soft_effective_action_count": None,
        }
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

    def _build_action_mask_inflation_offsets(self, grid_index) -> List[Tuple[int, int]]:
        if grid_index is None:
            return []

        resolution = max(float(getattr(grid_index, "grid_resolution", 0.0)), 1e-6)
        radius = max(0.0, float(self.occupancy_inflation_radius))
        radius_in_cells = int(math.ceil(radius / resolution))
        offsets: List[Tuple[int, int]] = []
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                if (dx * dx + dy * dy) * (resolution * resolution) <= radius * radius + 1e-9:
                    offsets.append((dx, dy))
        return offsets

    def _control_repeat_count(self) -> int:
        return max(int(round(float(self.primitive_interval_ms) / float(self.control_interval_ms))), 1)

    @staticmethod
    def _state_cache_row(state: ArticulatedState) -> np.ndarray:
        return np.array(
            [
                float(state.x),
                float(state.y),
                float(state.heading),
                float(state.rear_heading),
                0.0 if state.speed is None else float(state.speed),
                float(getattr(state, "steering", 0.0)),
            ],
            dtype=np.float64,
        )

    def _ensure_control_prefix_cache(self, participant) -> None:
        repeat_count = self._control_repeat_count()
        if (
            self._control_prefix_state_cache is not None
            and self._control_prefix_cache_repeat_count == repeat_count
        ):
            return

        actions = np.asarray(getattr(self.primitive_library, "actions"), dtype=np.float64)
        total_primitives = int(actions.shape[0])
        total_control_steps = int(actions.shape[1]) * repeat_count
        cache = np.zeros((total_primitives, total_control_steps + 1, 6), dtype=np.float64)
        physics_model = participant.physics_model

        for primitive_id in range(total_primitives):
            state = ArticulatedState(
                frame=0,
                x=0.0,
                y=0.0,
                heading=0.0,
                speed=0.0,
                accel=0.0,
                rear_heading=0.0,
                steering=0.0,
            )
            state = physics_model.ensure_articulated_state(state)
            cache[primitive_id, 0] = self._state_cache_row(state)
            control_idx = 0
            for steering_rate, speed in actions[primitive_id]:
                for _ in range(repeat_count):
                    state, _, _ = physics_model.step(
                        state=state,
                        steering=float(steering_rate),
                        speed=float(speed),
                        interval=self.control_interval_ms,
                    )
                    control_idx += 1
                    cache[primitive_id, control_idx] = self._state_cache_row(state)

        self._control_prefix_state_cache = cache
        self._control_prefix_cache_repeat_count = repeat_count
        self._control_prefix_cache_control_steps = total_control_steps

    def _compose_cached_state(
        self,
        current_state: ArticulatedState,
        cache_row: np.ndarray,
        control_step_index: int,
    ) -> ArticulatedState:
        base_heading = float(current_state.heading)
        local_x = float(cache_row[0])
        local_y = float(cache_row[1])
        cos_heading = math.cos(base_heading)
        sin_heading = math.sin(base_heading)
        world_x = float(current_state.x) + cos_heading * local_x - sin_heading * local_y
        world_y = float(current_state.y) + sin_heading * local_x + cos_heading * local_y
        front_heading = _wrap_pi(base_heading + float(cache_row[2]))
        rear_heading = _wrap_pi(float(current_state.rear_heading) + float(cache_row[3]))
        return ArticulatedState(
            frame=int(current_state.frame + control_step_index * self.control_interval_ms),
            x=world_x,
            y=world_y,
            heading=front_heading,
            speed=float(cache_row[4]),
            accel=0.0,
            rear_heading=rear_heading,
            steering=float(cache_row[5]),
        )

    def _cached_prefix_state(
        self,
        participant,
        current_state: ArticulatedState,
        primitive_id: int,
        control_prefix_steps: int,
    ) -> ArticulatedState:
        self._ensure_control_prefix_cache(participant)
        if self._control_prefix_state_cache is None:
            return current_state
        max_steps = min(
            max(int(control_prefix_steps), 0),
            int(self._control_prefix_state_cache.shape[1] - 1),
        )
        if max_steps <= 0:
            return current_state
        cache_row = self._control_prefix_state_cache[int(primitive_id), max_steps]
        return self._compose_cached_state(current_state, cache_row, max_steps)

    def _cached_prefix_rollout(
        self,
        participant,
        current_state: ArticulatedState,
        primitive_id: int,
        control_prefix_steps: int,
    ) -> List[ArticulatedState]:
        self._ensure_control_prefix_cache(participant)
        if self._control_prefix_state_cache is None:
            return [current_state]
        max_steps = min(
            max(int(control_prefix_steps), 0),
            int(self._control_prefix_state_cache.shape[1] - 1),
        )
        states = [current_state]
        for control_idx in range(1, max_steps + 1):
            cache_row = self._control_prefix_state_cache[int(primitive_id), control_idx]
            states.append(self._compose_cached_state(current_state, cache_row, control_idx))
        return states

    def _stationary_prefix_rollout(
        self,
        current_state: ArticulatedState,
        control_prefix_steps: int,
    ) -> List[ArticulatedState]:
        states = [current_state]
        for control_idx in range(1, max(int(control_prefix_steps), 0) + 1):
            states.append(
                ArticulatedState(
                    frame=int(current_state.frame + control_idx * self.control_interval_ms),
                    x=float(current_state.x),
                    y=float(current_state.y),
                    heading=float(current_state.heading),
                    speed=0.0,
                    accel=0.0,
                    rear_heading=float(current_state.rear_heading),
                    steering=0.0,
                )
            )
        return states

    def _compute_safe_prefix_steps(self, observation: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        index = getattr(self, "_ray_safety_index", None)
        if observation is None or index is None:
            return None, {}

        lidar_num = int(getattr(self.ppo_configs, "LIDAR_NUM", 120))
        lidar_range = float(getattr(self.ppo_configs, "LIDAR_RANGE", 30.0))
        lidar = np.asarray(observation, dtype=np.float64).reshape(-1)[:lidar_num]
        if lidar.size < lidar_num:
            padded = np.ones((lidar_num,), dtype=np.float64)
            padded[: lidar.size] = lidar
            lidar = padded

        dist_obs = np.clip(lidar, 0.0, 1.0) * lidar_range
        dist_star = np.asarray(index.dist_star, dtype=np.float32)
        ray_count = min(int(dist_star.shape[2]), int(dist_obs.shape[0]))
        safe_by_ray = dist_star[:, :, :ray_count] <= dist_obs[:ray_count][None, None, :]
        safe_step = np.all(safe_by_ray, axis=2)
        prefix_safe = np.cumprod(safe_step.astype(np.int8), axis=1)
        prefix_steps = np.sum(prefix_safe, axis=1).astype(np.int32)
        debug = {
            "positive_step_count": int(np.count_nonzero(prefix_steps > 0)),
            "safe_step_len_mean": float(np.mean(prefix_steps)) if prefix_steps.size else 0.0,
            "safe_step_len_min": float(np.min(prefix_steps)) if prefix_steps.size else 0.0,
            "safe_step_len_max": float(np.max(prefix_steps)) if prefix_steps.size else 0.0,
        }
        return prefix_steps, debug

    def _soft_mask_from_prefix_steps(self, prefix_steps: np.ndarray, gamma: float, eps: float) -> np.ndarray:
        horizon = max(float(self.primitive_library.horizon), 1.0)
        soft = np.power(np.clip(np.asarray(prefix_steps, dtype=np.float32) / horizon, 0.0, 1.0), float(gamma))
        return np.clip(soft, float(eps), 1.0).astype(np.float32)

    def _terminal_context_active(
        self,
        scene,
        current_state: ArticulatedState,
        positive_count: Optional[int] = None,
    ) -> bool:
        terminal = self._goal_distance(scene, current_state) <= float(self.soft_mask_terminal_radius)
        if positive_count is None:
            return terminal
        return bool(terminal or positive_count < int(self.soft_mask_min_action_count))

    def _target_boxes(self, scene) -> Optional[Tuple[Polygon, ...]]:
        raw_boxes = scene.map_.customs.get("target_boxes")
        if raw_boxes is None:
            return None
        boxes: List[Polygon] = []
        for box_geom in raw_boxes:
            if box_geom is None:
                continue
            boxes.append(box_geom if isinstance(box_geom, Polygon) else Polygon(box_geom))
        return tuple(boxes) if boxes else None

    def _terminal_state_metrics(self, scene, state: ArticulatedState) -> Dict[str, float]:
        dest_state = scene.map_.customs.get("dest_state")
        position_error = self._goal_distance(scene, state)
        heading_error = 0.0
        articulation_error = 0.0
        if dest_state is not None:
            heading_error = abs(_wrap_pi(float(state.heading) - float(dest_state.heading)))
            articulation_error = abs(
                _wrap_pi(float(state.articulation_angle) - float(getattr(dest_state, "articulation_angle", 0.0)))
            )

        front_overlap = 0.0
        rear_overlap = 0.0
        mean_overlap = 0.0
        target_boxes = self._target_boxes(scene)
        if target_boxes:
            current_boxes = tuple(Polygon(box_ring) for box_ring in self._state_boxes(state))
            overlaps = []
            for current_box, target_box in zip(current_boxes, target_boxes):
                area_target = float(target_box.area) + 1e-9
                overlap_area = float(current_box.intersection(target_box).area)
                overlaps.append(float(overlap_area / area_target))
            if overlaps:
                front_overlap = float(overlaps[0])
                rear_overlap = float(overlaps[1]) if len(overlaps) > 1 else float(overlaps[0])
                mean_overlap = float(np.mean(overlaps))

        return {
            "position_error": float(position_error),
            "heading_error_rad": float(heading_error),
            "heading_error_deg": float(np.degrees(heading_error)),
            "articulation_error_rad": float(articulation_error),
            "front_overlap": float(front_overlap),
            "rear_overlap": float(rear_overlap),
            "mean_overlap": float(mean_overlap),
            "speed_abs": abs(0.0 if state.speed is None else float(state.speed)),
        }

    def _is_precisely_parked(self, scene, state: ArticulatedState, *, require_stop_speed: bool) -> bool:
        target_boxes = self._target_boxes(scene)
        if not target_boxes:
            return False
        metrics = self._terminal_state_metrics(scene, state)
        if require_stop_speed and metrics["speed_abs"] > float(self.parked_stop_speed_mps):
            return False
        return bool(
            metrics["heading_error_deg"] <= float(self.terminal_heading_tolerance_deg)
            and metrics["mean_overlap"] >= float(self.terminal_overlap_target)
            and metrics["front_overlap"] >= float(self.terminal_front_overlap_target)
            and metrics["rear_overlap"] >= float(self.terminal_rear_overlap_min)
        )

    def _terminal_candidate_key(self, metrics: Dict[str, float], control_prefix_steps: int) -> Tuple[float, ...]:
        front_overlap_deficit = max(0.0, float(self.terminal_front_overlap_target) - float(metrics["front_overlap"]))
        mean_overlap_deficit = max(0.0, float(self.terminal_overlap_target) - float(metrics["mean_overlap"]))
        rear_overlap_deficit = max(0.0, float(self.terminal_rear_overlap_min) - float(metrics["rear_overlap"]))
        return (
            float(front_overlap_deficit > 1e-6),
            float(mean_overlap_deficit > 1e-6),
            float(rear_overlap_deficit > 1e-6),
            float(front_overlap_deficit),
            float(mean_overlap_deficit),
            float(rear_overlap_deficit),
            float(metrics["position_error"]),
            float(metrics["heading_error_rad"]),
            float(metrics["articulation_error_rad"]),
            float(control_prefix_steps),
        )

    def _build_occupied_cells_from_lidar(self, lidar_norm: np.ndarray):
        if self._action_mask_index is None:
            return None

        lidar_values = np.asarray(lidar_norm, dtype=np.float64).reshape(-1)
        lidar_num = int(getattr(self.ppo_configs, "LIDAR_NUM", lidar_values.shape[0]))
        lidar_range = float(getattr(self.ppo_configs, "LIDAR_RANGE", 30.0))
        lidar_values = lidar_values[:lidar_num]
        if lidar_values.shape[0] == 0:
            return set()

        beam_angles = np.linspace(0.0, 2.0 * math.pi, lidar_values.shape[0], endpoint=False)
        distances = np.clip(lidar_values, 0.0, 1.0) * lidar_range
        hit_mask = distances < (0.98 * lidar_range)
        occupied_cells = set()

        for beam_index in np.nonzero(hit_mask)[0]:
            distance = float(distances[beam_index])
            angle = float(beam_angles[beam_index])
            cell = self._action_mask_index.world_to_cell(
                distance * math.cos(angle),
                distance * math.sin(angle),
            )
            if cell is None:
                continue
            cell_x, cell_y = cell
            for offset_x, offset_y in self._action_mask_inflation_offsets:
                occupied_cells.add((cell_x + offset_x, cell_y + offset_y))

        return occupied_cells

    def _get_action_mask_candidate_ids(
        self,
        observation: Optional[np.ndarray],
        n_actions: int,
    ) -> Optional[np.ndarray]:
        if not self.mask_use_fast_prune or self._action_mask_index is None or observation is None:
            return None

        try:
            lidar_num = int(getattr(self.ppo_configs, "LIDAR_NUM", 120))
            observation_vec = np.asarray(observation, dtype=np.float64).reshape(-1)
            lidar_obs = observation_vec[:lidar_num]
            occupied_cells = self._build_occupied_cells_from_lidar(lidar_obs)
            if occupied_cells is None:
                return None
            candidate_mask = self._action_mask_index.fast_prune_primitives(occupied_cells)
            candidate_mask = np.asarray(candidate_mask, dtype=np.bool_).reshape(-1)
            if candidate_mask.shape[0] != int(n_actions):
                return None
            return np.flatnonzero(candidate_mask).astype(np.int64)
        except Exception:
            return None

    def _terminal_weights_from_observation(self, observation: np.ndarray) -> np.ndarray:
        n_actions = int(self.primitive_library.size)
        deltas = getattr(self.primitive_library, "deltas", None)
        if deltas is None:
            return np.ones((n_actions,), dtype=np.float32)
        deltas = np.asarray(deltas, dtype=np.float64)
        if deltas.shape[0] != n_actions or deltas.shape[1] < 3:
            return np.ones((n_actions,), dtype=np.float32)

        lidar_num = int(getattr(self.ppo_configs, "LIDAR_NUM", 120))
        target = np.asarray(observation, dtype=np.float64).reshape(-1)[lidar_num : lidar_num + 7]
        if target.shape[0] < 7:
            return np.ones((n_actions,), dtype=np.float32)

        dist = float(target[0]) * float(getattr(self.ppo_configs, "MAX_DIST_TO_DEST", 70.0))
        rel_angle = math.atan2(float(target[2]), float(target[1]))
        rel_heading = math.atan2(float(target[4]), float(target[3]))
        articulation = math.atan2(float(target[6]), float(target[5]))
        goal_x = dist * math.cos(rel_angle)
        goal_y = dist * math.sin(rel_angle)

        radius = max(float(self.soft_mask_terminal_radius), 1e-6)
        heading_scale = max(float(self.soft_mask_terminal_heading_scale), 1e-6)
        articulation_scale = max(float(self.soft_mask_terminal_articulation_scale), 1e-6)
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dtheta = deltas[:, 2]
        dgamma = deltas[:, 3] if deltas.shape[1] > 3 else np.zeros_like(dtheta)
        pos_after = np.sqrt((goal_x - dx) * (goal_x - dx) + (goal_y - dy) * (goal_y - dy))
        heading_err = np.abs((rel_heading - dtheta + np.pi) % (2.0 * np.pi) - np.pi)
        articulation_err = np.abs((articulation - dgamma + np.pi) % (2.0 * np.pi) - np.pi)
        progress = np.clip((max(dist, 1e-6) - pos_after) / radius, -1.0, 1.0)
        pos_score = np.exp(-np.square(pos_after / radius))
        heading_score = np.exp(-np.square(heading_err / heading_scale))
        articulation_score = np.exp(-np.square(articulation_err / articulation_scale))
        weights = (
            0.72
            + 0.28 * pos_score
            + 0.22 * heading_score
            + 0.08 * articulation_score
            + 0.18 * np.maximum(progress, 0.0)
            - 0.10 * np.maximum(-progress, 0.0)
        )
        return np.clip(
            weights,
            float(self.soft_mask_terminal_weight_min),
            float(self.soft_mask_terminal_weight_max),
        ).astype(np.float32)

    def _compute_soft_ray_action_mask(
        self,
        scene,
        current_state: ArticulatedState,
        observation: Optional[np.ndarray],
    ) -> np.ndarray:
        started_at = time.perf_counter()
        total_actions = int(self.primitive_library.size)
        eps = float(self.soft_mask_eps)
        index = getattr(self, "_ray_safety_index", None)
        stats = {
            "precomputed_available": self._action_mask_index is not None,
            "precomputed_used": False,
            "precomputed_candidate_count": None,
            "precomputed_fallback_to_full": False,
            "precomputed_index_kind": None
            if self._action_mask_index is None
            else str(getattr(self._action_mask_index, "index_kind", "unknown")),
            "precomputed_index_source": self._action_mask_index_source,
            "ray_safety_available": index is not None,
            "ray_safety_used": False,
            "soft_terminal_reweight_applied": False,
            "soft_mask_fallback": None,
        }

        self._last_safe_prefix_steps = None

        if observation is None:
            mask = np.ones((total_actions,), dtype=np.float32)
            stats["soft_mask_fallback"] = "missing_observation"
            self._safe_prefix_steps_cached = None
        elif index is None:
            candidate_ids = self._get_action_mask_candidate_ids(observation, total_actions)
            mask = np.full((total_actions,), eps, dtype=np.float32)
            if candidate_ids is None:
                mask[:] = 1.0
                stats["soft_mask_fallback"] = "no_ray_safety_no_fast_index"
                stats["precomputed_fallback_to_full"] = True
            else:
                mask[np.asarray(candidate_ids, dtype=np.int64)] = 1.0
                stats["soft_mask_fallback"] = "fast_prune"
                stats["precomputed_used"] = True
                stats["precomputed_candidate_count"] = int(candidate_ids.shape[0])
            self._safe_prefix_steps_cached = None
        else:
            prefix_steps, debug = self._compute_safe_prefix_steps(observation)
            if prefix_steps is None:
                mask = np.ones((total_actions,), dtype=np.float32)
                self._safe_prefix_steps_cached = None
            else:
                mask = self._soft_mask_from_prefix_steps(
                    prefix_steps,
                    gamma=float(self.soft_mask_gamma),
                    eps=eps,
                )
                self._safe_prefix_steps_cached = prefix_steps.copy()
                self._last_safe_prefix_steps = prefix_steps.copy()
            stats.update(debug)
            stats["ray_safety_used"] = True

            positive_count = int(debug.get("positive_step_count", 0))
            terminal_context = self._terminal_context_active(
                scene,
                current_state,
                positive_count=positive_count,
            )
            if terminal_context:
                terminal_mask = self._soft_mask_from_prefix_steps(
                    prefix_steps,
                    gamma=float(self.soft_mask_terminal_gamma),
                    eps=float(self.soft_mask_terminal_eps),
                )
                terminal_debug = dict(debug)
                terminal_weights = self._terminal_weights_from_observation(observation)
                mask = np.clip(
                    terminal_mask * terminal_weights,
                    float(self.soft_mask_terminal_eps),
                    1.0,
                ).astype(np.float32)
                stats.update({f"terminal_{key}": value for key, value in terminal_debug.items()})
                stats["soft_terminal_reweight_applied"] = True

        stats["soft_mask_ms"] = float((time.perf_counter() - started_at) * 1000.0)
        stats["soft_mask_min"] = float(np.min(mask)) if mask.size else 0.0
        stats["soft_mask_max"] = float(np.max(mask)) if mask.size else 0.0
        stats["soft_mask_mean"] = float(np.mean(mask)) if mask.size else 0.0
        stats["soft_effective_action_count"] = int(np.count_nonzero(mask > (float(np.min(mask)) + 1e-6)))
        if self._last_safe_prefix_steps is not None:
            stats["safe_prefix_step_len_mean"] = float(np.mean(self._last_safe_prefix_steps))
            stats["safe_prefix_step_len_min"] = float(np.min(self._last_safe_prefix_steps))
            stats["safe_prefix_step_len_max"] = float(np.max(self._last_safe_prefix_steps))
        self._last_action_mask_stats = stats
        return mask.astype(np.float32)

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

    def _lidar_distances_from_observation(self, observation: Optional[np.ndarray]) -> np.ndarray:
        lidar_num = int(getattr(self.ppo_configs, "LIDAR_NUM", 120))
        lidar_range = float(getattr(self.ppo_configs, "LIDAR_RANGE", 30.0))
        if observation is None:
            return np.full((lidar_num,), lidar_range, dtype=np.float64)

        lidar = np.asarray(observation, dtype=np.float64).reshape(-1)[:lidar_num]
        if lidar.shape[0] < lidar_num:
            padded = np.ones((lidar_num,), dtype=np.float64)
            padded[: lidar.shape[0]] = lidar
            lidar = padded
        return np.clip(lidar, 0.0, 1.0) * lidar_range

    def _directional_sector_for_primitive(
        self,
        current_state: ArticulatedState,
        primitive_actions: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        primitive_actions = np.asarray(primitive_actions, dtype=np.float64).reshape(-1, 2)
        if primitive_actions.shape[0] == 0:
            return 0.0, float(self.safety_forward_sector_half_angle), 0.0, 0.0, 0.0

        steering_rate = float(primitive_actions[0, 0])
        speed = float(primitive_actions[0, 1])
        primitive_dt = max(float(self.primitive_interval_ms) / 1000.0, 1e-6)
        current_steering = 0.0 if getattr(current_state, "steering", None) is None else float(current_state.steering)
        projected_steering = current_steering + steering_rate * primitive_dt
        direction_base = 0.0 if speed >= 0.0 else math.pi
        sector_center = _wrap_pi(direction_base + projected_steering * float(self.safety_steering_sector_center_gain))
        sector_half_angle = float(self.safety_forward_sector_half_angle) + abs(projected_steering) * float(
            self.safety_steering_sector_half_angle_gain
        )
        sector_half_angle = float(
            np.clip(
                sector_half_angle,
                float(self.safety_forward_sector_half_angle),
                float(self.safety_forward_sector_max_half_angle),
            )
        )
        projected_travel = abs(speed) * primitive_dt
        return sector_center, sector_half_angle, projected_travel, steering_rate, speed

    def _distance_to_boundary_along_world_angle(
        self,
        scene,
        current_state: ArticulatedState,
        world_angle: float,
    ) -> float:
        min_x, max_x, min_y, max_y = scene.map_.boundary
        origin_x = float(current_state.x)
        origin_y = float(current_state.y)
        direction_x = math.cos(world_angle)
        direction_y = math.sin(world_angle)
        candidates: List[float] = []

        if abs(direction_x) > 1e-9:
            for edge_x in (min_x, max_x):
                distance = (float(edge_x) - origin_x) / direction_x
                if distance < 0.0:
                    continue
                hit_y = origin_y + distance * direction_y
                if float(min_y) - 1e-6 <= hit_y <= float(max_y) + 1e-6:
                    candidates.append(float(distance))

        if abs(direction_y) > 1e-9:
            for edge_y in (min_y, max_y):
                distance = (float(edge_y) - origin_y) / direction_y
                if distance < 0.0:
                    continue
                hit_x = origin_x + distance * direction_x
                if float(min_x) - 1e-6 <= hit_x <= float(max_x) + 1e-6:
                    candidates.append(float(distance))

        if not candidates:
            return float("inf")
        return float(min(candidates))

    def _directional_clearance(
        self,
        scene,
        current_state: ArticulatedState,
        observation: Optional[np.ndarray],
        sector_center: float,
        sector_half_angle: float,
    ) -> Dict[str, Any]:
        lidar_distances = self._lidar_distances_from_observation(observation)
        if lidar_distances.shape[0] == 0:
            lidar_range = float(getattr(self.ppo_configs, "LIDAR_RANGE", 30.0))
            return {
                "sector_lidar_distance_m": lidar_range,
                "sector_boundary_distance_m": float("inf"),
                "sector_clearance_distance_m": lidar_range,
                "sector_beam_count": 0,
            }

        beam_angles = np.linspace(0.0, 2.0 * math.pi, lidar_distances.shape[0], endpoint=False)
        angular_error = np.abs(np.array([_wrap_pi(angle - sector_center) for angle in beam_angles], dtype=np.float64))
        sector_mask = angular_error <= max(float(sector_half_angle), math.pi / float(lidar_distances.shape[0]))
        if not np.any(sector_mask):
            sector_mask[int(np.argmin(angular_error))] = True
        sector_lidar_distance = float(np.min(lidar_distances[sector_mask]))

        sample_angles = [
            _wrap_pi(sector_center - sector_half_angle),
            _wrap_pi(sector_center),
            _wrap_pi(sector_center + sector_half_angle),
        ]
        boundary_distance = min(
            self._distance_to_boundary_along_world_angle(
                scene,
                current_state,
                float(current_state.heading) + sample_angle,
            )
            for sample_angle in sample_angles
        )
        clearance_distance = float(min(sector_lidar_distance, boundary_distance))
        return {
            "sector_lidar_distance_m": sector_lidar_distance,
            "sector_boundary_distance_m": float(boundary_distance),
            "sector_clearance_distance_m": clearance_distance,
            "sector_beam_count": int(np.count_nonzero(sector_mask)),
        }

    def _evaluate_directional_stop(
        self,
        scene,
        current_state: ArticulatedState,
        observation: Optional[np.ndarray],
        primitive_actions: np.ndarray,
        *,
        store: bool = True,
    ) -> Dict[str, Any]:
        sector_center, sector_half_angle, projected_travel, steering_rate, speed = self._directional_sector_for_primitive(
            current_state,
            primitive_actions,
        )
        clearance = self._directional_clearance(
            scene,
            current_state,
            observation,
            sector_center,
            sector_half_angle,
        )
        predicted_collision_distance = float(projected_travel + float(self.safety_collision_buffer_m))
        continue_will_collide = bool(abs(speed) > 1e-6 and clearance["sector_clearance_distance_m"] <= predicted_collision_distance)
        stop_triggered = bool(
            clearance["sector_clearance_distance_m"] <= float(self.safety_stop_distance_m)
            and continue_will_collide
        )
        stats = {
            "stop_triggered": stop_triggered,
            "stop_reason": "directional_clearance" if stop_triggered else None,
            "stop_replan_requested": bool(stop_triggered and self.replan_on_emergency_stop),
            "stop_lidar_distance_m": float(clearance["sector_lidar_distance_m"]),
            "stop_boundary_distance_m": float(clearance["sector_boundary_distance_m"]),
            "stop_clearance_distance_m": float(clearance["sector_clearance_distance_m"]),
            "stop_sector_center_rad": float(sector_center),
            "stop_sector_half_angle_rad": float(sector_half_angle),
            "stop_sector_beam_count": int(clearance["sector_beam_count"]),
            "stop_projected_travel_m": float(projected_travel),
            "stop_predicted_collision_distance_m": predicted_collision_distance,
            "stop_first_speed": float(speed),
            "stop_first_steering_rate": float(steering_rate),
            "stop_safety_distance_m": float(self.safety_stop_distance_m),
            "stop_continue_will_collide": continue_will_collide,
        }
        if store:
            self._last_safety_stop_stats = stats
        return stats

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
        candidates = self._ranked_primitive_candidates(observation, action_mask=action_mask)
        if not candidates:
            return np.zeros((0,), dtype=np.int64)
        return np.asarray([candidate["primitive_id"] for candidate in candidates], dtype=np.int64)

    def _ranked_primitive_candidates(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        action_dist = self.agent._actor_forward(observation, action_mask=action_mask)
        probabilities = action_dist.probs.detach().cpu().numpy().reshape(-1)
        ranked_ids = np.argsort(probabilities)[::-1]
        limit = min(int(self.max_candidate_primitives), int(ranked_ids.shape[0]))
        return [
            {
                "primitive_id": int(primitive_id),
                "probability": float(probabilities[int(primitive_id)]),
                "rank": int(rank),
            }
            for rank, primitive_id in enumerate(ranked_ids[:limit])
        ]

    def _is_rollout_feasible(self, scene, rollout_states: Sequence[ArticulatedState]) -> bool:
        for rollout_state in rollout_states[1:]:
            if self._state_out_of_bounds(scene, rollout_state) or self._state_hits_obstacle(scene, rollout_state):
                return False
        return True

    def _choose_emergency_primitive_id(self) -> int:
        configured = self.emergency_primitive_id
        if configured is not None:
            try:
                configured_id = int(configured)
                if 0 <= configured_id < int(self.primitive_library.size):
                    return configured_id
            except Exception:
                pass

        actions = np.asarray(getattr(self.primitive_library, "actions"), dtype=np.float64)
        deltas = np.asarray(getattr(self.primitive_library, "deltas"), dtype=np.float64)
        speed_cost = np.mean(np.abs(actions[:, :, 1]), axis=1)
        steer_cost = 0.25 * np.mean(np.abs(actions[:, :, 0]), axis=1)
        delta_cost = 0.15 * np.linalg.norm(deltas[:, :2], axis=1) if deltas.ndim == 2 and deltas.shape[1] >= 2 else 0.0
        cost = speed_cost + steer_cost + delta_cost
        return int(np.argmin(cost))

    def _compute_action_mask(
        self,
        scene,
        participant,
        current_state: ArticulatedState,
        observation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if (
            self.action_mask_mode != "soft_ray"
            and
            self._action_mask_cached is not None
            and self._action_mask_calls_since_update < (self.action_mask_update_every_k - 1)
        ):
            self._action_mask_calls_since_update += 1
            self._last_action_mask_stats = {
                "precomputed_available": self._action_mask_index is not None,
                "precomputed_used": False,
                "precomputed_candidate_count": None,
                "precomputed_fallback_to_full": False,
                "precomputed_index_kind": None
                if self._action_mask_index is None
                else str(getattr(self._action_mask_index, "index_kind", "unknown")),
                "precomputed_index_source": self._action_mask_index_source,
                "ray_safety_available": self._ray_safety_index is not None,
                "soft_mask_ms": None,
                "soft_effective_action_count": None,
            }
            return self._action_mask_cached.copy()

        total_actions = int(self.primitive_library.size)

        if self.action_mask_mode == "soft_ray":
            mask = self._compute_soft_ray_action_mask(scene, current_state, observation)
            self._action_mask_cached = mask.copy()
            self._action_mask_calls_since_update = 0
            return mask

        candidate_ids = None
        precomputed_used = False
        if self.action_mask_mode != "full":
            candidate_ids = self._get_action_mask_candidate_ids(observation, total_actions)
            precomputed_used = candidate_ids is not None

        self._last_action_mask_stats = {
            "precomputed_available": self._action_mask_index is not None,
            "precomputed_used": precomputed_used,
            "precomputed_candidate_count": None if candidate_ids is None else int(candidate_ids.shape[0]),
            "precomputed_fallback_to_full": False,
            "precomputed_index_kind": None
            if self._action_mask_index is None
            else str(getattr(self._action_mask_index, "index_kind", "unknown")),
            "precomputed_index_source": self._action_mask_index_source,
            "ray_safety_available": self._ray_safety_index is not None,
            "soft_mask_ms": None,
            "soft_effective_action_count": None,
        }

        if self.action_mask_mode == "fast_only":
            if candidate_ids is None:
                mask = np.ones(total_actions, dtype=np.int8)
                self._last_action_mask_stats["precomputed_fallback_to_full"] = True
            else:
                mask = np.zeros(total_actions, dtype=np.int8)
                mask[candidate_ids] = 1
            if not mask.any():
                mask[:] = 1
                self._last_action_mask_stats["precomputed_fallback_to_full"] = True
            self._action_mask_cached = mask.copy()
            self._action_mask_calls_since_update = 0
            return mask

        if candidate_ids is None:
            mask = np.ones(total_actions, dtype=np.int8)
            eval_ids = range(total_actions)
        else:
            mask = np.zeros(total_actions, dtype=np.int8)
            eval_ids = candidate_ids

        for primitive_id in eval_ids:
            primitive_actions = np.asarray(self.primitive_library.get_actions(int(primitive_id)), dtype=np.float64)
            stop_stats = self._evaluate_directional_stop(
                scene,
                current_state,
                observation,
                primitive_actions,
                store=False,
            )
            if not bool(stop_stats["stop_triggered"]):
                mask[int(primitive_id)] = 1

        if not mask.any():
            mask[:] = 1
            if candidate_ids is not None:
                self._last_action_mask_stats["precomputed_fallback_to_full"] = True

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
        action_mask = (
            self._compute_action_mask(scene, participant, current_state, observation=observation)
            if self.use_action_mask
            else None
        )
        prefix_mode = self.action_mask_mode == "soft_ray" and self._ray_safety_index is not None
        if prefix_mode:
            ranked_candidates = self._ranked_primitive_candidates(observation, action_mask=action_mask)
            safe_prefix_steps = self._last_safe_prefix_steps
            if safe_prefix_steps is None:
                safe_prefix_steps = np.zeros((int(self.primitive_library.size),), dtype=np.int32)

            repeat_count = self._control_repeat_count()
            total_control_horizon = int(self.primitive_library.horizon) * repeat_count
            terminal_mode = self._terminal_context_active(
                scene,
                current_state,
                positive_count=int(np.count_nonzero(safe_prefix_steps > 0)),
            )
            selected_primitive_id = int(ranked_candidates[0]["primitive_id"]) if ranked_candidates else -1
            selected_safe_prefix_steps = (
                int(safe_prefix_steps[selected_primitive_id]) if 0 <= selected_primitive_id < safe_prefix_steps.shape[0] else 0
            )

            if self._is_precisely_parked(scene, current_state, require_stop_speed=False):
                emergency_id = self._choose_emergency_primitive_id()
                control_prefix_steps = repeat_count
                zero_actions = np.zeros((1, 2), dtype=np.float64)
                self._last_safety_stop_stats = {
                    "stop_triggered": False,
                    "stop_reason": "parked_latch",
                    "stop_replan_requested": False,
                }
                self._last_guard_stats = {
                    "guard_enabled": True,
                    "guard_attempts": int(min(len(ranked_candidates), self.max_guard_candidates)),
                    "guard_selected_failed": False,
                    "guard_fallback_used": bool(selected_primitive_id != int(emergency_id) and selected_primitive_id >= 0),
                    "guard_emergency_used": True,
                    "guard_selected_primitive_id": int(selected_primitive_id),
                    "guard_final_primitive_id": int(emergency_id),
                    "guard_selected_safe_prefix_steps": int(selected_safe_prefix_steps),
                    "guard_final_safe_prefix_steps": 0,
                    "guard_control_prefix_steps": int(control_prefix_steps),
                    "guard_prefix_truncated": True,
                    "guard_terminal_mode": bool(terminal_mode),
                    "guard_mode": "parked_latch",
                }
                selection_info = {
                    "primitive_prefix_steps": 1,
                    "control_prefix_steps": int(control_prefix_steps),
                    "prefix_truncated": True,
                    "replan_when_controls_exhausted": False,
                    "terminal_mode_active": bool(terminal_mode),
                    "parked_latch_active": True,
                    "safe_prefix_primitive_steps": 0,
                    "safe_prefix_control_steps": 0,
                    "guard_mode": "parked_latch",
                }
                return (
                    int(emergency_id),
                    zero_actions,
                    self._stationary_prefix_rollout(current_state, control_prefix_steps),
                    action_mask,
                    selection_info,
                )

            chosen: Optional[Dict[str, Any]] = None
            if terminal_mode:
                best_terminal = None
                for candidate in ranked_candidates[: int(self.max_guard_candidates)]:
                    primitive_id = int(candidate["primitive_id"])
                    safe_steps = int(safe_prefix_steps[primitive_id])
                    safe_control_steps = safe_steps * repeat_count
                    if safe_control_steps <= 0:
                        continue
                    for control_prefix_steps in range(1, safe_control_steps + 1):
                        prefix_state = self._cached_prefix_state(
                            participant,
                            current_state,
                            primitive_id,
                            control_prefix_steps,
                        )
                        metrics = self._terminal_state_metrics(scene, prefix_state)
                        terminal_key = self._terminal_candidate_key(metrics, control_prefix_steps) + (
                            -float(candidate["probability"]),
                        )
                        candidate_entry = {
                            "primitive_id": primitive_id,
                            "primitive_prefix_steps": int(math.ceil(control_prefix_steps / repeat_count)),
                            "control_prefix_steps": int(control_prefix_steps),
                            "safe_prefix_primitive_steps": int(safe_steps),
                            "safe_prefix_control_steps": int(safe_control_steps),
                            "metrics": metrics,
                            "probability": float(candidate["probability"]),
                            "key": terminal_key,
                        }
                        if best_terminal is None or candidate_entry["key"] < best_terminal["key"]:
                            best_terminal = candidate_entry
                chosen = best_terminal
            else:
                for candidate in ranked_candidates[: int(self.max_guard_candidates)]:
                    primitive_id = int(candidate["primitive_id"])
                    safe_steps = int(safe_prefix_steps[primitive_id])
                    safe_control_steps = safe_steps * repeat_count
                    if safe_control_steps <= 0:
                        continue
                    chosen = {
                        "primitive_id": primitive_id,
                        "primitive_prefix_steps": int(safe_steps),
                        "control_prefix_steps": int(safe_control_steps),
                        "safe_prefix_primitive_steps": int(safe_steps),
                        "safe_prefix_control_steps": int(safe_control_steps),
                        "metrics": self._terminal_state_metrics(
                            scene,
                            self._cached_prefix_state(
                                participant,
                                current_state,
                                primitive_id,
                                safe_control_steps,
                            ),
                        ),
                        "probability": float(candidate["probability"]),
                    }
                    break

            if chosen is None:
                emergency_id = self._choose_emergency_primitive_id()
                control_prefix_steps = repeat_count
                zero_actions = np.zeros((1, 2), dtype=np.float64)
                self._last_safety_stop_stats = {
                    "stop_triggered": True,
                    "stop_reason": "zero_safe_prefix",
                    "stop_replan_requested": bool(self.replan_on_emergency_stop),
                }
                self._last_guard_stats = {
                    "guard_enabled": True,
                    "guard_attempts": int(min(len(ranked_candidates), self.max_guard_candidates)),
                    "guard_selected_failed": True,
                    "guard_fallback_used": False,
                    "guard_emergency_used": True,
                    "guard_selected_primitive_id": int(selected_primitive_id),
                    "guard_final_primitive_id": int(emergency_id),
                    "guard_selected_safe_prefix_steps": int(selected_safe_prefix_steps),
                    "guard_final_safe_prefix_steps": 0,
                    "guard_control_prefix_steps": int(control_prefix_steps),
                    "guard_prefix_truncated": True,
                    "guard_terminal_mode": bool(terminal_mode),
                    "guard_mode": "zero_safe_prefix",
                }
                selection_info = {
                    "primitive_prefix_steps": 1,
                    "control_prefix_steps": int(control_prefix_steps),
                    "prefix_truncated": True,
                    "replan_when_controls_exhausted": False,
                    "terminal_mode_active": bool(terminal_mode),
                    "parked_latch_active": False,
                    "safe_prefix_primitive_steps": 0,
                    "safe_prefix_control_steps": 0,
                    "guard_mode": "zero_safe_prefix",
                }
                return (
                    int(emergency_id),
                    zero_actions,
                    self._stationary_prefix_rollout(current_state, control_prefix_steps),
                    action_mask,
                    selection_info,
                )

            primitive_id = int(chosen["primitive_id"])
            primitive_actions_full = np.asarray(self.primitive_library.get_actions(primitive_id), dtype=np.float64)
            primitive_prefix_steps = max(int(chosen["primitive_prefix_steps"]), 1)
            control_prefix_steps = max(int(chosen["control_prefix_steps"]), 1)
            primitive_actions = primitive_actions_full[:primitive_prefix_steps].copy()
            rollout_states = self._cached_prefix_rollout(
                participant,
                current_state,
                primitive_id,
                control_prefix_steps,
            )
            prefix_truncated = bool(control_prefix_steps < total_control_horizon)
            self._last_safety_stop_stats = {
                "stop_triggered": False,
                "stop_reason": None,
                "stop_replan_requested": False,
            }
            self._last_guard_stats = {
                "guard_enabled": True,
                "guard_attempts": int(min(len(ranked_candidates), self.max_guard_candidates)),
                "guard_selected_failed": bool(selected_safe_prefix_steps * repeat_count < total_control_horizon),
                "guard_fallback_used": bool(primitive_id != selected_primitive_id),
                "guard_emergency_used": False,
                "guard_selected_primitive_id": int(selected_primitive_id),
                "guard_final_primitive_id": int(primitive_id),
                "guard_selected_safe_prefix_steps": int(selected_safe_prefix_steps),
                "guard_final_safe_prefix_steps": int(chosen["safe_prefix_primitive_steps"]),
                "guard_control_prefix_steps": int(control_prefix_steps),
                "guard_prefix_truncated": bool(prefix_truncated),
                "guard_terminal_mode": bool(terminal_mode),
                "guard_mode": "terminal_prefix" if terminal_mode else "safe_prefix",
            }
            selection_info = {
                "primitive_prefix_steps": int(primitive_prefix_steps),
                "control_prefix_steps": int(control_prefix_steps),
                "prefix_truncated": bool(prefix_truncated),
                "replan_when_controls_exhausted": bool(prefix_truncated),
                "terminal_mode_active": bool(terminal_mode),
                "parked_latch_active": False,
                "safe_prefix_primitive_steps": int(chosen["safe_prefix_primitive_steps"]),
                "safe_prefix_control_steps": int(chosen["safe_prefix_control_steps"]),
                "guard_mode": "terminal_prefix" if terminal_mode else "safe_prefix",
            }
            return primitive_id, primitive_actions, rollout_states, action_mask, selection_info

        primitive_id, _ = self.agent.choose_action(
            observation,
            deterministic=self.deterministic,
            action_mask=action_mask,
        )
        primitive_id = int(primitive_id)
        primitive_actions = np.asarray(self.primitive_library.get_actions(primitive_id), dtype=np.float64)
        stop_stats = self._evaluate_directional_stop(scene, current_state, observation, primitive_actions, store=True)
        full_control_steps = int(primitive_actions.shape[0]) * self._control_repeat_count()
        self._last_guard_stats = {
            "guard_enabled": self.action_mask_mode == "soft_ray",
            "guard_attempts": 1,
            "guard_selected_failed": bool(stop_stats["stop_triggered"]),
            "guard_fallback_used": False,
            "guard_emergency_used": bool(stop_stats["stop_triggered"]),
            "guard_selected_primitive_id": int(primitive_id),
            "guard_final_primitive_id": int(primitive_id),
            "guard_selected_safe_prefix_steps": None,
            "guard_final_safe_prefix_steps": None,
            "guard_control_prefix_steps": int(full_control_steps),
            "guard_prefix_truncated": False,
            "guard_terminal_mode": False,
            "guard_mode": "directional_stop_guard",
        }

        if bool(stop_stats["stop_triggered"]):
            emergency_id = self._choose_emergency_primitive_id()
            emergency_actions = np.zeros((1, 2), dtype=np.float64)
            emergency_rollout = self._stationary_prefix_rollout(current_state, self._control_repeat_count())
            self._last_guard_stats["guard_final_primitive_id"] = int(emergency_id)
            selection_info = {
                "primitive_prefix_steps": 1,
                "control_prefix_steps": int(self._control_repeat_count()),
                "prefix_truncated": True,
                "replan_when_controls_exhausted": False,
                "terminal_mode_active": False,
                "parked_latch_active": False,
                "safe_prefix_primitive_steps": 0,
                "safe_prefix_control_steps": 0,
                "guard_mode": "directional_stop_guard",
            }
            return int(emergency_id), emergency_actions, emergency_rollout, action_mask, selection_info

        rollout_states = self._rollout_primitive(participant, primitive_actions, state=current_state)
        selection_info = {
            "primitive_prefix_steps": int(primitive_actions.shape[0]),
            "control_prefix_steps": int(full_control_steps),
            "prefix_truncated": False,
            "replan_when_controls_exhausted": False,
            "terminal_mode_active": False,
            "parked_latch_active": False,
            "safe_prefix_primitive_steps": None,
            "safe_prefix_control_steps": None,
            "guard_mode": "directional_stop_guard",
        }
        return primitive_id, primitive_actions, rollout_states, action_mask, selection_info

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
        primitive_id, primitive_actions, rollout_states, action_mask, selection_info = self._choose_closed_loop_primitive(
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
        if action_mask is None:
            feasible_count = None
        elif self.action_mask_mode == "soft_ray":
            feasible_count = self._last_action_mask_stats.get("soft_effective_action_count")
        else:
            feasible_count = int(np.count_nonzero(action_mask))
        metadata["action_mask_feasible_count"] = feasible_count
        metadata["action_mask_precomputed_available"] = bool(
            self._last_action_mask_stats.get("precomputed_available", False)
        )
        metadata["action_mask_precomputed_used"] = bool(
            self._last_action_mask_stats.get("precomputed_used", False)
        )
        metadata["action_mask_precomputed_candidate_count"] = self._last_action_mask_stats.get(
            "precomputed_candidate_count"
        )
        metadata["action_mask_precomputed_fallback_to_full"] = bool(
            self._last_action_mask_stats.get("precomputed_fallback_to_full", False)
        )
        metadata["action_mask_precomputed_index_kind"] = self._last_action_mask_stats.get(
            "precomputed_index_kind"
        )
        metadata["action_mask_precomputed_index_source"] = self._last_action_mask_stats.get(
            "precomputed_index_source"
        )
        metadata["action_mask_ray_safety_available"] = bool(
            self._last_action_mask_stats.get("ray_safety_available", False)
        )
        metadata["action_mask_ray_safety_used"] = bool(
            self._last_action_mask_stats.get("ray_safety_used", False)
        )
        metadata["action_mask_soft_mask_ms"] = self._last_action_mask_stats.get("soft_mask_ms")
        metadata["action_mask_soft_min"] = self._last_action_mask_stats.get("soft_mask_min")
        metadata["action_mask_soft_max"] = self._last_action_mask_stats.get("soft_mask_max")
        metadata["action_mask_soft_mean"] = self._last_action_mask_stats.get("soft_mask_mean")
        metadata["action_mask_soft_terminal_reweight_applied"] = bool(
            self._last_action_mask_stats.get("soft_terminal_reweight_applied", False)
        )
        metadata["action_mask_soft_fallback"] = self._last_action_mask_stats.get("soft_mask_fallback")
        metadata["safety_stop_distance_m"] = float(self.safety_stop_distance_m)
        metadata["safety_forward_sector_half_angle_rad"] = float(self.safety_forward_sector_half_angle)
        metadata["safety_collision_buffer_m"] = float(self.safety_collision_buffer_m)
        metadata["replan_on_emergency_stop"] = bool(self.replan_on_emergency_stop)
        metadata.update(self._last_safety_stop_stats)
        metadata.update(self._last_guard_stats)
        metadata.update(selection_info)
        metadata["replan_every_steps"] = int(self.replan_every_steps)
        control_actions = self._expand_primitive_controls(primitive_actions)
        control_prefix_steps = selection_info.get("control_prefix_steps")
        if control_prefix_steps is not None:
            control_actions = control_actions[: int(control_prefix_steps)]
        metadata["control_actions_shape"] = tuple(int(dim) for dim in control_actions.shape)
        return PPOPlanningResult(
            primitive_id=int(primitive_id),
            primitive_actions=primitive_actions,
            control_actions=control_actions,
            observation=observation,
            reference=reference,
            metadata=metadata,
        )
