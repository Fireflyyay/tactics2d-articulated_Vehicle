import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tactics2d.map.element import Map
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator

from .pygame_runtime import (
    PygameSceneRenderer,
    SimulationRunner,
    adapt_generated_scene,
    create_default_participant,
)


DEFAULT_LEVELS = ("Normal", "Complex", "Extrem")


def discover_default_checkpoint() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "BestCheckPoint" / "PPO_best.pt"
        if candidate.exists():
            return str(candidate)
    return ""


def discover_default_ppo_root() -> Optional[str]:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "PPO_articulated_vehicle"
        if candidate.exists():
            return str(candidate)
    return None


@dataclass
class WheelLoaderEpisodeReport:
    level: str
    episode_index: int
    seed: int
    success: bool
    final_status: str
    step_count: int
    total_wall_time_ms: float
    planning_calls: int
    planning_runtime_ms: List[float]
    last_plan_runtime_ms: Optional[float]


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * float(q)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _runtime_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    normalized = [float(value) for value in values]
    return {
        "mean_ms": float(statistics.fmean(normalized)),
        "p50_ms": _percentile(normalized, 0.50),
        "p95_ms": _percentile(normalized, 0.95),
        "max_ms": float(max(normalized)),
    }


def _generate_scene(level: str, seed: int, scene_type: str, ppo_root: Optional[str]):
    generator = WheelLoaderScenarioGenerator(
        backend="ppo",
        scene_type=scene_type,
        map_level=level,
        ppo_root=ppo_root,
    )
    map_ = Map(name=f"wheel_loader_{level.lower()}_{seed}", scenario_type="wheel_loader")
    generate_result = generator.generate(map_, seed=seed)
    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)
    participant = create_default_participant(scene)
    return map_, scene, participant


def run_wheel_loader_episode(
    checkpoint_path: str,
    level: str,
    seed: int,
    ppo_root: Optional[str] = None,
    mode: str = "background",
    scene_type: str = "navigation",
    dt_ms: int = 100,
    max_steps: int = 1500,
    fps: int = 30,
    window_size: Sequence[int] = (960, 960),
    deterministic: bool = True,
    replan_every_steps: int = 1,
) -> WheelLoaderEpisodeReport:
    map_, scene, participant = _generate_scene(level=level, seed=seed, scene_type=scene_type, ppo_root=ppo_root)
    headless = str(mode).strip().lower() != "visual"
    renderer = PygameSceneRenderer(
        boundary=map_.boundary,
        window_size=(int(window_size[0]), int(window_size[1])),
        fps=fps,
        title=scene.title,
        headless=headless,
    )

    started_at = time.perf_counter()
    try:
        runner = SimulationRunner(
            scene=scene,
            participant=participant,
            renderer=renderer,
            dt_ms=dt_ms,
            max_steps=max_steps,
            wheel_loader_planner={
                "mode": "ppo",
                "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
                "ppo_root": ppo_root,
                "replan_every_steps": replan_every_steps,
                "deterministic": deterministic,
            },
        )
        runner.run()
    finally:
        renderer.close()

    total_wall_time_ms = float((time.perf_counter() - started_at) * 1000.0)
    planning_runtime_ms = [float(item["runtime_ms"]) for item in runner.planning_history]
    last_plan_runtime_ms = planning_runtime_ms[-1] if planning_runtime_ms else None
    return WheelLoaderEpisodeReport(
        level=str(level),
        episode_index=-1,
        seed=int(seed),
        success=runner.last_status == "goal_reached",
        final_status=str(runner.last_status or "unknown"),
        step_count=int(runner.current_step),
        total_wall_time_ms=total_wall_time_ms,
        planning_calls=int(len(planning_runtime_ms)),
        planning_runtime_ms=planning_runtime_ms,
        last_plan_runtime_ms=last_plan_runtime_ms,
    )


def run_wheel_loader_stress_suite(
    checkpoint_path: str,
    levels: Optional[Iterable[str]] = None,
    episodes_per_level: int = 10,
    seed: int = 42,
    ppo_root: Optional[str] = None,
    mode: str = "background",
    scene_type: str = "navigation",
    dt_ms: int = 100,
    max_steps: int = 1500,
    fps: int = 30,
    window_size: Sequence[int] = (960, 960),
    deterministic: bool = True,
    replan_every_steps: int = 1,
) -> Dict[str, object]:
    normalized_levels = [str(level) for level in (levels or DEFAULT_LEVELS)]
    episode_reports: List[WheelLoaderEpisodeReport] = []
    report_by_level: Dict[str, Dict[str, object]] = {}

    for level_index, level in enumerate(normalized_levels):
        level_reports: List[WheelLoaderEpisodeReport] = []
        for episode_index in range(int(episodes_per_level)):
            episode_seed = int(seed) + level_index * 100000 + episode_index
            report = run_wheel_loader_episode(
                checkpoint_path=checkpoint_path,
                level=level,
                seed=episode_seed,
                ppo_root=ppo_root,
                mode=mode,
                scene_type=scene_type,
                dt_ms=dt_ms,
                max_steps=max_steps,
                fps=fps,
                window_size=window_size,
                deterministic=deterministic,
                replan_every_steps=replan_every_steps,
            )
            report.episode_index = int(episode_index)
            level_reports.append(report)
            episode_reports.append(report)

        success_count = sum(1 for report in level_reports if report.success)
        planning_samples = [
            runtime
            for report in level_reports
            for runtime in report.planning_runtime_ms
        ]
        report_by_level[level] = {
            "episodes": int(len(level_reports)),
            "successes": int(success_count),
            "success_rate": (float(success_count) / float(len(level_reports))) if level_reports else None,
            "avg_steps": float(statistics.fmean(report.step_count for report in level_reports)) if level_reports else None,
            "avg_planning_calls": float(statistics.fmean(report.planning_calls for report in level_reports)) if level_reports else None,
            "avg_episode_wall_time_ms": (
                float(statistics.fmean(report.total_wall_time_ms for report in level_reports)) if level_reports else None
            ),
            "inference_time_ms": _runtime_stats(planning_samples),
            "statuses": {
                status: sum(1 for report in level_reports if report.final_status == status)
                for status in sorted({report.final_status for report in level_reports})
            },
        }

    aggregate_planning_samples = [
        runtime
        for report in episode_reports
        for runtime in report.planning_runtime_ms
    ]
    total_successes = sum(1 for report in episode_reports if report.success)
    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "ppo_root": ppo_root,
        "mode": str(mode),
        "scene_type": str(scene_type),
        "deterministic": bool(deterministic),
        "replan_every_steps": int(replan_every_steps),
        "episodes_per_level": int(episodes_per_level),
        "levels": normalized_levels,
        "summary": {
            "episodes": int(len(episode_reports)),
            "successes": int(total_successes),
            "success_rate": (float(total_successes) / float(len(episode_reports))) if episode_reports else None,
            "avg_steps": float(statistics.fmean(report.step_count for report in episode_reports)) if episode_reports else None,
            "avg_episode_wall_time_ms": (
                float(statistics.fmean(report.total_wall_time_ms for report in episode_reports)) if episode_reports else None
            ),
            "inference_time_ms": _runtime_stats(aggregate_planning_samples),
        },
        "per_level": report_by_level,
        "episodes": [asdict(report) for report in episode_reports],
    }


def dump_stress_report(report: Dict[str, object], output_path: str) -> str:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(destination)