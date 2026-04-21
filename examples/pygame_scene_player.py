import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactics2d.map.element import Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator
from tactics2d.renderer import (
    PygameSceneRenderer,
    SimulationRunner,
    adapt_generated_scene,
    create_default_participant,
)


def _generate_scene(generator, map_, seed=None):
    if seed is None:
        return generator.generate(map_)
    try:
        return generator.generate(map_, seed=seed)
    except TypeError:
        return generator.generate(map_)


def _build_generator(args):
    if args.scene == "parking":
        return ParkingLotGenerator(type_proportion=args.type_proportion)
    if args.scene == "racing":
        if RacingTrackGenerator is None:
            raise RuntimeError("RacingTrackGenerator is unavailable because cpp_geometry is missing.")
        return RacingTrackGenerator()
    return WheelLoaderScenarioGenerator(
        backend=args.backend,
        scene_type=args.scene_type,
        map_level=args.map_level,
        ppo_root=args.ppo_root,
        width=args.legacy_width,
        height=args.legacy_height,
        num_obstacles=args.legacy_obstacles,
        obstacle_radius_range=(args.legacy_obstacle_min_radius, args.legacy_obstacle_max_radius),
        min_obstacle_spacing=args.legacy_min_spacing,
    )


def _discover_default_checkpoint() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "BestCheckPoint" / "PPO_best.pt"
        if candidate.exists():
            return str(candidate)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Play generator scenes with a pygame runtime.")
    parser.add_argument("--scene", choices=["wheel_loader", "parking", "racing"], default="wheel_loader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dt-ms", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=960)
    parser.add_argument("--type-proportion", type=float, default=0.5)
    parser.add_argument("--backend", choices=["ppo", "legacy"], default="ppo")
    parser.add_argument("--scene-type", choices=["navigation", "bay", "parallel"], default="navigation")
    parser.add_argument("--map-level", default="Normal")
    parser.add_argument("--ppo-root", default=None)
    parser.add_argument("--legacy-width", type=float, default=50.0)
    parser.add_argument("--legacy-height", type=float, default=50.0)
    parser.add_argument("--legacy-obstacles", type=int, default=8)
    parser.add_argument("--legacy-obstacle-min-radius", type=float, default=1.0)
    parser.add_argument("--legacy-obstacle-max-radius", type=float, default=3.0)
    parser.add_argument("--legacy-min-spacing", type=float, default=5.0)
    parser.add_argument("--wheel-loader-planner", choices=["default", "ppo"], default="default")
    parser.add_argument("--ppo-checkpoint", default=_discover_default_checkpoint())
    parser.add_argument("--ppo-stochastic", action="store_true")
    args = parser.parse_args()

    generator = _build_generator(args)
    map_ = Map(name=f"{args.scene}_scene", scenario_type=args.scene)
    generate_result = _generate_scene(generator, map_, seed=args.seed)

    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)
    participant = create_default_participant(scene)
    renderer = PygameSceneRenderer(
        boundary=map_.boundary,
        window_size=(args.window_width, args.window_height),
        fps=args.fps,
        title=scene.title,
    )

    try:
        wheel_loader_planner = None
        if args.scene == "wheel_loader" and args.wheel_loader_planner == "ppo":
            if not args.ppo_checkpoint:
                raise RuntimeError("PPO planner requires --ppo-checkpoint.")
            wheel_loader_planner = {
                "mode": "ppo",
                "checkpoint_path": args.ppo_checkpoint,
                "ppo_root": args.ppo_root,
                "deterministic": not args.ppo_stochastic,
            }

        runner = SimulationRunner(
            scene=scene,
            participant=participant,
            renderer=renderer,
            dt_ms=args.dt_ms,
            max_steps=args.max_steps,
            wheel_loader_planner=wheel_loader_planner,
        )
        runner.run()
    finally:
        renderer.close()


if __name__ == "__main__":
    main()