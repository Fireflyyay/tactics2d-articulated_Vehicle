import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactics2d.renderer.wheel_loader_stress import (
    DEFAULT_LEVELS,
    discover_default_checkpoint,
    discover_default_ppo_root,
    dump_stress_report,
    run_wheel_loader_stress_suite,
)


def main():
    parser = argparse.ArgumentParser(description="Stress test PPO wheel loader checkpoints on random scenes.")
    parser.add_argument("--checkpoint", default=discover_default_checkpoint())
    parser.add_argument("--ppo-root", default=discover_default_ppo_root())
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--episodes-per-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["visual", "background"], default="background")
    parser.add_argument("--scene-type", choices=["navigation", "bay", "parallel"], default="navigation")
    parser.add_argument("--dt-ms", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=960)
    parser.add_argument("--replan-every-steps", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not args.checkpoint:
        raise RuntimeError("A checkpoint path is required. Use --checkpoint /home/cyberbus/Public/BestCheckPoint/PPO_best.pt")

    report = run_wheel_loader_stress_suite(
        checkpoint_path=args.checkpoint,
        levels=args.levels,
        episodes_per_level=args.episodes_per_level,
        seed=args.seed,
        ppo_root=args.ppo_root,
        mode=args.mode,
        scene_type=args.scene_type,
        dt_ms=args.dt_ms,
        max_steps=args.max_steps,
        fps=args.fps,
        window_size=(args.window_width, args.window_height),
        deterministic=not args.stochastic,
        replan_every_steps=args.replan_every_steps,
    )

    if args.output:
        dump_stress_report(report, args.output)

    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["per_level"], indent=2))


if __name__ == "__main__":
    main()