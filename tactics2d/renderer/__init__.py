##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize renderer utilities.
# @Author: Tactics2D Team
# @Version: 0.2.0

from tactics2d.sensor import MatplotlibRenderer

try:
	from .pygame_runtime import (
		PygameSceneRenderer,
		SceneDescription,
		SimulationRunner,
		adapt_generated_scene,
		create_default_participant,
	)
	from .wheel_loader_stress import (
		DEFAULT_LEVELS,
		discover_default_checkpoint,
		discover_default_ppo_root,
		dump_stress_report,
		run_wheel_loader_episode,
		run_wheel_loader_stress_suite,
	)
except ModuleNotFoundError:
	PygameSceneRenderer = None
	SceneDescription = None
	SimulationRunner = None
	adapt_generated_scene = None
	create_default_participant = None
	DEFAULT_LEVELS = None
	discover_default_checkpoint = None
	discover_default_ppo_root = None
	dump_stress_report = None
	run_wheel_loader_episode = None
	run_wheel_loader_stress_suite = None

__all__ = ["MatplotlibRenderer"]
if SceneDescription is not None:
	__all__.extend(
		[
			"SceneDescription",
			"adapt_generated_scene",
			"create_default_participant",
			"PygameSceneRenderer",
			"SimulationRunner",
			"DEFAULT_LEVELS",
			"discover_default_checkpoint",
			"discover_default_ppo_root",
			"dump_stress_report",
			"run_wheel_loader_episode",
			"run_wheel_loader_stress_suite",
		]
	)
