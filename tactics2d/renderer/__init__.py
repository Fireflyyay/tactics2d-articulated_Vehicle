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
except ModuleNotFoundError:
	PygameSceneRenderer = None
	SceneDescription = None
	SimulationRunner = None
	adapt_generated_scene = None
	create_default_participant = None

__all__ = ["MatplotlibRenderer"]
if SceneDescription is not None:
	__all__.extend(
		[
			"SceneDescription",
			"adapt_generated_scene",
			"create_default_participant",
			"PygameSceneRenderer",
			"SimulationRunner",
		]
	)
