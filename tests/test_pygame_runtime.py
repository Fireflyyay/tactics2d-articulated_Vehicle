import os
import sys

sys.path.append(".")
sys.path.append("..")

import pytest

from tactics2d.map.element import Map
from tactics2d.map.generator import ParkingLotGenerator, RacingTrackGenerator
from tactics2d.map.generator.generate_wheel_loader_scenario import WheelLoaderScenarioGenerator
from tactics2d.renderer import (
    PygameSceneRenderer,
    SimulationRunner,
    adapt_generated_scene,
    create_default_participant,
)


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


@pytest.mark.render
def test_parking_scene_adapter_and_headless_step():
    generator = ParkingLotGenerator(type_proportion=1.0)
    map_ = Map(name="parking_scene", scenario_type="parking")
    generate_result = generator.generate(map_)

    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)
    participant = create_default_participant(scene)
    renderer = PygameSceneRenderer(boundary=map_.boundary, window_size=(640, 640), headless=True)
    try:
        runner = SimulationRunner(scene=scene, participant=participant, renderer=renderer, dt_ms=100, max_steps=5)
        active = runner.step_once()
        runner.render()
    finally:
        renderer.close()

    assert scene.participant_kind == "vehicle"
    assert scene.reference_path.length > 0.0
    assert participant.current_state.frame == 100
    assert active in {True, False}


@pytest.mark.render
def test_wheel_loader_navigation_adapter_and_headless_step():
    generator = WheelLoaderScenarioGenerator(backend="ppo", scene_type="navigation")
    map_ = Map(name="wheel_loader_scene", scenario_type="navigation")
    generate_result = generator.generate(map_, seed=4)

    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)
    participant = create_default_participant(scene)
    renderer = PygameSceneRenderer(boundary=map_.boundary, window_size=(640, 640), headless=True)
    try:
        runner = SimulationRunner(scene=scene, participant=participant, renderer=renderer, dt_ms=200, max_steps=5)
        runner.step_once()
        runner.render()
    finally:
        renderer.close()

    assert scene.participant_kind == "wheel_loader"
    assert scene.metadata["reference_path_source"] == "articulated_reference_trajectory"
    assert participant.current_state.frame == 200


@pytest.mark.render
@pytest.mark.skipif(RacingTrackGenerator is None, reason="cpp_geometry extension is unavailable")
def test_racing_scene_adapter_uses_center_line():
    generator = RacingTrackGenerator()
    map_ = Map(name="racing_scene", scenario_type="racing")
    generate_result = generator.generate(map_)
    scene = adapt_generated_scene(map_, generator=generator, generate_result=generate_result)

    assert scene.participant_kind == "vehicle"
    assert scene.metadata["reference_path_source"] == "roadline:center_line"
    assert scene.reference_path.length > 0.0