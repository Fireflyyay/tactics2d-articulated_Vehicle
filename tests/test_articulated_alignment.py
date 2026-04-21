import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import pytest

from tactics2d.participant.element.wheel_loader import WheelLoader
from tactics2d.participant.trajectory import ArticulatedState, State
from tactics2d.participant.trajectory.articulated_state import wrap_angle
from tactics2d.physics import ArticulatedVehicleKinematics
from tactics2d.utils.ppo_articulated_defaults import (
    PPO_ACCEL_RANGE,
    PPO_ANGULAR_SPEED_RANGE,
    PPO_ARTICULATED_COLOR,
    PPO_FRONT_OVERHANG,
    PPO_HITCH_OFFSET,
    PPO_LENGTH,
    PPO_MAX_ARTICULATION,
    PPO_REAR_OVERHANG,
    PPO_SPEED_RANGE,
    PPO_TRAILER_LENGTH,
    PPO_WIDTH,
)


def _ppo_reference_step(state, steering, speed, interval_ms, hitch_offset, trailer_length, phi_max):
    front_x = float(state.x)
    front_y = float(state.y)
    front_heading = float(state.heading)
    rear_heading = float(state.rear_heading)
    dt = 2.5 / 1000.0
    n_step = int(round(interval_ms / 2.5))

    for _ in range(n_step):
        articulation = wrap_angle(front_heading - rear_heading)
        effective_steering = float(steering)
        if articulation >= phi_max and steering > 0:
            effective_steering = 0.0
        elif articulation <= -phi_max and steering < 0:
            effective_steering = 0.0

        denom = hitch_offset * np.cos(articulation) + trailer_length
        if abs(denom) < 1e-6:
            denom = 1e-6

        front_heading_dot = (speed * np.sin(articulation) + trailer_length * effective_steering) / denom
        rear_heading_dot = front_heading_dot - effective_steering
        x_dot = speed * np.cos(front_heading)
        y_dot = speed * np.sin(front_heading)

        front_x += x_dot * dt
        front_y += y_dot * dt
        front_heading += front_heading_dot * dt
        rear_heading += rear_heading_dot * dt

    return front_x, front_y, front_heading, rear_heading


@pytest.mark.physics
def test_articulated_state_roundtrip_from_rear_axle():
    state = ArticulatedState.from_rear_axle(
        frame=0,
        x=4.0,
        y=3.0,
        rear_heading=0.25,
        hitch_offset=1.5,
        trailer_length=1.5,
        articulation_angle=0.18,
        speed=1.2,
    )

    rear_x, rear_y, rear_heading = state.get_rear_axle_position(1.5, 1.5)
    assert rear_x == pytest.approx(4.0)
    assert rear_y == pytest.approx(3.0)
    assert rear_heading == pytest.approx(0.25)
    assert state.articulation_angle == pytest.approx(0.18)


@pytest.mark.physics
def test_articulated_kinematics_matches_ppo_reference():
    model = ArticulatedVehicleKinematics(
        l1=1.5,
        l2=1.5,
        articulation_range=(-np.deg2rad(36.0), np.deg2rad(36.0)),
        speed_range=(-2.5, 2.5),
        steering_rate_range=PPO_ANGULAR_SPEED_RANGE,
        interval=200,
        delta_t=2.5,
    )
    initial_state = ArticulatedState(
        frame=0,
        x=1.0,
        y=-2.0,
        heading=0.35,
        rear_heading=0.12,
        speed=0.8,
        steering=0.0,
    )

    next_state, applied_steering, applied_speed = model.step(
        initial_state,
        steering=0.2,
        speed=1.1,
        interval=200,
    )

    ref_x, ref_y, ref_heading, ref_rear_heading = _ppo_reference_step(
        initial_state,
        steering=applied_steering,
        speed=applied_speed,
        interval_ms=200,
        hitch_offset=1.5,
        trailer_length=1.5,
        phi_max=np.deg2rad(36.0),
    )

    assert next_state.x == pytest.approx(ref_x, abs=1e-9)
    assert next_state.y == pytest.approx(ref_y, abs=1e-9)
    assert next_state.heading == pytest.approx(ref_heading, abs=1e-9)
    assert next_state.rear_heading == pytest.approx(ref_rear_heading, abs=1e-9)


@pytest.mark.participant
def test_wheel_loader_defaults_match_ppo():
    wheel_loader = WheelLoader(id_=7, verify=True)

    assert wheel_loader.color == PPO_ARTICULATED_COLOR
    assert wheel_loader.width == pytest.approx(PPO_WIDTH)
    assert wheel_loader.height is None
    assert wheel_loader.length == pytest.approx(PPO_LENGTH)
    assert wheel_loader.hitch_offset == pytest.approx(PPO_HITCH_OFFSET)
    assert wheel_loader.trailer_length == pytest.approx(PPO_TRAILER_LENGTH)
    assert wheel_loader.front_overhang == pytest.approx(PPO_FRONT_OVERHANG)
    assert wheel_loader.rear_overhang == pytest.approx(PPO_REAR_OVERHANG)
    assert wheel_loader.max_articulation == pytest.approx(PPO_MAX_ARTICULATION)
    assert wheel_loader.speed_range == pytest.approx(PPO_SPEED_RANGE)
    assert wheel_loader.accel_range == pytest.approx(PPO_ACCEL_RANGE)
    assert wheel_loader.angular_speed_range == pytest.approx(PPO_ANGULAR_SPEED_RANGE)
    assert wheel_loader.physics_model.steering_rate_range == pytest.approx(PPO_ANGULAR_SPEED_RANGE)


@pytest.mark.participant
def test_wheel_loader_coerces_legacy_rear_axle_state():
    wheel_loader = WheelLoader(
        id_=0,
        width=2.0,
        hitch_offset=1.5,
        trailer_length=1.5,
        front_overhang=1.0,
        rear_overhang=1.0,
        verify=False,
    )
    legacy_state = State(frame=0, x=5.0, y=6.0, heading=0.4, speed=1.0)
    wheel_loader.add_state(legacy_state, articulation_angle=0.2)

    current_state = wheel_loader.current_state
    assert isinstance(current_state, ArticulatedState)
    rear_x, rear_y, rear_heading = wheel_loader.get_rear_axle_position()
    assert rear_x == pytest.approx(5.0)
    assert rear_y == pytest.approx(6.0)
    assert rear_heading == pytest.approx(0.4)
    assert current_state.articulation_angle == pytest.approx(0.2)


@pytest.mark.participant
def test_wheel_loader_pose_matches_ppo_boxes():
    wheel_loader = WheelLoader(
        id_=1,
        width=2.0,
        hitch_offset=1.5,
        trailer_length=1.5,
        front_overhang=1.0,
        rear_overhang=1.0,
        verify=False,
    )
    state = ArticulatedState(frame=0, x=0.0, y=0.0, heading=0.0, rear_heading=0.0, speed=0.0)
    wheel_loader.add_state(state)

    rear_poly, front_poly = wheel_loader.get_pose()
    assert front_poly.bounds == pytest.approx((-1.5, -1.0, 1.0, 1.0))
    assert rear_poly.bounds == pytest.approx((-4.0, -1.0, -1.5, 1.0))