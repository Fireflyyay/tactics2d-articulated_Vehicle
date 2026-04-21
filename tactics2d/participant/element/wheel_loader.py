import logging
from typing import Any, Tuple

import numpy as np
from shapely.geometry import LinearRing, Polygon

from tactics2d.participant.trajectory import ArticulatedState, State, Trajectory
from tactics2d.physics import ArticulatedVehicleKinematics
from tactics2d.utils.ppo_articulated_defaults import (
    PPO_ACCEL_RANGE,
    PPO_ANGULAR_SPEED_RANGE,
    PPO_ARTICULATED_COLOR,
    PPO_DEFAULT_DELTA_T_MS,
    PPO_DEFAULT_INTERVAL_MS,
    PPO_FRONT_OVERHANG,
    PPO_HITCH_OFFSET,
    PPO_MAX_ARTICULATION,
    PPO_REAR_OVERHANG,
    PPO_SPEED_RANGE,
    PPO_TRAILER_LENGTH,
    PPO_WHEEL_BASE,
    PPO_WIDTH,
    build_front_vehicle_box,
    build_rear_vehicle_box,
)

from .participant_base import ParticipantBase


class WheelLoader(ParticipantBase):
    r"""Active-articulation vehicle participant.

    Internally the vehicle state is aligned with PPO_articulated_vehicle:
    the stored pose is the front axle center, while the rear body heading is
    carried in ``ArticulatedState.rear_heading``.

    For backward compatibility, plain ``State`` inputs are interpreted as rear
    axle states and converted on insertion.
    """

    __annotations__ = {
        "type_": str,
        "length": float,
        "width": float,
        "height": float,
        "axle_distance": float,
        "bucket_length": float,
        "rear_section_length": float,
        "front_section_length": float,
        "hitch_offset": float,
        "trailer_length": float,
        "front_overhang": float,
        "rear_overhang": float,
        "max_articulation": float,
        "max_angular_speed": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = PPO_ARTICULATED_COLOR

    def __init__(
        self, id_: Any, type_: str = "wheel_loader", trajectory: Trajectory = None, **kwargs
    ):
        super().__init__(id_, type_, trajectory, **kwargs)

        if self.width is None:
            self.width = PPO_WIDTH

        requested_axle_distance = kwargs.get("axle_distance", PPO_WHEEL_BASE)
        requested_hitch_offset = kwargs.get("hitch_offset", PPO_HITCH_OFFSET)
        requested_trailer_length = kwargs.get(
            "trailer_length", requested_axle_distance - requested_hitch_offset
        )

        self.hitch_offset = float(requested_hitch_offset)
        self.trailer_length = float(requested_trailer_length)
        self.axle_distance = float(self.hitch_offset + self.trailer_length)

        self.max_articulation = kwargs.get("max_articulation", PPO_MAX_ARTICULATION)
        self.max_angular_speed = kwargs.get(
            "max_angular_speed", abs(PPO_ANGULAR_SPEED_RANGE[1])
        )
        self.max_speed = kwargs.get("max_speed", PPO_SPEED_RANGE[1])
        self.max_accel = kwargs.get("max_accel", PPO_ACCEL_RANGE[1])
        self.max_decel = kwargs.get("max_decel", abs(PPO_ACCEL_RANGE[0]))

        remaining_body_length = None
        if self.length is not None:
            remaining_body_length = max(float(self.length) - self.axle_distance, 0.0)

        front_section_length = kwargs.get("front_section_length")
        rear_section_length = kwargs.get("rear_section_length")
        explicit_front_overhang = kwargs.get("front_overhang")
        explicit_rear_overhang = kwargs.get("rear_overhang")

        if explicit_front_overhang is not None:
            self.front_overhang = float(explicit_front_overhang)
        elif front_section_length is not None:
            self.front_overhang = max(float(front_section_length) - self.hitch_offset, 0.0)
        elif remaining_body_length is not None:
            self.front_overhang = remaining_body_length / 2.0
        else:
            self.front_overhang = PPO_FRONT_OVERHANG

        if explicit_rear_overhang is not None:
            self.rear_overhang = float(explicit_rear_overhang)
        elif rear_section_length is not None:
            self.rear_overhang = max(float(rear_section_length) - self.trailer_length, 0.0)
        elif remaining_body_length is not None:
            self.rear_overhang = max(remaining_body_length - self.front_overhang, 0.0)
        else:
            self.rear_overhang = PPO_REAR_OVERHANG

        self.front_section_length = float(self.hitch_offset + self.front_overhang)
        self.rear_section_length = float(self.trailer_length + self.rear_overhang)
        self.bucket_length = kwargs.get("bucket_length", self.front_overhang)
        self.length = float(self.front_overhang + self.axle_distance + self.rear_overhang)

        self.articulation_range = (-self.max_articulation, self.max_articulation)
        self.angular_speed_range = (-self.max_angular_speed, self.max_angular_speed)
        self.speed_range = (-self.max_speed, self.max_speed)
        self.accel_range = (-self.max_decel, self.max_accel)

        if self.verify:
            if "physics_model" in kwargs and kwargs["physics_model"] is not None:
                self.physics_model = kwargs["physics_model"]
            else:
                self._auto_construct_physics_model()

        self._front_bbox = build_front_vehicle_box(
            width=self.width,
            hitch_offset=self.hitch_offset,
            front_overhang=self.front_overhang,
        )
        self._rear_bbox = build_rear_vehicle_box(
            width=self.width,
            trailer_length=self.trailer_length,
            rear_overhang=self.rear_overhang,
        )

        self._current_articulation = 0.0

    @property
    def geometry(self) -> Tuple[LinearRing, LinearRing]:
        return self._rear_bbox, self._front_bbox

    @property
    def current_articulation(self) -> float:
        state = self.current_state
        if isinstance(state, ArticulatedState):
            return state.articulation_angle
        return self._current_articulation

    @current_articulation.setter
    def current_articulation(self, value: float):
        self._current_articulation = float(value)
        state = self.current_state
        if isinstance(state, ArticulatedState):
            state.rear_heading = state.heading - self._current_articulation
            state.update_trailer_loc(self.hitch_offset, self.trailer_length)

    def _auto_construct_physics_model(self):
        self.physics_model = ArticulatedVehicleKinematics(
            L=self.axle_distance,
            l1=self.hitch_offset,
            l2=self.trailer_length,
            articulation_range=self.articulation_range,
            speed_range=self.speed_range,
            steering_rate_range=self.angular_speed_range,
            accel_range=self.accel_range,
            interval=PPO_DEFAULT_INTERVAL_MS,
            delta_t=PPO_DEFAULT_DELTA_T_MS,
        )

    def build_state_from_rear_axle(
        self,
        frame: int,
        x: float,
        y: float,
        heading: float,
        speed: float = None,
        accel: float = None,
        articulation_angle: float = None,
        steering: float = 0.0,
        vx: float = None,
        vy: float = None,
        ax: float = None,
        ay: float = None,
    ) -> ArticulatedState:
        articulation = self.current_articulation if articulation_angle is None else articulation_angle
        return ArticulatedState.from_rear_axle(
            frame=frame,
            x=x,
            y=y,
            rear_heading=heading,
            hitch_offset=self.hitch_offset,
            trailer_length=self.trailer_length,
            articulation_angle=articulation,
            speed=speed,
            accel=accel,
            steering=steering,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
        )

    def _coerce_state(
        self, state: State, articulation_angle: float = None
    ) -> ArticulatedState:
        if isinstance(state, ArticulatedState):
            rear_heading = state.rear_heading
            if articulation_angle is not None:
                rear_heading = state.heading - float(articulation_angle)
            coerced = ArticulatedState(
                frame=state.frame,
                x=state.x,
                y=state.y,
                heading=state.heading,
                vx=state.vx,
                vy=state.vy,
                speed=state.speed,
                ax=state.ax,
                ay=state.ay,
                accel=state.accel,
                rear_heading=rear_heading,
                steering=state.steering,
            )
            coerced.update_trailer_loc(self.hitch_offset, self.trailer_length)
            return coerced

        return self.build_state_from_rear_axle(
            frame=state.frame,
            x=state.x,
            y=state.y,
            heading=state.heading,
            speed=state.speed,
            accel=state.accel,
            articulation_angle=articulation_angle,
            steering=getattr(state, "steering", 0.0),
            vx=state.vx,
            vy=state.vy,
            ax=state.ax,
            ay=state.ay,
        )

    def bind_trajectory(self, trajectory: Trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory must be an instance of Trajectory.")

        normalized_trajectory = Trajectory(self.id_)
        for frame in trajectory.frames:
            normalized_trajectory.add_state(self._coerce_state(trajectory.get_state(frame)))

        if self.verify:
            if not self._verify_trajectory(normalized_trajectory):
                self.trajectory = Trajectory(self.id_)
                logging.warning(
                    f"The trajectory is invalid. Wheel loader {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = normalized_trajectory
        else:
            self.trajectory = normalized_trajectory
            logging.debug(
                f"Wheel loader {self.id_} is bound to a trajectory without verification."
            )

    def add_state(self, state: State, articulation_angle: float = None):
        normalized_state = self._coerce_state(state, articulation_angle=articulation_angle)
        self._current_articulation = normalized_state.articulation_angle
        if not self.verify or self.physics_model is None:
            self.trajectory.add_state(normalized_state)
        elif self.physics_model.verify_state(normalized_state, self.trajectory.last_state):
            self.trajectory.add_state(normalized_state)
        else:
            raise RuntimeError(
                "Invalid state checked by the physics model %s."
                % (self.physics_model.__class__.__name__)
            )

    def get_pose(self, frame: int = None, articulation_angle: float = None) -> Tuple[Polygon, Polygon]:
        state = self._coerce_state(self.trajectory.get_state(frame), articulation_angle=articulation_angle)
        front_pose, rear_pose = state.create_boxes(
            self._front_bbox,
            self._rear_bbox,
            self.hitch_offset,
            self.trailer_length,
        )
        return Polygon(rear_pose), Polygon(front_pose)

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        return None

    def get_rear_axle_position(
        self, frame: int = None, articulation_angle: float = None
    ) -> Tuple[float, float, float]:
        state = self._coerce_state(self.trajectory.get_state(frame), articulation_angle=articulation_angle)
        return state.get_rear_axle_position(self.hitch_offset, self.trailer_length)

    def get_rear_axle_state(self, frame: int = None, articulation_angle: float = None) -> State:
        state = self._coerce_state(self.trajectory.get_state(frame), articulation_angle=articulation_angle)
        return state.to_rear_axle_state(self.hitch_offset, self.trailer_length)

    def get_front_axle_position(
        self, frame: int = None, articulation_angle: float = None
    ) -> Tuple[float, float, float]:
        state = self._coerce_state(self.trajectory.get_state(frame), articulation_angle=articulation_angle)
        return state.x, state.y, state.heading