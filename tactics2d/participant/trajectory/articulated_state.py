from typing import Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Point

from .state import State


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class ArticulatedState(State):
    """State container aligned with the PPO articulated vehicle model.

    The reference pose is the front axle center. The rear body heading is stored
    separately so the articulation angle is given by ``heading - rear_heading``.
    """

    __annotations__ = {
        **State.__annotations__,
        "rear_heading": float,
        "steering": float,
    }

    def __init__(
        self,
        frame: int,
        x: float = 0,
        y: float = 0,
        heading: float = 0,
        vx: float = None,
        vy: float = None,
        speed: float = None,
        ax: float = None,
        ay: float = None,
        accel: float = None,
        rear_heading: float = None,
        steering: float = 0.0,
    ):
        super().__init__(
            frame=frame,
            x=x,
            y=y,
            heading=heading,
            vx=vx,
            vy=vy,
            speed=speed,
            ax=ax,
            ay=ay,
            accel=accel,
        )
        setattr(self, "rear_heading", heading if rear_heading is None else rear_heading)
        setattr(self, "steering", steering)

    @classmethod
    def from_rear_axle(
        cls,
        frame: int,
        x: float,
        y: float,
        rear_heading: float,
        hitch_offset: float,
        trailer_length: float,
        articulation_angle: float = 0.0,
        speed: float = None,
        accel: float = None,
        steering: float = 0.0,
        vx: float = None,
        vy: float = None,
        ax: float = None,
        ay: float = None,
    ) -> "ArticulatedState":
        heading = wrap_angle(rear_heading + articulation_angle)
        hinge_x = x + trailer_length * np.cos(rear_heading)
        hinge_y = y + trailer_length * np.sin(rear_heading)
        front_x = hinge_x + hitch_offset * np.cos(heading)
        front_y = hinge_y + hitch_offset * np.sin(heading)
        state = cls(
            frame=frame,
            x=front_x,
            y=front_y,
            heading=heading,
            vx=vx,
            vy=vy,
            speed=speed,
            ax=ax,
            ay=ay,
            accel=accel,
            rear_heading=rear_heading,
            steering=steering,
        )
        state.update_trailer_loc(hitch_offset, trailer_length)
        return state

    @property
    def loc(self) -> Point:
        return Point(self.x, self.y)

    @property
    def articulation_angle(self) -> float:
        return wrap_angle(self.heading - self.rear_heading)

    def get_pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.heading)

    def get_hinge_position(self, hitch_offset: float) -> Tuple[float, float]:
        hinge_x = self.x - hitch_offset * np.cos(self.heading)
        hinge_y = self.y - hitch_offset * np.sin(self.heading)
        return hinge_x, hinge_y

    def get_rear_axle_position(
        self, hitch_offset: float, trailer_length: float
    ) -> Tuple[float, float, float]:
        hinge_x, hinge_y = self.get_hinge_position(hitch_offset)
        rear_x = hinge_x - trailer_length * np.cos(self.rear_heading)
        rear_y = hinge_y - trailer_length * np.sin(self.rear_heading)
        return rear_x, rear_y, self.rear_heading

    def update_trailer_loc(self, hitch_offset: float, trailer_length: float) -> Point:
        rear_x, rear_y, _ = self.get_rear_axle_position(hitch_offset, trailer_length)
        trailer_loc = Point(rear_x, rear_y)
        object.__setattr__(self, "trailer_loc", trailer_loc)
        return trailer_loc

    def create_boxes(
        self,
        front_box: LinearRing,
        rear_box: LinearRing,
        hitch_offset: float,
        trailer_length: float,
    ) -> Tuple[LinearRing, LinearRing]:
        rear_x, rear_y, rear_heading = self.get_rear_axle_position(hitch_offset, trailer_length)
        front_transform = [
            np.cos(self.heading),
            -np.sin(self.heading),
            np.sin(self.heading),
            np.cos(self.heading),
            self.x,
            self.y,
        ]
        rear_transform = [
            np.cos(rear_heading),
            -np.sin(rear_heading),
            np.sin(rear_heading),
            np.cos(rear_heading),
            rear_x,
            rear_y,
        ]
        return (
            affine_transform(front_box, front_transform),
            affine_transform(rear_box, rear_transform),
        )

    def to_rear_axle_state(self, hitch_offset: float, trailer_length: float) -> State:
        rear_x, rear_y, rear_heading = self.get_rear_axle_position(hitch_offset, trailer_length)
        return State(
            frame=self.frame,
            x=rear_x,
            y=rear_y,
            heading=rear_heading,
            speed=self.speed,
            accel=self.accel,
        )