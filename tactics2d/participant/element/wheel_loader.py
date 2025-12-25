##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: wheel_loader.py
# @Description: This file defines a class for an articulated wheel loader vehicle.
# @Author: Tactics2D Team
# @Version: 1.0.0

import logging
from typing import Any, Tuple

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Polygon

from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.physics import ArticulatedVehicleKinematics

from .participant_base import ParticipantBase


class WheelLoader(ParticipantBase):
    r"""This class defines an articulated wheel loader vehicle.

    The wheel loader consists of two sections (front and rear) connected by a hinge.
    The vehicle's state is represented by the rear axle center position and heading.
    The articulation angle controls the relative angle between front and rear sections.

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle. Defaults to "wheel_loader".
        trajectory (Trajectory): The trajectory of the vehicle.
        color (tuple): The color of the vehicle. Defaults to "#ff6b6b" (red).
        length (float): The total length of the vehicle (rear section + front section). The unit is meter.
        width (float): The width of the vehicle. The unit is meter.
        height (float): The height of the vehicle. The unit is meter.
        axle_distance (float): The distance between front and rear axles. The unit is meter. Defaults to 3.0.
        bucket_length (float): The length of the bucket/loader. The unit is meter. Defaults to 2.0.
        rear_section_length (float): The length of the rear section. The unit is meter.
        front_section_length (float): The length of the front section (including bucket). The unit is meter.
        max_articulation (float): The maximum articulation angle. The unit is radian. Defaults to π/3.
        max_speed (float): The maximum speed. The unit is meter per second. Defaults to 5.0.
        max_accel (float): The maximum acceleration. The unit is meter per second squared. Defaults to 2.0.
        max_decel (float): The maximum deceleration. The unit is meter per second squared. Defaults to 5.0.
        articulation_range (Tuple[float, float]): The range of articulation angle. The unit is radian.
        speed_range (Tuple[float, float]): The range of speed. The unit is meter per second.
        accel_range (Tuple[float, float]): The range of acceleration. The unit is meter per second squared.
        verify (bool): Whether to verify the trajectory. Defaults to False.
        physics_model (ArticulatedVehicleKinematics): The physics model of the vehicle.
        geometry (LinearRing): The geometry shape of the vehicle. This attribute is **read-only**.
        current_state (State): The current state of the vehicle. This attribute is **read-only**.
        current_articulation (float): The current articulation angle. The unit is radian.
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
        "max_articulation": float,
        "max_speed": float,
        "max_accel": float,
        "max_decel": float,
        "verify": bool,
    }
    _default_color = "#ff6b6b"  # red

    def __init__(
        self, id_: Any, type_: str = "wheel_loader", trajectory: Trajectory = None, **kwargs
    ):
        r"""Initialize the wheel loader.

        Args:
            id_ (int): The unique identifier of the vehicle.
            type_ (str, optional): The type of the vehicle. Defaults to "wheel_loader".
            trajectory (Trajectory, optional): The trajectory of the vehicle.

        Keyword Args:
            length (float, optional): The total length of the vehicle. Defaults to None.
            width (float, optional): The width of the vehicle. The unit is meter. Defaults to None.
            height (float, optional): The height of the vehicle. The unit is meter. Defaults to None.
            color (tuple, optional): The color of the vehicle. Defaults to None.
            axle_distance (float, optional): The distance between front and rear axles. Defaults to 3.0.
            bucket_length (float, optional): The length of the bucket. Defaults to 2.0.
            rear_section_length (float, optional): The length of the rear section. Defaults to None (auto-calculated).
            front_section_length (float, optional): The length of the front section. Defaults to None (auto-calculated).
            max_articulation (float, optional): The maximum articulation angle. Defaults to π/3.
            max_speed (float, optional): The maximum speed. Defaults to 5.0.
            max_accel (float, optional): The maximum acceleration. Defaults to 2.0.
            max_decel (float, optional): The maximum deceleration. Defaults to 5.0.
            verify (bool): Whether to verify the trajectory. Defaults to False.
            physics_model (ArticulatedVehicleKinematics): The physics model. Defaults to None.
        """
        super().__init__(id_, type_, trajectory, **kwargs)

        # Set default values
        self.axle_distance = kwargs.get("axle_distance", 3.0)
        self.bucket_length = kwargs.get("bucket_length", 2.0)
        self.max_articulation = kwargs.get("max_articulation", np.pi / 3)
        self.max_speed = kwargs.get("max_speed", 5.0)
        self.max_accel = kwargs.get("max_accel", 2.0)
        self.max_decel = kwargs.get("max_decel", 5.0)

        # Calculate section lengths if not provided
        if self.length is not None:
            # Default: rear section is slightly longer than front section
            if kwargs.get("rear_section_length") is None:
                self.rear_section_length = (self.length - self.bucket_length) * 0.55
            else:
                self.rear_section_length = kwargs.get("rear_section_length")
            
            if kwargs.get("front_section_length") is None:
                self.front_section_length = (self.length - self.bucket_length) * 0.45 + self.bucket_length
            else:
                self.front_section_length = kwargs.get("front_section_length")
        else:
            # Default lengths if total length not specified
            self.rear_section_length = kwargs.get("rear_section_length", 2.1)
            self.front_section_length = kwargs.get("front_section_length", 2.0)
            self.length = self.rear_section_length + self.front_section_length

        # Set ranges
        self.articulation_range = (-self.max_articulation, self.max_articulation)
        self.speed_range = (-self.max_speed, self.max_speed)
        self.accel_range = (-self.max_decel, self.max_accel)

        # Initialize physics model if verify is True
        if self.verify:
            if "physics_model" in kwargs and kwargs["physics_model"] is not None:
                self.physics_model = kwargs["physics_model"]
            else:
                self._auto_construct_physics_model()

        # Initialize geometry
        if not None in [self.length, self.width]:
            # Rear section bounding box (relative to rear axle)
            # The hinge is at x = 0.5 * axle_distance
            rear_front_x = 0.5 * self.axle_distance
            rear_back_x = rear_front_x - self.rear_section_length
            
            self._rear_bbox = LinearRing(
                [
                    [rear_front_x, -0.5 * self.width],
                    [rear_front_x, 0.5 * self.width],
                    [rear_back_x, 0.5 * self.width],
                    [rear_back_x, -0.5 * self.width],
                ]
            )
            
            # Front section bounding box (relative to front axle)
            # The hinge is at x = -0.5 * axle_distance
            front_back_x = -0.5 * self.axle_distance
            front_front_x = front_back_x + self.front_section_length
            
            self._front_bbox = LinearRing(
                [
                    [front_front_x, -0.5 * self.width],
                    [front_front_x, 0.5 * self.width],
                    [front_back_x, 0.5 * self.width],
                    [front_back_x, -0.5 * self.width],
                ]
            )
        else:
            self._rear_bbox = None
            self._front_bbox = None

        # Current articulation angle (stored separately as State doesn't include it)
        self.current_articulation = 0.0

    @property
    def geometry(self) -> Tuple[LinearRing, LinearRing]:
        """Get the geometry of the vehicle (rear and front sections).

        Returns:
            Tuple[LinearRing, LinearRing]: (rear_section_geometry, front_section_geometry)
        """
        return self._rear_bbox, self._front_bbox

    def _auto_construct_physics_model(self):
        """Auto-construct the physics model for the wheel loader."""
        if not None in [self.axle_distance]:
            self.physics_model = ArticulatedVehicleKinematics(
                L=self.axle_distance,
                articulation_range=self.articulation_range,
                speed_range=self.speed_range,
                accel_range=self.accel_range,
                interval=100,
            )
        else:
            self.verify = False
            logging.info(
                "Cannot construct a physics model for the wheel loader. The state verification is turned off."
            )

    def bind_trajectory(self, trajectory: Trajectory):
        """Bind a trajectory to the wheel loader.

        Args:
            trajectory (Trajectory): The trajectory to bind.

        Raises:
            TypeError: If the input trajectory is not of type Trajectory.
        """
        if not isinstance(trajectory, Trajectory):
            raise TypeError("The trajectory must be an instance of Trajectory.")

        if self.verify:
            if not self._verify_trajectory(trajectory):
                self.trajectory = Trajectory(self.id_)
                logging.warning(
                    f"The trajectory is invalid. Wheel loader {self.id_} is not bound to the trajectory."
                )
            else:
                self.trajectory = trajectory
        else:
            self.trajectory = trajectory
            logging.debug(f"Wheel loader {self.id_} is bound to a trajectory without verification.")

    def add_state(self, state: State, articulation_angle: float = 0.0):
        """Add a state to the wheel loader.

        Args:
            state (State): The state to add.
            articulation_angle (float, optional): The articulation angle at this state. Defaults to 0.0.
        """
        self.current_articulation = articulation_angle
        if not self.verify or self.physics_model is None:
            self.trajectory.add_state(state)
        elif self.physics_model.verify_state(state, self.trajectory.last_state):
            self.trajectory.add_state(state)
        else:
            raise RuntimeError(
                "Invalid state checked by the physics model %s."
                % (self.physics_model.__class__.__name__)
            )

    def get_pose(self, frame: int = None, articulation_angle: float = None) -> Tuple[Polygon, Polygon]:
        """Get the pose of the wheel loader at the requested frame.

        Args:
            frame (int, optional): The frame to get the vehicle's pose. Defaults to None (current frame).
            articulation_angle (float, optional): The articulation angle. If None, uses current_articulation.

        Returns:
            Tuple[Polygon, Polygon]: (rear_section_pose, front_section_pose) - The bounding boxes of rear and front sections.
        """
        state = self.trajectory.get_state(frame)
        if articulation_angle is None:
            articulation_angle = self.current_articulation

        # Get front axle position
        x_f, y_f, heading_f = self.physics_model.get_front_axle_position(state, articulation_angle)

        # Transform rear section (centered at rear axle)
        transform_matrix_rear = [
            np.cos(state.heading),
            -np.sin(state.heading),
            np.sin(state.heading),
            np.cos(state.heading),
            state.location[0],
            state.location[1],
        ]
        rear_pose = affine_transform(self._rear_bbox, transform_matrix_rear)

        # Transform front section (centered at front axle)
        transform_matrix_front = [
            np.cos(heading_f),
            -np.sin(heading_f),
            np.sin(heading_f),
            np.cos(heading_f),
            x_f,
            y_f,
        ]
        front_pose = affine_transform(self._front_bbox, transform_matrix_front)

        return Polygon(rear_pose), Polygon(front_pose)

    def get_trace(self, frame_range: Tuple[int, int] = None) -> LinearRing:
        """Get the trace of the wheel loader within the requested frame range.

        Args:
            frame_range (Tuple[int, int], optional): The requested frame range.

        Returns:
            LinearRing: The trace of the vehicle (simplified as rear axle trace).
        """
        # Simplified implementation: return rear axle trace
        # Full implementation would require combining both sections' traces
        return None

    def get_rear_axle_position(self, frame: int = None) -> Tuple[float, float, float]:
        """Get the rear axle center position and heading.

        Args:
            frame (int, optional): The frame. Defaults to None (current frame).

        Returns:
            Tuple[float, float, float]: (x, y, heading) - rear axle position and heading.
        """
        state = self.trajectory.get_state(frame)
        return state.x, state.y, state.heading

    def get_front_axle_position(self, frame: int = None, articulation_angle: float = None) -> Tuple[float, float, float]:
        """Get the front axle center position and heading.

        Args:
            frame (int, optional): The frame. Defaults to None (current frame).
            articulation_angle (float, optional): The articulation angle. If None, uses current_articulation.

        Returns:
            Tuple[float, float, float]: (x, y, heading) - front axle position and heading.
        """
        state = self.trajectory.get_state(frame)
        if articulation_angle is None:
            articulation_angle = self.current_articulation
        return self.physics_model.get_front_axle_position(state, articulation_angle)

