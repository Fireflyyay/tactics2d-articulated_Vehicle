##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: articulated_pure_pursuit_controller.py
# @Description: Pure pursuit controller for articulated vehicles (wheel loaders).
# @Author: Tactics2D Team
# @Version: 1.0.0

from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.participant.trajectory.state import State

from .acceleration_controller import AccelerationController


class ArticulatedPurePursuitController:
    """This class implements a pure pursuit controller for articulated vehicles (wheel loaders).

    The controller tracks a path using different reference points based on driving direction:
    - Forward: Uses front axle center as reference point
    - Backward: Uses rear axle center as reference point

    Attributes:
        kp (float): The proportional gain for speed error adjustment. Defaults to 3.5.
        accel_change_rate (float): The limitation to acceleration change rate. Defaults to 3.0 m^2/s.
        max_accel (float): The upper limit of acceleration. Defaults to 1.5 m/s^2.
        min_accel (float): The lower limit of acceleration (deceleration). Defaults to -4.0 m/s^2.
        interval (float): The time interval between commands. Defaults to 1.0 s.
        min_pre_aiming_distance (float): The minimum preview distance. Defaults to 2.0 m.
        target_speed (float): The target speed. Defaults to 2.0 m/s.
    """

    kp = 3.5
    accel_change_rate = 3.0
    max_accel = 1.5
    min_accel = -4.0
    interval = 1.0
    min_pre_aiming_distance = 2.0
    target_speed = 2.0

    def __init__(self, min_pre_aiming_distance: float = 2.0, target_speed: float = 2.0):
        """Initialize the articulated pure pursuit controller.

        Args:
            min_pre_aiming_distance (float, optional): The minimum preview distance. Defaults to 2.0 m.
            target_speed (float, optional): The target speed. Defaults to 2.0 m/s.
        """
        self.min_pre_aiming_distance = min_pre_aiming_distance
        self.target_speed = target_speed
        self._longitudinal_control = AccelerationController(target_speed=target_speed)

    def _lateral_control_forward(
        self, 
        front_axle_state: Tuple[float, float, float], 
        rear_heading: float,
        pre_aiming_point: Point, 
        axle_distance: float
    ) -> float:
        """Calculate articulation angle for forward motion.

        Args:
            front_axle_state (Tuple[float, float, float]): (x, y, heading) of front axle.
            rear_heading (float): The heading of the rear section.
            pre_aiming_point (Point): The preview point on the path.
            axle_distance (float): The distance between front and rear axles.

        Returns:
            float: The articulation angle command (radian).
        """
        x_f, y_f, _ = front_axle_state
        
        # Calculate angle to preview point
        pre_aiming_angle = np.arctan2(
            pre_aiming_point.y - y_f, 
            pre_aiming_point.x - x_f
        )
        
        # Calculate distance to preview point
        distance = np.sqrt(
            (pre_aiming_point.y - y_f)**2 + (pre_aiming_point.x - x_f)**2
        )
        
        # Calculate desired heading change
        # Use rear heading as reference to avoid oscillation caused by 
        # feedback loop (heading_f depends on articulation_angle)
        heading_error = pre_aiming_angle - rear_heading
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # For articulated vehicles, the articulation angle relates to turning radius
        if distance > 0:
            # Calculate desired curvature
            curvature = 2.0 * np.sin(heading_error) / distance
            
            # Convert curvature to articulation angle
            articulation_angle = np.arctan(curvature * axle_distance)
        else:
            articulation_angle = 0.0
        
        return articulation_angle

    def _lateral_control_backward(
        self, 
        rear_axle_state: Tuple[float, float, float], 
        pre_aiming_point: Point, 
        axle_distance: float
    ) -> float:
        """Calculate articulation angle for backward motion.

        Args:
            rear_axle_state (Tuple[float, float, float]): (x, y, heading) of rear axle.
            pre_aiming_point (Point): The preview point on the path.
            axle_distance (float): The distance between front and rear axles.

        Returns:
            float: The articulation angle command (radian).
        """
        x_r, y_r, heading_r = rear_axle_state
        
        # For backward motion, we need to consider the reverse kinematics
        # The preview point should be behind the vehicle
        pre_aiming_angle = np.arctan2(
            pre_aiming_point.y - y_r, 
            pre_aiming_point.x - x_r
        )
        
        distance = np.sqrt(
            (pre_aiming_point.y - y_r)**2 + (pre_aiming_point.x - x_r)**2
        )
        
        # For backward motion, the heading is reversed
        heading_error = pre_aiming_angle - (heading_r + np.pi)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        if distance > 0:
            curvature = 2.0 * np.sin(heading_error) / distance
            # For backward, articulation angle is opposite
            articulation_angle = -np.arctan(curvature * axle_distance)
        else:
            articulation_angle = 0.0
        
        return articulation_angle

    def step(
        self, 
        rear_axle_state: State, 
        front_axle_state: Tuple[float, float, float],
        waypoints: LineString, 
        axle_distance: float = 3.0,
        is_forward: bool = True
    ) -> Tuple[float, float]:
        """Output articulation angle and acceleration commands.

        Args:
            rear_axle_state (State): The current state of rear axle center.
            front_axle_state (Tuple[float, float, float]): (x, y, heading) of front axle center.
            waypoints (LineString): The path to follow.
            axle_distance (float, optional): The distance between axles. Defaults to 3.0 m.
            is_forward (bool, optional): Whether moving forward. Defaults to True.

        Returns:
            Tuple[float, float]: (articulation_angle, accel) - control commands.
        """
        # Determine reference point based on direction
        if is_forward:
            ref_x, ref_y, ref_heading = front_axle_state
            ref_point = Point(ref_x, ref_y)
        else:
            ref_x, ref_y = rear_axle_state.location
            ref_heading = rear_axle_state.heading
            ref_point = Point(ref_x, ref_y)
        
        # Calculate preview distance
        speed = abs(rear_axle_state.speed) if rear_axle_state.speed is not None else 0.0
        pre_aiming_distance = speed * self.interval
        pre_aiming_distance = max(pre_aiming_distance, self.min_pre_aiming_distance)
        
        # Find preview point on path
        # Project reference point onto path
        try:
            # Find closest point on path
            closest_point = waypoints.interpolate(waypoints.project(ref_point))
            
            # Move forward/backward along path
            if is_forward:
                # Move forward along path
                path_length = waypoints.project(closest_point) + pre_aiming_distance
            else:
                # Move backward along path
                path_length = waypoints.project(closest_point) - pre_aiming_distance
            
            # Clamp to path bounds
            path_length = max(0, min(path_length, waypoints.length))
            pre_aiming_point = waypoints.interpolate(path_length)
        except:
            # Fallback: use end point
            if is_forward:
                pre_aiming_point = Point(waypoints.coords[-1])
            else:
                pre_aiming_point = Point(waypoints.coords[0])
        
        # Calculate articulation angle
        if is_forward:
            articulation_angle = self._lateral_control_forward(
                front_axle_state, rear_axle_state.heading, pre_aiming_point, axle_distance
            )
        else:
            articulation_angle = self._lateral_control_backward(
                (ref_x, ref_y, ref_heading), pre_aiming_point, axle_distance
            )
        
        # Calculate acceleration
        _, accel = self._longitudinal_control.step(rear_axle_state)
        
        # Adjust acceleration based on direction
        if not is_forward:
            accel = -accel  # Reverse acceleration for backward motion
        
        return articulation_angle, accel

