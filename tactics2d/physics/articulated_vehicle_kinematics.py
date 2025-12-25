##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: articulated_vehicle_kinematics.py
# @Description: This file implements a kinematic model for an articulated vehicle (wheel loader).
# @Author: Tactics2D Team
# @Version: 1.0.0

from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import State

from .physics_model_base import PhysicsModelBase


class ArticulatedVehicleKinematics(PhysicsModelBase):
    r"""This class implements a kinematic model for an articulated vehicle (wheel loader).

    The model assumes:
    1. The vehicle operates in a 2D plane (x-y).
    2. The vehicle consists of two rigid bodies (front and rear sections) connected by a hinge.
    3. The rear section is driven, and the articulation angle is controlled.
    4. The vehicle's state is represented by the rear axle center position and heading.

    The state includes:
    - x_r, y_r: Rear axle center position
    - heading_r: Rear section heading angle
    - articulation_angle: Hinge angle between front and rear sections

    Attributes:
        L (float): The distance between front and rear axles. The unit is meter.
        articulation_range (Union[float, Tuple[float, float]], optional): The articulation angle range. 
            The unit is radian.
        speed_range (Union[float, Tuple[float, float]], optional): The speed range. 
            The unit is meter per second (m/s).
        accel_range (Union[float, Tuple[float, float]], optional): The acceleration range. 
            The unit is meter per second squared (m/s^2).
        interval (int, optional): The time interval between states. The unit is millisecond. Defaults to 100.
        delta_t (int, optional): The discrete time step. The unit is millisecond. Defaults to 5 ms.
    """

    def __init__(
        self,
        L: float = 3.0,
        l1: float = None,
        l2: float = None,
        articulation_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = 100,
        delta_t: int = None,
    ):
        """Initialize the articulated vehicle kinematic model.

        Args:
            L (float): The distance between front and rear axles. The unit is meter. Defaults to 3.0.
            l1 (float, optional): The distance from front axle to hinge. Defaults to L/2.
            l2 (float, optional): The distance from rear axle to hinge. Defaults to L/2.
            articulation_range (Union[float, Tuple[float, float]], optional): The range of articulation angle. 
                The unit is radian. Defaults to None (no limit).
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. 
                The unit is meter per second (m/s). Defaults to None (no limit).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. 
                The unit is meter per second squared (m/s^2). Defaults to None (no limit).
            interval (int, optional): The time interval between states. The unit is millisecond. Defaults to 100.
            delta_t (int, optional): The discrete time step. The unit is millisecond. Defaults to 5 ms.
        """
        self.L = L
        self.l1 = l1 if l1 is not None else L / 2
        self.l2 = l2 if l2 is not None else L / 2

        if isinstance(articulation_range, float):
            self.articulation_range = None if articulation_range < 0 else [-articulation_range, articulation_range]
        elif hasattr(articulation_range, "__len__") and len(articulation_range) == 2:
            if articulation_range[0] >= articulation_range[1]:
                self.articulation_range = None
            else:
                self.articulation_range = articulation_range
        else:
            self.articulation_range = None

        if isinstance(speed_range, float):
            self.speed_range = None if speed_range < 0 else [-speed_range, speed_range]
        elif hasattr(speed_range, "__len__") and len(speed_range) == 2:
            if speed_range[0] >= speed_range[1]:
                self.speed_range = None
            else:
                self.speed_range = speed_range
        else:
            self.speed_range = None

        if isinstance(accel_range, float):
            self.accel_range = None if accel_range < 0 else [-accel_range, accel_range]
        elif hasattr(accel_range, "__len__") and len(accel_range) == 2:
            if accel_range[0] >= accel_range[1]:
                self.accel_range = None
            else:
                self.accel_range = accel_range
        else:
            self.accel_range = None

        self.interval = interval

        if delta_t is None:
            self.delta_t = self._DELTA_T
        else:
            self.delta_t = max(delta_t, self._MIN_DELTA_T)
            if self.interval is not None:
                self.delta_t = min(self.delta_t, self.interval)

    def _step(self, state: State, accel: float, articulation_angle: float, interval: int) -> Tuple[State, float]:
        """Internal step function for state update.

        Args:
            state (State): Current state (rear axle center).
            accel (float): Acceleration command.
            articulation_angle (float): Target articulation angle command.
            interval (int): Time interval in milliseconds.

        Returns:
            Tuple[State, float]: New state after update and the applied articulation angle.
        """
        dts = [float(self.delta_t) / 1000] * int(interval // self.delta_t)
        if interval % self.delta_t > 0:
            dts.append(float(interval % self.delta_t) / 1000)

        x_r, y_r = state.location
        heading_r = state.heading
        v = state.speed
        
        # Current articulation angle is not stored in State, so we assume it's passed in or tracked externally.
        # However, the step function signature implies we receive the TARGET articulation angle.
        # But wait, the previous implementation treated `articulation_angle` as the CURRENT angle for kinematic calculation?
        # "gamma = articulation_angle".
        # If it's a command, we should move towards it.
        # But standard kinematic models often take steering angle as input state for the step.
        # Let's assume `articulation_angle` is the TARGET angle for the END of the step, 
        # or we need to know the start angle to calculate omega.
        # Since we don't have the start angle in `state`, we have a problem if we want to calculate omega.
        # BUT, the caller `WheelLoader.step` passes `articulation_angle` which comes from the controller.
        # The controller outputs a desired angle.
        # If we want to simulate the dynamics of changing angle, we need the previous angle.
        # The `WheelLoader` class stores `current_articulation`.
        # But `_step` is a method of `ArticulatedVehicleKinematics` which is stateless regarding the vehicle instance.
        # We need to change the signature of `step` or `_step` to accept `current_articulation`.
        # However, I cannot easily change the signature of `step` without breaking other code (like `WheelLoader.add_state` logic?).
        # Wait, `WheelLoader` calls `physics_model.step`.
        # I can change `step` signature in `ArticulatedVehicleKinematics` and update `WheelLoader` call.
        
        # For now, let's assume the input `articulation_angle` is the DESIRED angle, 
        # and we need to estimate omega.
        # Or, we can assume the input `articulation_angle` IS the angle during this step (held constant?), 
        # but that gives omega=0.
        # The user wants "Control input is hinge torque... produces omega".
        # This implies we should simulate the change.
        
        # Let's assume for this implementation that we are given the target angle, 
        # and we move towards it with some max omega, or we just calculate the kinematic change 
        # assuming we reach it linearly over `interval`.
        # But we don't know the starting angle here!
        # I will modify `step` to take `current_articulation_angle` as an optional argument.
        pass
        
    def step(self, state: State, accel: float, articulation_angle: float, interval: int = None, current_articulation: float = None) -> Tuple[State, float, float]:
        """Update the state of the articulated vehicle.

        Args:
            state (State): The current state (rear axle center).
            accel (float): The acceleration command. The unit is m/s^2.
            articulation_angle (float): The target articulation angle command. The unit is radian.
            interval (int, optional): The time interval. The unit is millisecond. Defaults to None.
            current_articulation (float, optional): The current articulation angle. Defaults to None.

        Returns:
            Tuple[State, float, float]: (new_state, applied_accel, applied_articulation_angle)
        """
        # Clip inputs to valid ranges
        if self.accel_range is not None:
            accel = np.clip(accel, *self.accel_range)
        if self.articulation_range is not None:
            articulation_angle = np.clip(articulation_angle, *self.articulation_range)
        
        interval = interval if interval is not None else self.interval
        
        # If current_articulation is not provided, assume it's the same as target (no steering change)
        if current_articulation is None:
            current_articulation = articulation_angle

        next_state, final_articulation = self._step(state, accel, articulation_angle, interval, current_articulation)

        return next_state, accel, final_articulation

    def _step(self, state: State, accel: float, target_articulation: float, interval: int, current_articulation: float) -> Tuple[State, float]:
        dts = [float(self.delta_t) / 1000] * int(interval // self.delta_t)
        if interval % self.delta_t > 0:
            dts.append(float(interval % self.delta_t) / 1000)

        x_r, y_r = state.location
        heading_r = state.heading
        v = state.speed
        gamma = current_articulation
        
        # Calculate omega required to reach target
        total_dt = interval / 1000.0
        if total_dt > 0:
            omega = (target_articulation - current_articulation) / total_dt
        else:
            omega = 0

        for dt in dts:
            # Kinematic equations for articulated vehicle
            # Unified model including static steering (scrubbing)
            # theta2_dot = (v * sin(gamma) - l2 * omega * cos(gamma)) / (l1 * cos(gamma) + l2)
            
            denom = self.l1 * np.cos(gamma) + self.l2
            dheading_r = (v * np.sin(gamma) - self.l2 * omega * np.cos(gamma)) / denom
            
            # Update heading
            heading_r += dheading_r * dt
            
            # Update articulation angle
            gamma += omega * dt
            
            # Update position
            # Rear axle moves due to forward velocity AND steering rotation around hinge
            # dx_r = v * cos(heading_r) + dheading_r_steer * l2 * sin(heading_r)
            # dy_r = v * sin(heading_r) - dheading_r_steer * l2 * cos(heading_r)
            # where dheading_r_steer is the part of dheading_r due to omega
            
            dheading_r_steer = (-self.l2 * omega * np.cos(gamma)) / denom
            
            dx_r = v * np.cos(heading_r) + dheading_r_steer * self.l2 * np.sin(heading_r)
            dy_r = v * np.sin(heading_r) - dheading_r_steer * self.l2 * np.cos(heading_r)
            
            x_r += dx_r * dt
            y_r += dy_r * dt
            
            v += accel * dt

            # Clip speed to range
            if self.speed_range is not None:
                v = np.clip(v, *self.speed_range)

        # Normalize heading to [0, 2*pi]
        heading_r = np.mod(heading_r, 2 * np.pi)

        # Create new state
        new_state = State(
            frame=state.frame + interval,
            x=x_r,
            y=y_r,
            heading=heading_r,
            vx=v * np.cos(heading_r),
            vy=v * np.sin(heading_r),
            speed=v,
            accel=accel,
        )

        return new_state, gamma

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """Verify the validity of a state transition.

        Args:
            state (State): The new state.
            last_state (State): The last state.
            interval (int, optional): The time interval. The unit is millisecond.

        Returns:
            bool: True if the state transition is valid, False otherwise.
        """
        if interval is None:
            interval = state.frame - last_state.frame
        
        dt = float(interval) / 1000

        # Basic checks
        if None in [self.articulation_range, self.speed_range, self.accel_range]:
            return True

        # Check speed range
        if self.speed_range is not None:
            if not (self.speed_range[0] <= state.speed <= self.speed_range[1]):
                return False

        # Check acceleration constraints (rough check)
        if self.accel_range is not None and dt > 0:
            speed_change = state.speed - last_state.speed
            max_speed_change = self.accel_range[1] * dt
            min_speed_change = self.accel_range[0] * dt
            if not (min_speed_change <= speed_change <= max_speed_change):
                return False

        # Check position change (rough check based on max speed)
        if self.speed_range is not None and dt > 0:
            max_speed = max(abs(self.speed_range[0]), abs(self.speed_range[1]))
            max_distance = max_speed * dt * 1.5  # Allow some margin
            distance = np.sqrt((state.x - last_state.x)**2 + (state.y - last_state.y)**2)
            if distance > max_distance:
                return False

        return True

    def get_front_axle_position(self, state: State, articulation_angle: float) -> Tuple[float, float, float]:
        """Get the front axle center position and heading.

        Args:
            state (State): The current state (rear axle center).
            articulation_angle (float): The articulation angle. The unit is radian.

        Returns:
            Tuple[float, float, float]: (x_f, y_f, heading_f) - front axle position and heading.
        """
        x_r, y_r = state.location
        heading_r = state.heading
        
        # Hinge point (using l2)
        # Assuming rear axle is behind hinge, so hinge is ahead of rear axle along heading_r
        # O = O2 + l2 * [cos(theta2), sin(theta2)]
        x_j = x_r + self.l2 * np.cos(heading_r)
        y_j = y_r + self.l2 * np.sin(heading_r)
        
        # Front section heading
        heading_f = heading_r + articulation_angle
        
        # Front axle center (using l1)
        # Assuming front axle is ahead of hinge along heading_f
        # O1 = O + l1 * [cos(theta1), sin(theta1)]
        x_f = x_j + self.l1 * np.cos(heading_f)
        y_f = y_j + self.l1 * np.sin(heading_f)
        
        return x_f, y_f, heading_f

