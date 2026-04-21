from typing import Tuple, Union

import numpy as np

from tactics2d.participant.trajectory import ArticulatedState, State
from tactics2d.participant.trajectory.articulated_state import wrap_angle

from .physics_model_base import PhysicsModelBase


class ArticulatedVehicleKinematics(PhysicsModelBase):
    r"""PPO-aligned active-articulation kinematics.

    The reference state follows PPO_articulated_vehicle:
    - ``x, y, heading`` describe the front axle center and front body heading.
    - ``rear_heading`` stores the rear body heading.
    - controls are a steering-rate-like articulation command and a commanded speed.
    """

    def __init__(
        self,
        L: float = None,
        l1: float = None,
        l2: float = None,
        articulation_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        steering_rate_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: Union[int, float] = 200,
        delta_t: Union[int, float] = 2.5,
    ):
        if l1 is None and l2 is None and L is None:
            l1 = 1.5
            l2 = 1.5
        elif l1 is None and l2 is None:
            l1 = L / 2
            l2 = L / 2
        elif l1 is None:
            l1 = float(L - l2) if L is not None else float(l2)
        elif l2 is None:
            l2 = float(L - l1) if L is not None else float(l1)

        self.hitch_offset = float(l1)
        self.trailer_length = float(l2)
        self.L = float(self.hitch_offset + self.trailer_length)
        self.l1 = self.hitch_offset
        self.l2 = self.trailer_length

        self.articulation_range = self._normalize_range(articulation_range)
        self.speed_range = self._normalize_range(speed_range)
        self.steering_rate_range = self._normalize_range(steering_rate_range)
        if self.steering_rate_range is None:
            self.steering_rate_range = self.articulation_range
        self.accel_range = self._normalize_range(accel_range)

        self.interval = float(interval)
        self.delta_t = float(delta_t) if delta_t is not None else 2.5
        if self.delta_t <= 0:
            self.delta_t = 2.5

    def _normalize_range(self, value):
        if isinstance(value, (int, float)):
            magnitude = float(value)
            return None if magnitude < 0 else (-magnitude, magnitude)
        if hasattr(value, "__len__") and len(value) == 2:
            low = float(value[0])
            high = float(value[1])
            return None if low >= high else (low, high)
        return None

    def _clip(self, value: float, value_range):
        if value_range is None:
            return float(value)
        return float(np.clip(value, value_range[0], value_range[1]))

    def ensure_articulated_state(
        self, state: State, current_articulation: float = None
    ) -> ArticulatedState:
        if isinstance(state, ArticulatedState):
            state.update_trailer_loc(self.hitch_offset, self.trailer_length)
            return state

        articulation = 0.0 if current_articulation is None else float(current_articulation)
        rear_heading = float(state.heading) - articulation
        articulated_state = ArticulatedState(
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
            steering=getattr(state, "steering", 0.0),
        )
        articulated_state.update_trailer_loc(self.hitch_offset, self.trailer_length)
        return articulated_state

    def _iter_time_steps(self, interval: float):
        remaining = float(interval) / 1000.0
        dt = float(self.delta_t) / 1000.0
        if remaining <= 0:
            return []

        dts = []
        while remaining > 1e-12:
            current_dt = min(dt, remaining)
            dts.append(current_dt)
            remaining -= current_dt

        return dts

    def _step_aligned(
        self,
        state: ArticulatedState,
        steering: float,
        speed: float,
        interval: float,
        accel: float = None,
    ) -> ArticulatedState:
        front_x = float(state.x)
        front_y = float(state.y)
        front_heading = float(state.heading)
        rear_heading = float(state.rear_heading)
        commanded_speed = float(speed)
        commanded_steering = float(steering)

        articulation_low = None
        articulation_high = None
        if self.articulation_range is not None:
            articulation_low, articulation_high = self.articulation_range

        for dt in self._iter_time_steps(interval):
            articulation = wrap_angle(front_heading - rear_heading)
            effective_steering = commanded_steering
            if articulation_low is not None and articulation_high is not None:
                if articulation >= articulation_high and commanded_steering > 0:
                    effective_steering = 0.0
                elif articulation <= articulation_low and commanded_steering < 0:
                    effective_steering = 0.0

            denom = self.hitch_offset * np.cos(articulation) + self.trailer_length
            if abs(denom) < 1e-6:
                denom = 1e-6

            front_heading_dot = (
                commanded_speed * np.sin(articulation) + self.trailer_length * effective_steering
            ) / denom
            rear_heading_dot = front_heading_dot - effective_steering
            x_dot = commanded_speed * np.cos(front_heading)
            y_dot = commanded_speed * np.sin(front_heading)

            front_x += x_dot * dt
            front_y += y_dot * dt
            front_heading += front_heading_dot * dt
            rear_heading += rear_heading_dot * dt

        next_state = ArticulatedState(
            frame=int(round(state.frame + interval)),
            x=front_x,
            y=front_y,
            heading=front_heading,
            vx=commanded_speed * np.cos(front_heading),
            vy=commanded_speed * np.sin(front_heading),
            speed=commanded_speed,
            accel=accel,
            rear_heading=rear_heading,
            steering=commanded_steering,
        )
        next_state.update_trailer_loc(self.hitch_offset, self.trailer_length)
        return next_state

    def step(
        self,
        state: State,
        steering: float = None,
        speed: float = None,
        interval: Union[int, float] = None,
        action: Tuple[float, float] = None,
        accel: float = None,
        articulation_angle: float = None,
        current_articulation: float = None,
    ):
        """Advance the articulated vehicle state.

        Aligned mode:
            ``step(state, steering, speed, interval=...)``
            ``step(state, action=(steering, speed), interval=...)``

        Legacy compatibility mode:
            ``step(state, accel=..., articulation_angle=..., interval=..., current_articulation=...)``
        """
        interval = float(self.interval if interval is None else interval)
        articulated_state = self.ensure_articulated_state(
            state, current_articulation=current_articulation
        )

        if action is not None:
            steering, speed = action

        if accel is not None or articulation_angle is not None:
            if accel is None or articulation_angle is None:
                raise TypeError(
                    "Legacy compatibility mode requires both accel and articulation_angle."
                )

            applied_accel = self._clip(accel, self.accel_range)
            total_dt = interval / 1000.0
            base_speed = 0.0 if articulated_state.speed is None else articulated_state.speed
            target_speed = base_speed + applied_accel * total_dt
            target_speed = self._clip(target_speed, self.speed_range)

            target_articulation = self._clip(articulation_angle, self.articulation_range)
            current_articulation_value = articulated_state.articulation_angle
            steering_command = 0.0
            if total_dt > 0:
                steering_command = (
                    target_articulation - current_articulation_value
                ) / total_dt
            steering_command = self._clip(steering_command, self.steering_rate_range)

            next_state = self._step_aligned(
                articulated_state,
                steering=steering_command,
                speed=target_speed,
                interval=interval,
                accel=applied_accel,
            )
            return next_state, applied_accel, next_state.articulation_angle

        if steering is None or speed is None:
            raise TypeError(
                "Aligned mode requires steering and speed, or action=(steering, speed)."
            )

        applied_steering = self._clip(steering, self.steering_rate_range)
        applied_speed = self._clip(speed, self.speed_range)
        next_state = self._step_aligned(
            articulated_state,
            steering=applied_steering,
            speed=applied_speed,
            interval=interval,
            accel=0.0,
        )
        return next_state, applied_steering, applied_speed

    def verify_state(
        self, state: State, last_state: State, interval: Union[int, float] = None
    ) -> bool:
        interval = float(state.frame - last_state.frame if interval is None else interval)
        if interval <= 0:
            return False

        articulated_state = self.ensure_articulated_state(state)
        articulated_last_state = self.ensure_articulated_state(last_state)
        dt = interval / 1000.0

        if self.speed_range is not None:
            if not self.speed_range[0] <= articulated_state.speed <= self.speed_range[1]:
                return False

        articulation = articulated_state.articulation_angle
        if self.articulation_range is not None:
            if not self.articulation_range[0] <= articulation <= self.articulation_range[1]:
                return False

        if self.steering_rate_range is not None:
            articulation_delta = wrap_angle(
                articulated_state.articulation_angle - articulated_last_state.articulation_angle
            )
            articulation_rate = articulation_delta / dt
            if not self.steering_rate_range[0] <= articulation_rate <= self.steering_rate_range[1]:
                return False

        max_speed = abs(articulated_state.speed)
        if self.speed_range is not None:
            max_speed = max(abs(self.speed_range[0]), abs(self.speed_range[1]))
        max_distance = max_speed * dt * 1.1 + 1e-6
        actual_distance = np.hypot(
            articulated_state.x - articulated_last_state.x,
            articulated_state.y - articulated_last_state.y,
        )
        if actual_distance > max_distance:
            return False

        return True

    def get_front_axle_position(
        self, state: State, articulation_angle: float = None
    ) -> Tuple[float, float, float]:
        articulated_state = self.ensure_articulated_state(
            state, current_articulation=articulation_angle
        )
        return articulated_state.x, articulated_state.y, articulated_state.heading

    def get_rear_axle_position(
        self, state: State, articulation_angle: float = None
    ) -> Tuple[float, float, float]:
        articulated_state = self.ensure_articulated_state(
            state, current_articulation=articulation_angle
        )
        return articulated_state.get_rear_axle_position(self.hitch_offset, self.trailer_length)

    def get_hinge_position(
        self, state: State, articulation_angle: float = None
    ) -> Tuple[float, float]:
        articulated_state = self.ensure_articulated_state(
            state, current_articulation=articulation_angle
        )
        return articulated_state.get_hinge_position(self.hitch_offset)