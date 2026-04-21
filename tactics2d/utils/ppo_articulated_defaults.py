import numpy as np
from shapely.geometry import LinearRing


PPO_ARTICULATED_COLOR = (30, 144, 255, 255)

PPO_FRONT_OVERHANG = 1.0
PPO_REAR_OVERHANG = 1.0
PPO_WIDTH = 2.0
PPO_HITCH_OFFSET = 1.5
PPO_TRAILER_LENGTH = 1.5
PPO_WHEEL_BASE = PPO_HITCH_OFFSET + PPO_TRAILER_LENGTH
PPO_LENGTH = PPO_FRONT_OVERHANG + PPO_WHEEL_BASE + PPO_REAR_OVERHANG

PPO_MAX_ARTICULATION = float(np.deg2rad(36.0))
PPO_SPEED_RANGE = (-2.5, 2.5)
PPO_ACCEL_RANGE = (-1.0, 1.0)
PPO_ANGULAR_SPEED_RANGE = (-0.5, 0.5)

PPO_DEFAULT_INTERVAL_MS = 200.0
PPO_DEFAULT_DELTA_T_MS = 2.5
PPO_START_DEST_ARTICULATION_RANGE = (-float(np.deg2rad(10.0)), float(np.deg2rad(10.0)))

PPO_DEFAULT_MAP_LEVEL = "Normal"
PPO_NAVIGATION_BOUNDARY = (-40.0, 40.0, -40.0, 40.0)
PPO_SCENE_MARGIN = 13.0

PPO_START_AREA_COLOR = "#6495ed"
PPO_TARGET_AREA_COLOR = "#458b00"


def build_front_vehicle_box(width: float, hitch_offset: float, front_overhang: float) -> LinearRing:
    return LinearRing(
        [
            (-hitch_offset, -0.5 * width),
            (front_overhang, -0.5 * width),
            (front_overhang, 0.5 * width),
            (-hitch_offset, 0.5 * width),
        ]
    )


def build_rear_vehicle_box(width: float, trailer_length: float, rear_overhang: float) -> LinearRing:
    return LinearRing(
        [
            (-rear_overhang, -0.5 * width),
            (trailer_length, -0.5 * width),
            (trailer_length, 0.5 * width),
            (-rear_overhang, 0.5 * width),
        ]
    )