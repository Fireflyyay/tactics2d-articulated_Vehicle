##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the map generator module.
# @Author: Yueyuan Li
# @Version: 1.0.0


from .generate_parking_lot import ParkingLotGenerator
from .generate_ppo_parking_map import PPOParkingMapGenerator
try:
	from .generate_racing_track import RacingTrackGenerator
except ModuleNotFoundError:
	RacingTrackGenerator = None

__all__ = ["ParkingLotGenerator", "PPOParkingMapGenerator"]
if RacingTrackGenerator is not None:
	__all__.append("RacingTrackGenerator")
