##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description:
# @Author: Tactics2D Team
# @Version:

from .articulated_mpc_controller import (
    HAS_SCIPY,
    ArticulatedMPCController,
    ArticulatedReferenceTrajectory,
    MPCSolveResult,
    build_articulated_reference_trajectory,
)
from .pid_controller import PIDController

try:
    from .articulated_pure_pursuit_controller import ArticulatedPurePursuitController
except ModuleNotFoundError:
    ArticulatedPurePursuitController = None

try:
    from .pure_pursuit_controller import PurePursuitController
except ModuleNotFoundError:
    PurePursuitController = None

__all__ = [
    "PIDController",
    "ArticulatedMPCController",
    "ArticulatedReferenceTrajectory",
    "MPCSolveResult",
    "build_articulated_reference_trajectory",
    "HAS_SCIPY",
]
if PurePursuitController is not None:
    __all__.append("PurePursuitController")
if ArticulatedPurePursuitController is not None:
    __all__.append("ArticulatedPurePursuitController")
