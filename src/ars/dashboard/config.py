"""SessionConfig: the state accumulated as a user walks Steps 1-5.

Plain dataclasses only -- no pygame here -- so config can be built/tested
headless and, later, saved/loaded as JSON without touching the UI code.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VehicleConfig:
    mass_kg: float = 798.0  # F1-scale default
    length_m: float = 4.5
    width_m: float = 2.0


@dataclass
class LidarConfig:
    max_range_m: float = 40.0
    # v1: exactly one forward-facing ray. Not user-editable yet -- surfaced
    # as read-only in the UI. Multi-ray / configurable FOV is backlog.
    num_rays: int = 1
    fov_rad: float = 0.0


@dataclass
class PhysicsConfig:
    """v1 placeholder -- no real choice or params yet, one model exists."""

    model_name: str = "kinematic_stub (v1 default, no tunable params yet)"


@dataclass
class AgentConfig:
    kind: str = "dummy_expert"  # only option in v1


@dataclass
class SessionConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
