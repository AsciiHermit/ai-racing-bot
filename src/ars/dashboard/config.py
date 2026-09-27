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
    """Physics backend choice. "dynamic_bicycle" (Phase 1's slip-based tire
    model) is the default; "kinematic_stub" is kept as a faster/degenerate
    option for ablations, not a live tunable-params picker yet."""

    model_name: str = "dynamic_bicycle"


@dataclass
class AgentConfig:
    kind: str = "dummy_expert"  # only option in v1


@dataclass
class SessionConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
