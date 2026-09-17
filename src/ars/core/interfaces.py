"""Abstract contracts between modules.

Ownership (v1):
  VehiclePhysics -> Module A (physics teammate)
  Track          -> Module B (software, this repo ships one fixed loop track)
  Sensor         -> Module C (software, minimal set for v1)

Everything downstream (env/, agents/) depends only on these Protocols, never
on a concrete implementation. Swap any implementation without touching the
env wrapper as long as it satisfies the contract.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ars.core.types import SensorFrame, TrackSample, VehicleAction, VehicleState


@runtime_checkable
class VehiclePhysics(Protocol):
    """Owned by the physics teammate (Module A).

    Implementations are pure/stateless w.r.t. anything outside the returned
    VehicleState: given a state and an action, produce the next state. Any
    internal params (mass, tire curve, etc.) are constructor config.
    """

    def reset(self, x: float, y: float, heading: float) -> VehicleState:
        """Return an initial at-rest VehicleState at the given pose."""
        ...

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        """Advance the vehicle by dt seconds under the given control input."""
        ...


@runtime_checkable
class Track(Protocol):
    """Owned by the software team (Module B). v1 ships FixedLoopTrack."""

    length: float  # total centerline arc length, m

    def query(self, x: float, y: float) -> TrackSample:
        """Project a world point onto the track and return local geometry."""
        ...

    def sample_at_s(self, s: float) -> TrackSample:
        """Return centerline geometry at arc-length s (for spawning, etc.)."""
        ...

    def is_on_track(self, x: float, y: float) -> bool:
        ...


@runtime_checkable
class Sensor(Protocol):
    """Owned by the software team (Module C). Each sensor contributes one
    or more named arrays to the SensorFrame; agents read by key."""

    name: str

    def read(self, state: VehicleState, track: Track) -> SensorFrame:
        ...
