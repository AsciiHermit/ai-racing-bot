# ARS -- Autonomous Racing Simulator

A lightweight, 2D, math-only (no rendering-heavy) multi-agent racing simulator
for RL research. v1 scope: single EV, simple closed-loop track, modular
architecture so each teammate can own and build one piece independently.

## Architecture

Everything downstream depends only on the contracts in `ars/core/interfaces.py`
(three Python `Protocol`s), never on a concrete implementation:

```
ars/core/       Protocols + shared dataclasses (VehiclePhysics, Track, Sensor)
ars/physics/    Vehicle dynamics -- implements VehiclePhysics
ars/track/      Track geometry -- implements Track
ars/sensors/    Observation sources -- implement Sensor
ars/env/        Gymnasium env wrapper -- composes the three above
ars/agents/     RL agents (baseline + novel algorithm)
```

Because everything is typed against the Protocols, any implementation can be
swapped without touching the others -- e.g. the physics teammate can replace
`KinematicBicyclePhysics` with a full 4-wheel dynamic model, and nothing in
`ars/env/` or `ars/agents/` needs to change, as long as the new class exposes
`reset(x, y, heading)` and `step(state, action, dt)`.

### Module ownership (v1)

| Module | Owns | Status |
|---|---|---|
| `ars/physics` | Vehicle dynamics | `KinematicBicyclePhysics` stub in place -- **replace with real EV physics** |
| `ars/track` | Track geometry | `FixedLoopTrack` + `make_simple_oval()` -- simple loop, no banking |
| `ars/sensors` | Observations | `TrackPoseSensor`, `ProprioceptiveSensor`, `LidarSensor` |
| `ars/env` | Gym integration | `RacingEnv`, registered as `ARS-Racing-v0` |
| `ars/agents` | RL agents | `RandomAgent` baseline only so far |

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import ars.env  # registers "ARS-Racing-v0"
import gymnasium as gym

env = gym.make("ARS-Racing-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Or use the factory directly:

```python
from ars.env import make_default_env

env = make_default_env()
```

Manual end-to-end check (random agent, one episode):

```bash
python scripts/smoke_test.py
```

## Testing

```bash
pytest
```

Each module has its own test directory (`tests/physics`, `tests/track`,
`tests/sensors`, `tests/env`) so a teammate can run and iterate on just
their module: `pytest tests/physics -q`.

## Contract details

- **`VehiclePhysics`** (`ars/core/interfaces.py`): `reset(x, y, heading) -> VehicleState`,
  `step(state, action, dt) -> VehicleState`. Physics teammate's implementation
  is pure -- no hidden state beyond what's in `VehicleState`; all vehicle
  parameters (mass, tire curve, etc.) are constructor config.
- **`Track`**: `query(x, y) -> TrackSample` (nearest-point projection),
  `sample_at_s(s) -> TrackSample`, `is_on_track(x, y) -> bool`.
- **`Sensor`**: `read(state, track) -> SensorFrame`, contributes one named
  array. The env concatenates whatever sensor list it's given, in order --
  observation space shape follows automatically.

All cross-module data is plain dataclasses/numpy arrays from `ars/core/types.py`
-- never a module's internal representation.

## What's a stub vs. real

- `ars.physics.KinematicBicyclePhysics` is a placeholder (no tire slip, no
  load transfer, no aero) so the rest of the stack has something to run
  against on day one. This is Module A's real work -- replace it with a
  proper EV dynamics model behind the same `VehiclePhysics` Protocol.
- `ars.track.make_simple_oval()` is a genuine v1 track (two straights, two
  same-direction 180-degree turns) -- not a stub, but intentionally simple.
  A closed loop's turns must sum to a full 2*pi in signed curvature, so this
  two-turn oval only ever turns one direction; a track with a real mixed
  left+right turn sequence needs either careful multi-arc closure math or a
  waypoint/spline-based `Track` implementation -- left for v1.1/backlog.
