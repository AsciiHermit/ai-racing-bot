# ARS -- Autonomous Racing Simulator

A lightweight, 2D, math-first (rendering is optional, not the core loop)
multi-agent racing simulator for RL research. v1 scope: single 4-wheel F1-style
EV, simple closed-loop track, modular architecture so each teammate can own
and build one piece independently.

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
ars/viz/        Optional pygame live viewer -- reads env.track/env.vehicle_state only
ars/dashboard/  5-step setup GUI (Track -> Vehicle -> Physics -> Agent -> Simulation)
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
| `ars/sensors` | Observations | `GpsSensor`, `ImuSensor` (localization), `TrackPoseSensor`, `ProprioceptiveSensor`, `LidarSensor` |
| `ars/env` | Gym integration | `RacingEnv`, registered as `ARS-Racing-v0` |
| `ars/agents` | RL agents | `RandomAgent`, `DummyExpertAgent` (hand-coded line follower) |
| `ars/viz` | Visualization | `LiveViewer` (pygame) -- track + car rendered live |
| `ars/dashboard` | Setup GUI | `DashboardApp` -- 5-step config + run flow, see below |

**If you're picking up the physics/vehicle-dynamics work, read
[PHYSICS.md](PHYSICS.md) instead of starting from this README** -- it's a
dedicated handoff doc covering exactly what the current placeholder does
and doesn't do, the interface you need to satisfy, and the open design
question you'll hit early (how a 4-wheel state type fits into the shared
`VehicleState`).

## Setup (clone + install)

**Requires Python 3.12.** `pygame` (the `viz` extra, needed for the live
viewer and dashboard) has no prebuilt wheel for Python 3.13/3.14 yet and
fails to build from source unless you have a full MSVC toolchain installed
-- save yourself the trouble and use 3.12.

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/ars.git
cd ars
```

### 2. Get Python 3.12

Check what you have first:

```bash
python3.12 --version   # macOS/Linux
py -3.12 --version     # Windows
```

If that fails, install it:

- **Windows**: `py install 3.12` (via the [py launcher](https://docs.python.org/3/using/windows.html#the-python-launcher-for-windows)), or download from [python.org](https://www.python.org/downloads/).
- **macOS**: `brew install python@3.12`, or download from [python.org](https://www.python.org/downloads/).
- **Linux**: `sudo apt install python3.12 python3.12-venv` (Debian/Ubuntu) or your distro's equivalent.

If your system already has multiple Python versions installed, always use
a **virtual environment** (below) rather than relying on whichever `python`
resolves first on `PATH` -- mixing installs is a common source of "works
for me" bugs across a team.

### 3. Create a virtual environment and install

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev,viz]"

# macOS/Linux
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,viz]"
```

This installs the package in editable mode (code changes take effect
immediately, no reinstall needed) plus:
- `dev` -- `pytest`, for running the test suite
- `viz` -- `pygame`, for the live viewer and setup dashboard

Everyone on the team should use their own `.venv` (already gitignored)
rather than a shared/system Python.

### 4. Verify it worked

```bash
# Windows
.venv\Scripts\python -m pytest -q

# macOS/Linux
.venv/bin/python -m pytest -q
```

All tests should pass. Then try the dashboard (see [below](#dashboard-setup-gui)):

```bash
# Windows
.venv\Scripts\python scripts\dashboard.py

# macOS/Linux
.venv/bin/python scripts/dashboard.py
```

The rest of this README uses bare `python`/`pytest` for brevity -- substitute
`.venv\Scripts\python` / `.venv/bin/python` (or activate the venv first:
`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on
macOS/Linux) for whichever your shell needs.

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

Manual end-to-end check (random agent, one episode, no rendering):

```bash
python scripts/smoke_test.py
```

## Visualization

`ars.viz.LiveViewer` opens a pygame window and renders the track (boundaries
+ centerline), the car (top-down F1 sprite, `ars/viz/assets/f1_car_top_down.jpg`,
scaled to the configured length/width and rotated to match heading), and the
forward lidar's range as a line from the car's nose -- each frame reading
only `env.track` and `env.vehicle_state`, plus `ViewerConfig.lidar_range`/
`lidar_fov` for the sensor overlay. It doesn't know or care which physics/
track implementation is behind those. Drive it yourself with the keyboard:

```bash
python scripts/drive.py
```

Arrow keys / WASD to throttle, brake, steer. Esc or close the window to quit.

To render an agent's rollout instead of driving manually, call
`viewer.draw(env.vehicle_state)` once per `env.step()` in your own loop --
see `scripts/drive.py` for the pattern (poll events, step env, set HUD text,
draw).

## Dashboard (setup GUI)

A step-by-step pygame GUI for configuring and launching a run, so a user
doesn't need to write Python to try the simulator:

```bash
python scripts/dashboard.py
```

Five steps, Next/Back to move between them (`ars/dashboard/steps/`):

1. **Track** -- visualizer only, no config. v1 has one fixed track (simple
   oval); shows total length, per-segment length, turn radius, track width.
2. **Vehicle** -- mass, length, width (numeric fields), plus the one fixed
   v1 sensor (forward lidar) with its range editable. A live top-down
   preview shows the car sprite and the lidar range drawn to scale, so
   editing a field visibly changes the picture. Sensor count/type isn't
   configurable yet.
3. **Physics** -- placeholder. v1 has one physics model and no tunable
   params; becomes a real model picker once the 4-wheel F1 physics lands.
4. **Agent** -- v1 ships one option, `DummyExpertAgent` (`ars/agents/dummy_expert.py`):
   a hand-coded, non-learned line-following controller (steers toward
   centerline, eases off throttle in turns). "Bring your own RL agent" /
   "train it" are named in the UI as the intended v1.1+ paths.
5. **Simulation** -- read-only summary of steps 1-4, plus a **Run** button
   that builds the env + agent from your choices (`ars/dashboard/builder.py`)
   and opens the live viewer, agent-driving instead of keyboard-driving.
   Esc or close that window to return to the dashboard.

`SessionConfig` (`ars/dashboard/config.py`) is the plain-dataclass state
that accumulates across steps -- Back never loses what you entered. Each
`Screen` subclass only touches its own slice of it, so adding a new
configurable field (e.g. once physics gets real params) means editing one
step's screen, not the whole flow.

## Testing

```bash
pytest
```

Each module has its own test directory (`tests/physics`, `tests/track`,
`tests/sensors`, `tests/env`, `tests/viz`, `tests/dashboard`, `tests/agents`)
so a teammate can run and iterate on just their module: `pytest tests/physics -q`.
`tests/viz` and `tests/dashboard` run headless (SDL dummy video driver) so
they work without a display, e.g. in CI.

## Contract details

- **`VehiclePhysics`** (`ars/core/interfaces.py`): `reset(x, y, heading) -> VehicleState`,
  `step(state, action, dt) -> VehicleState`. Physics teammate's implementation
  is pure -- no hidden state beyond what's in `VehicleState`; all vehicle
  parameters (mass, tire curve, etc.) are constructor config.
- **`Track`**: `query(x, y) -> TrackSample` (nearest-point projection),
  `sample_at_s(s) -> TrackSample`, `is_on_track(x, y) -> bool`.
- **`Sensor`**: `read(state, track) -> SensorFrame`, contributes one named
  array. The env concatenates whatever sensor list it's given, in order --
  observation space shape follows automatically. Also `reset(state) ->
  None`, called once per episode by `RacingEnv.reset()` before the first
  `read()` -- a no-op for stateless sensors, but required for any sensor
  that carries memory across steps (e.g. `ImuSensor` integrating
  acceleration from consecutive velocities -- without this hook, the
  first reading of a new episode would difference against the previous
  episode's final velocity and report a spurious spike).

All cross-module data is plain dataclasses/numpy arrays from `ars/core/types.py`
-- never a module's internal representation.

## Sensors (v1 fixed set)

- **`GpsSensor`** -- world position as **real latitude/longitude** (degrees),
  not raw sim x/y. The track's local `(0, 0)` origin is anchored to an
  arbitrary real-world lat/lon reference point (`GpsSensor.origin_lat_deg`/
  `origin_lon_deg`, default: equator/prime meridian) and local meters are
  converted via the equirectangular (local-tangent-plane) approximation --
  accurate at track-scale distances. Swap the origin for a real circuit's
  coordinates to anchor the sim there instead.
- **`ImuSensor`** -- accelerometer (body-frame `ax`, `ay`) + gyroscope
  (`yaw_rate`). This is what a real IMU actually measures -- *not*
  position or velocity. `VehicleState` has no acceleration field, so
  `ImuSensor` derives it by differencing consecutive `(vx, vy)` readings
  between calls (`(v_now - v_prev) / dt`), which is why it needs the
  `reset()` hook above. Velocity/odometry sensing (e.g. wheel speed) is a
  separate concern, left for the physics teammate's real vehicle model.
- Both are **ideal/noiseless** for v1 (no jitter, drift, bias, or
  update-rate limiting) -- ground-truth readings, matching how
  `TrackPoseSensor`/`ProprioceptiveSensor` already work. A noise model is
  v1.1+ scope.
- Lidar, GPS, and IMU are the only sensors in v1 -- multiple/configurable
  sensor types are explicitly out of scope for now (v1.1 backlog).

## Scale / real-world units

Every length is meters, every mass is kg, every speed is m/s, every angle
internally is radians (UI labels convert to degrees for display) -- one
consistent unit system across track, physics, sensors, viz, and dashboard,
so a value from one module is always directly usable by another with no
hidden conversion.

Reference values are checked against real F1 (2024-era regs/performance):
vehicle mass 798 kg, dims 4.5 x 2.0 m, wheelbase 2.6 m all match. Physics
constants (`ars.physics.KinematicBicycleParams`) are tuned to real F1
figures: top speed 95 m/s (~342 km/h), max acceleration ~1.2g, max braking
~5g, **max lateral (cornering) grip ~4.5g**. That last one matters: a
kinematic bicycle has no tire model, so without an explicit cap the car
could corner at any speed with zero slip -- e.g. taking the track's 25m-radius
turns at top speed would imply ~14.7g lateral force, well beyond what real
tires (or a real car) can generate. `KinematicBicyclePhysics.step()` now
caps speed so lateral acceleration (`v^2 * curvature`, from the vehicle's
own steer angle) never exceeds `max_lateral_accel` -- a stand-in for what
tire slip will eventually enforce for real. Verified: an agent driving at
the limit tops out at exactly 4.5g through the tightest turns, never above.

## What's a stub vs. real

- `ars.physics.KinematicBicyclePhysics` is a placeholder (bicycle model, no
  tire slip, no load transfer, no aero -- lateral grip is a single scalar
  cap, not a real tire curve) so the rest of the stack has something to run
  against on day one. **The real target is a 4-wheel F1-style vehicle**
  (independent wheel loads/slip, Pacejka-style tire model, aero) -- this is
  Module A's real work, still open. **See [PHYSICS.md](PHYSICS.md)** for
  the full handoff: what exactly the stub does and doesn't simulate, the
  interface you need to satisfy, the open `VehicleState` design question,
  and a suggested incremental path.
- `ars.track.make_simple_oval()` is a genuine v1 track (two straights, two
  same-direction 180-degree turns) -- not a stub, but intentionally simple.
  A closed loop's turns must sum to a full 2*pi in signed curvature, so this
  two-turn oval only ever turns one direction; a track with a real mixed
  left+right turn sequence needs either careful multi-arc closure math or a
  waypoint/spline-based `Track` implementation -- left for v1.1/backlog.
