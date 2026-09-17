# Physics & Vehicle Dynamics -- Handoff Notes

This is for whoever owns **Module A: Agent Physics** (vehicle dynamics).
It explains what's already built, exactly what's fake/placeholder about
it, and what to change (and where) to bring in real physics.

Read [README.md](README.md) first for the overall project layout. This
doc goes deep on just the physics side.

## TL;DR

- There's a working placeholder physics model (`KinematicBicyclePhysics`)
  that the whole rest of the project (env, sensors, rendering, dashboard,
  agents) is already wired against. Nothing else needs to change when you
  replace it, **as long as your replacement satisfies the same interface**
  (below).
- The placeholder is a **kinematic bicycle model**: no tire slip, no load
  transfer, no 4-wheel independence, no aero, mass isn't even used. It's
  tuned to real F1 reference numbers (top speed, braking, cornering grip)
  so it's *physically bounded* even though it's not *physically simulated*.
- Your job is to replace `ars/physics/kinematic_stub.py` with a real
  **4-wheel vehicle model** -- independent per-wheel slip/load, a
  Pacejka-style (or similar) tire curve, load transfer, and aero. This is
  the actual v1 target, not an optional stretch goal -- see "Suggested
  incremental path" below for how to get there without a giant first PR,
  but the destination is 4 wheels, not a 2-wheel bicycle model. There's an
  open design decision you need to make early (see "The VehicleState
  problem" below) before you get deep into implementation.
- **Localization sensors (GPS, IMU, lidar, track pose) are not yours** --
  those are for agent navigation/perception and are owned separately. Your
  sensor territory, if any, is vehicle-mechanics sensors (wheel speed,
  tire temp, etc.) -- see "Sensors: what's yours and what isn't" below.

## The one interface you must satisfy

Everything in the codebase depends on this Protocol
(`ars/core/interfaces.py`), never on the concrete class:

```python
class VehiclePhysics(Protocol):
    def reset(self, x: float, y: float, heading: float) -> VehicleState:
        """Return an initial at-rest VehicleState at the given pose."""
        ...

    def step(self, state: VehicleState, action: VehicleAction, dt: float) -> VehicleState:
        """Advance the vehicle by dt seconds under the given control input."""
        ...
```

That's it. Two methods. `VehicleAction` is:

```python
@dataclass
class VehicleAction:
    throttle: float  # 0..1
    brake: float      # 0..1
    steer: float       # -1..1 (left..right)
```

As long as your class has these two methods with these signatures, you can
drop it into `RacingEnv(physics=YourPhysics(), ...)` and everything else
(dashboard, viewer, sensors, agents, tests) keeps working unchanged. This
is by design -- see `ars/core/interfaces.py`'s module docstring for why
the whole codebase is built around these Protocol boundaries.

**Important:** `VehiclePhysics` implementations are expected to be
*pure/stateless* w.r.t. anything outside the returned `VehicleState` --
`step()` takes a state and returns a new one; it shouldn't mutate hidden
internal state that affects future calls (vehicle *parameters* like mass
or tire curve are fine as constructor config, since those don't change
step to step). This matters because `RacingEnv` calls `_probe_obs_dim()`
once at construction time using a throwaway `reset()` call before the
real episode starts -- if your physics has step-to-step memory that isn't
in `VehicleState`, that probe call could leave it in a weird state.

## What exists right now: `KinematicBicyclePhysics`

File: `ars/physics/kinematic_stub.py`

This is a **kinematic bicycle model** -- the simplest physically-plausible
vehicle abstraction that exists. Here's exactly what it does and doesn't
do:

**What it does:**
- Integrates position/heading from speed and steering angle each step.
- Clamps speed to a configurable max, with throttle-driven acceleration
  and brake-driven deceleration, plus simple speed-proportional drag.
- Rate-limits how fast the steering angle can change (`steer_rate`).
- **Caps cornering speed** so lateral acceleration (`v² × curvature`,
  where curvature comes from the current steer angle: `tan(steer_angle) /
  wheelbase`) never exceeds `max_lateral_accel`. This is a strict global
  scalar cap standing in for what a real tire model would naturally
  enforce -- without it, the car could corner at literally any speed with
  zero slip, which is physically impossible. This was a real bug we found
  and fixed (see the "What's actually missing" section below).

**What it does NOT do (this is the whole point -- it's a stub):**
- **No tire model.** No slip angle, no slip ratio, no grip curve
  (Pacejka or otherwise). `max_lateral_accel` is one flat number, not a
  function of load, slip, or tire compound.
- **No 4-wheel independence.** It's a bicycle model -- one "virtual"
  front wheel, one "virtual" rear wheel, no per-wheel state at all.
- **No load transfer.** Braking/accelerating/cornering don't shift
  weight between wheels, so there's no understeer/oversteer behavior,
  no grip changes under load.
- **`vy` (lateral velocity) is hardcoded to 0.** The car is geometrically
  incapable of sliding or drifting. It always points exactly where it's
  going.
- **Mass isn't used anywhere in the physics math.** The dashboard lets a
  user set `mass_kg` (`ars/dashboard/config.py`), but nothing currently
  reads that value into `KinematicBicyclePhysics` -- acceleration/braking
  are fixed m/s² constants, not force ÷ mass. This is a real gap you'll
  need to close.
- **No aero.** Drag is a token linear-in-speed term, not real drag
  (∝ v²) or downforce.

### Current tuning (real F1 reference values)

`KinematicBicycleParams` (same file):

```python
wheelbase: float = 2.6            # m
max_speed: float = 95.0            # m/s (~342 km/h, real F1 top speed)
max_accel: float = 12.0            # m/s^2 (~1.2g, from throttle=1)
max_decel: float = 5.0 * G         # m/s^2 (~5g, real F1 max braking)
drag_coeff: float = 0.02           # simple speed-proportional drag
max_steer_angle: float = radians(28)   # rad, at steer=+-1
steer_rate: float = radians(180)   # rad/s, max steer angle change
max_lateral_accel: float = 4.5 * G # m/s^2 (~4.5g, real F1 cornering grip)
```

These were deliberately checked against real F1 numbers (2024-era
regs/performance) -- see the "Scale / real-world units" section of the
main README for the full audit. Vehicle mass (798 kg) and dimensions
(4.5 × 2.0 m, wheelbase 2.6 m) also match real F1 and are consistent
across the dashboard, physics params, and renderer. **You should keep
this level of real-world grounding** in whatever you build -- pick
parameters that correspond to something real and say what they
correspond to in a comment, the same way this file does.

## What's actually missing (the bug we found and fixed)

Worth understanding since it'll matter for your tire model too: a
kinematic bicycle with *no* cap on cornering speed will happily let the
car corner at any speed with zero slip. We measured this concretely --
at the old (also placeholder, now-fixed) `max_speed=60 m/s`, taking the
track's 25m-radius turns would require **~14.7g** of lateral force, about
3× what real F1 tires generate (~4.5g). Nothing in the simulation
prevented an agent from doing this, meaning an RL agent training against
that setup could learn "cheats" that don't correspond to real driving.

**The fix we shipped** (a scalar cap on lateral acceleration) is a
stand-in. **A real tire model needs to enforce this properly**: grip
should be a function of normal load (which changes under load transfer),
slip angle/ratio, and tire compound -- not one constant number. When you
replace the stub, make sure whatever you build has *some* mechanism that
prevents physically-impossible cornering, the same way this cap does, but
grounded in your actual tire math instead of a flat ceiling.

## The `VehicleState` problem -- read this before you start

This is the one open design decision that affects the rest of the
codebase, so it's worth understanding before you write code.

`VehicleState` (`ars/core/types.py`) is the shared type every non-physics
module reads:

```python
@dataclass
class VehicleState:
    x: float          # m, world frame
    y: float          # m, world frame
    heading: float     # rad, 0 = +x axis, CCW positive
    vx: float          # m/s, body-frame longitudinal velocity
    vy: float          # m/s, body-frame lateral velocity
    yaw_rate: float     # rad/s
    steer_angle: float  # rad, current road-wheel steer angle
```

It's **chassis-level only** -- there's no room for 4 independent wheels
(per-wheel slip angle, slip ratio, load, wheel speed). A real 4-wheel
model needs somewhere to put that data, and `VehicleState` as it stands
doesn't have it.

**Decided direction** (from earlier discussion, not yet implemented):
your physics module should own its **own richer state type** --
something like:

```python
@dataclass
class WheelState:
    slip_angle: float
    slip_ratio: float
    load: float     # normal force, N
    omega: float     # wheel angular velocity, rad/s

@dataclass
class FourWheelVehicleState:
    x: float; y: float; heading: float
    vx: float; vy: float; yaw_rate: float
    wheels: tuple[WheelState, WheelState, WheelState, WheelState]  # FL, FR, RL, RR
```

...rather than cramming wheel fields into the shared `VehicleState`. The
reasoning: `ars/env`, `ars/sensors`, and `ars/viz` only ever need
chassis-level fields (`x`, `y`, `heading`, `vx`, `vy`, `yaw_rate`) -- they
never need per-wheel detail. Making them all depend on a bloated shared
type just so physics can carry wheel data would leak your internal
representation into every other module, breaking the decoupling the whole
architecture is built around.

**What's NOT yet built, and is your responsibility to set up** (this
was flagged as an open item, not resolved): every consumer of vehicle
state needs a stable way to read `x`/`y`/`heading`/`vx`/`vy`/`yaw_rate`
off *whatever type your physics returns*, without importing your
physics-internal types. The current codebase reads `VehicleState`'s
fields directly (`state.x`, `state.vx`, etc.) in these places:

- `ars/env/racing_env.py` -- `state.x`, `state.y` (track queries)
- `ars/sensors/pose_sensor.py` -- `state.x`, `state.y`, `state.heading`
- `ars/sensors/proprioceptive_sensor.py` -- `state.vx`, `state.vy`, `state.yaw_rate`, `state.steer_angle`
- `ars/sensors/gps_sensor.py` -- `state.x`, `state.y`
- `ars/sensors/imu_sensor.py` -- `state.vx`, `state.vy`, `state.yaw_rate`
- `ars/sensors/lidar_sensor.py` -- `state.x`, `state.y`, `state.heading`
- `ars/viz/live_viewer.py` -- `state.x`, `state.y`, `state.heading`
- `ars/agents/dummy_expert.py` -- `state.x`, `state.y`, `state.heading`
- `ars/dashboard/steps/step5_simulation.py` -- `state.vx`

If your `FourWheelVehicleState` (or whatever you call it) just happens to
*also* have `x`, `y`, `heading`, `vx`, `vy`, `yaw_rate` as plain
attributes (which it naturally will, since those are still meaningful
chassis-level quantities for a 4-wheel car), **all of the above keeps
working with zero changes**, because Python doesn't care about your
type's name or module, only that the attributes exist (duck typing).
`steer_angle` isn't strictly required by everything, but
`ProprioceptiveSensor` reads it, so keep it too (or update that one
sensor if you drop it in favor of 4 independent wheel angles).

The cleanest way to formalize this (not yet done, worth doing when you
start): add a small `ChassisState` `Protocol` to `ars/core/interfaces.py`
with just those 6-7 fields, and have `ars/env`, `ars/sensors`, `ars/viz`
type-hint against that instead of the concrete `VehicleState`. That's a
~10-line addition whenever you're ready for it -- ask the software side
to help if you'd rather not touch `ars/core/`.

## Where to actually write your code

1. Create a new file, e.g. `ars/physics/four_wheel.py` (or `dynamic_bicycle.py`
   if you're doing an intermediate step first -- see "Suggested incremental
   path" below).
2. Define your state type (if different from `VehicleState`) and your
   params dataclass (follow `KinematicBicycleParams`'s pattern: every
   field gets a unit comment and, where possible, a real-world reference
   value in the comment).
3. Implement a class with `reset(x, y, heading)` and
   `step(state, action, dt)` matching the `VehiclePhysics` Protocol.
4. Swap it into `ars/env/factory.py`'s `make_default_env()` and
   `ars/dashboard/builder.py`'s `build_env_and_agent()` (both currently
   hardcode `KinematicBicyclePhysics()` -- just change that one line in
   each) to actually run it end-to-end.
5. Write tests in `tests/physics/` following the existing
   `test_kinematic_stub.py` pattern -- particularly: reset is at rest,
   full throttle accelerates, brake decelerates, speed never exceeds max,
   **and some physically-grounded check that your tire model actually
   limits grip** (the equivalent of our
   `test_lateral_acceleration_never_exceeds_grip_cap` regression test).

You do **not** need to touch `ars/env/`, `ars/sensors/`, `ars/viz/`, or
`ars/dashboard/` for the swap itself to work -- only if you're also
resolving the `ChassisState` question above, or adding a wheel-speed/
odometry sensor (see below).

## Sensors: what's yours and what isn't

Current sensor set (`ars/sensors/`), all **ideal/noiseless** for v1:

- `TrackPoseSensor` -- lateral offset, heading error, curvature, arc-length (needs `Track`, not physics-specific)
- `ProprioceptiveSensor` -- `vx`, `vy`, `yaw_rate`, `steer_angle` straight off `VehicleState`
- `LidarSensor` -- single forward ray, distance to track boundary (needs `Track`)
- `GpsSensor` -- world position converted to real lat/lon (needs `state.x`, `state.y` only)
- `ImuSensor` -- accelerometer (`ax`, `ay`, derived by differencing consecutive velocities) + gyroscope (`yaw_rate` straight off `VehicleState`)

**Localization/navigation sensors (GPS, IMU, lidar, track pose) are NOT
your territory.** These exist so the *agent* can perceive where it is and
navigate -- that's owned by whoever's doing sensors for agent
intelligence, not physics. Don't add to, modify, or extend this group
(no noise model on GPS/IMU, no new lidar configurations, etc.) unless
that's explicitly handed to you separately.

**What IS yours**: sensors that measure the vehicle's own mechanical
state -- the stuff a real car's onboard systems measure about itself,
not about the world around it. The clearest example, and explicitly
**NOT yet built**: **wheel-speed/odometry sensing** (per-wheel angular
velocity, `omega` in the `WheelState` sketch above) -- this only makes
sense once your 4-wheel model exists, and it's naturally yours since it's
reading data your own physics model produces. Other things in this same
category if/when they become relevant to your model: tire
temperature/pressure sensors, brake temperature, suspension travel --
anything measuring the car's mechanical state rather than its position
or surroundings.

If you do add a sensor like this, follow the existing `Sensor` Protocol
pattern (see `ars/sensors/imu_sensor.py` for the closest existing
example, since it's the other sensor with memory across steps and needs
`reset()`) -- but keep it scoped to vehicle-mechanics data, not
navigation/localization data.

## Suggested incremental path

The target is a full 4-wheel model -- that's the v1 deliverable, not
optional. But you don't have to write it in one giant PR. A reasonable
path that keeps something always-running and testable at every step,
landing on 4 wheels rather than stopping short of it:

1. **Dynamic bicycle model** -- keep the 2-wheel abstraction *temporarily*,
   but replace the kinematic assumption (`vy = 0`, no slip) with an actual
   slip-angle/grip-curve relationship. This still fits in `VehicleState`
   as-is (no `ChassisState` work needed yet), and is a much smaller first
   step from the current stub than jumping straight to 4 wheels. Treat
   this as scaffolding you'll replace, not a resting point.
2. **Add load transfer** -- longitudinal (accel/braking) and lateral
   (cornering) weight transfer changing each "wheel's" (front/rear, in
   the bicycle model) effective grip.
3. **Go to 4 wheels** -- this is the actual target, and where you need the
   `VehicleState` / `ChassisState` decision above. Independent per-wheel
   slip angle, slip ratio, load, and grip.
4. **Add aero** (drag ∝ v², downforce ∝ v² affecting grip) whenever it
   matters for your research goals.

At every stage, you should be able to run:

```bash
.venv\Scripts\python scripts\dashboard.py     # Windows
.venv/bin/python scripts/dashboard.py          # macOS/Linux
```

...swap in your physics in Step 3 area (once that screen becomes a real
picker instead of a placeholder -- ask the software side, or do it
yourself, it's a small dashboard change), hit Run on Step 5, and watch
the car behave differently. If you don't want to wait for the dashboard
picker, the fastest loop is editing `ars/env/factory.py`'s one line and
running `scripts/drive.py` to drive it yourself with the keyboard.

## Questions this doc doesn't answer

If something about the interface contract itself needs to change (not
just what's behind it) -- e.g. `VehiclePhysics.step()` needs an extra
argument, or `VehicleAction` needs a new field -- that's a change to
`ars/core/`, which is explicitly software's owned territory per the
module ownership table in the main README. Flag it rather than changing
`ars/core/interfaces.py` or `ars/core/types.py` unilaterally, since
everything else in the repo depends on those staying stable.
