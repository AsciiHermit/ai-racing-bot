"""Simple four-wheel vehicle-dynamics model for Version 1 of the simulator.

WHAT THIS VERSION DOES
----------------------
- Models four separate tires: FL, FR, RL and RR.
- Uses a simple RWD EV model with a fixed maximum motor torque.
- Converts motor torque to drive force with: F = T/r.
- Calculates drag and rolling resistance.
- Calculates longitudinal acceleration from: F_long = F_drive - F_brake - F_drag - F_rolling
- Calculates lateral acceleration from the tire lateral forces.
- Updates Vx and Vy from ax and ay.
- Estimates longitudinal and lateral weight transfer, during acceleration, braking and cornering which varies the normal load on each tire.
- Uses the resulting normal load to calculate each tire's available grip: F_max = mu * N
- Models lateral tire slip with slip angle.
- Models the longitudinal tire slip using the slip ratio.
- Uses the traction circle so a tire cannot create unlimited combined force.
- Uses Ackermann steering for the front wheels.
- Calculates yaw moment and yaw acceleration so the car can actually turn.
- Default values considered to represent current f1 cars specifications.
- This is intentionally a Version-1 model: simple structure, but the
  important vehicle-dynamics equations are kept physically meaningful.

WHAT THIS VERSION DOES NOT DO
-----------------------------
- No gearbox, motor RPM, motor power curve or drivetrain efficiency.
- No suspension, pitch, roll or suspension geometry.
- No Pacejka/Magic Formula tire model.
- No tire temperature, pressure, wear or camber model.
- No ABS or traction-control system.
- No aerodynamic downforce.

MAIN PHYSICS EQUATIONS
----------------------
1. Motor drive force:
      F_drive = T_motor / r

2. Aerodynamic drag:
      F_drag = 0.5 * rho * Cd * A * V^2

3. Rolling resistance:
      F_rr = Crr * m * g

4. Longitudinal acceleration:
      F_long = F_drive - F_brake - F_drag - F_rr
      ax = F_long / m

5. Lateral acceleration in the body frame:
      ay = sum(Fy) / m - Vx * yaw_rate

6. Longitudinal weight transfer:
      DeltaN_long = m * ax * h / L

7. Lateral weight transfer:
      DeltaN_lat = m * ay * h / T

8. Tire grip:
      F_max = mu * N

9. Slip angle:
      alpha = -atan2(Vy_wheel, Vx_wheel)

10. Linear lateral tire force:
      Fy = C_alpha * alpha

11. Longitudinal slip model:
      Slip ratio = (V - w * R)/V
      where V is the longitudinal velocity of the vehicle, w is the wheel angular velocity and R is the tire radius.

12. Traction circle:
      sqrt(Fx^2 + Fy^2) <= mu * N

13. Yaw moment:
      Mz = sum(x_i * Fy_i - y_i * Fx_i)

14. Yaw acceleration:
      yaw_accel = Mz / Iz

15. Ackermann steering:
      tan(delta_inner) = L / (R - T/2)
      tan(delta_outer) = L / (R + T/2)
      where R is the turn radius to the vehicle centreline and T is the vehicle track width.

16. Turn radius:
      R = L/sin(theta)
      where theta is the steer angle.
"""

from __future__ import annotations  # Allows modern type hints to refer to classes defined later.
from dataclasses import dataclass  # Provides simple containers for vehicle and tire data.
import math  # Provides trigonometric functions and other mathematical functions.
from ars.core.types import VehicleAction  # Uses the existing simulator throttle/brake/steer input type.

G = 9.81  # Earth's gravitational acceleration in m/s^2.
EPSILON = 1.0e-8  # Small number used to avoid division by zero.


def clamp(value: float, low: float, high: float) -> float:  # Restricts a value to a chosen range.
    return max(low, min(value, high))  # Returns the lower limit, value, or upper limit.


@dataclass(frozen=True)  # Makes the parameter object mostly read-only after creation.
class FourWheelParams:  # Stores the physical parameters of the car.
    mass_kg: float = 770.0  # Vehicle mass in kg.
    wheelbase: float = 3.60  # Distance between front and rear axle centres in m.
    track_width: float = 1.60  # Distance between left and right wheels in m.
    weight_dist_front: float = 0.45  # Fraction of static vehicle weight carried by the front axle.
    cog_height: float = 0.25  # Height of the centre of gravity above the road in m.
    yaw_inertia: float = 1800.0  # Approximate yaw moment of inertia about the vertical axis in kg*m^2.
    tire_radius: float = 0.35  # Effective tire rolling radius in m.
    tire_mu: float = 1.50  # Tire-road friction coefficient.
    cornering_stiffness: float = 65000.0  # Approximate lateral tire stiffness C_alpha in N/rad for one tire.
    max_steer_angle: float = math.radians(30.0)  # Maximum steering angle in radians.
    steer_rate: float = math.radians(180.0)  # Maximum steering-angle change per second in rad/s.
    max_motor_torque: float = 7000.0  # Maximum motor torque in Nm.
    max_brake_force: float = 5000.0  # Maximum braking force in N.
    air_density: float = 1.225  # Air density in kg/m^3.
    drag_coeff: float = 0.90  # Aerodynamic drag coefficient of the vehicle Cd.
    frontal_area: float = 1.50  # Frontal area of the vehicle in m^2.
    C_rr: float = 0.015  # Rolling resistance coefficient Crr.
    low_speed_cutoff: float = 0.5  # Speed below which slip-angle forces are reduced.

@dataclass  # Automatically creates a constructor for each tire state.
class WheelState:  # Stores the important calculated quantities for one tire.
    slip_angle: float = 0.0  # Lateral slip angle alpha in radians.
    slip_ratio: float = 0.0  # Simple longitudinal slip estimate kappa; positive means driving slip.
    normal_load: float = 0.0  # Vertical tire load N.
    longitudinal_force: float = 0.0  # Final longitudinal tire force in N.
    lateral_force: float = 0.0  # Final lateral tire force in N.
    grip_usage: float = 0.0  # Fraction of the available traction circle being used.

@dataclass  # Automatically creates a constructor for the vehicle state.
class FourWheelVehicleState:  # Stores vehicle position, velocity, rotation and tire states.
    x: float  # World X position in m.
    y: float  # World Y position in m.
    heading: float  # Vehicle heading in radians.
    vx: float  # Longitudinal body-frame velocity in m/s.
    vy: float  # Lateral body-frame velocity in m/s.
    yaw_rate: float  # Yaw rate in rad/s.
    steer_angle: float  # Current steering angle in radians.
    wheels: tuple[WheelState, WheelState, WheelState, WheelState]  # FL, FR, RL, RR.

class FourWheelPhysics:  # Calculates the simplified four-wheel vehicle dynamics.
    _FRONT = (True, True, False, False)  # Identifies FL/FR as front wheels.
    _LEFT = (True, False, True, False)  # Identifies FL/RL as left-side wheels.

    def __init__(self, params: FourWheelParams | None = None):  # Creates the physics model.
        self.params = params if params is not None else FourWheelParams()  # Uses custom parameters or defaults.

    def reset(self, x: float, y: float, heading: float) -> FourWheelVehicleState:  # Creates the initial stationary state.
        p = self.params  # Stores a short reference to the parameters.
        front_each = p.mass_kg * G * p.weight_dist_front / 2.0  # Static normal load on each front tire.
        rear_each = p.mass_kg * G * (1.0 - p.weight_dist_front) / 2.0  # Static normal load on each rear tire.
        loads = (front_each, front_each, rear_each, rear_each)  # Places the four loads in FL, FR, RL, RR order.
        wheels = tuple(WheelState(normal_load=loads[i]) for i in range(4))  # Creates the four initial tire states.
        return FourWheelVehicleState(x, y, heading, 0.0, 0.0, 0.0, 0.0, wheels)  # Returns the stationary car.

    def _front_wheel_angles(self, steer_angle: float) -> tuple[float, float]:  # Calculates Ackermann steering angles.
        p = self.params  # Stores a short reference to the parameters.
        if abs(steer_angle) < EPSILON:  # Checks whether steering is effectively zero.
            return 0.0, 0.0  # Both front wheels point straight ahead.
        radius = p.wheelbase / abs(math.sin(steer_angle))  # Estimates centreline turn radius from tan(delta)=L/R.
        half_track = p.track_width / 2.0  # Calculates half the track width.
        inner_radius = radius - half_track  # Calculates the inside wheel's approximate turn radius.
        outer_radius = radius + half_track  # Calculates the outside wheel's approximate turn radius.
        inner_angle = math.atan(p.wheelbase / inner_radius)  # Calculates the inside Ackermann steering angle.
        outer_angle = math.atan(p.wheelbase / outer_radius)  # Calculates the outside Ackermann steering angle.
        direction = 1.0 if steer_angle > 0.0 else -1.0  # Preserves the sign of the steering input.
        if steer_angle > 0.0:  # Positive steering means the left wheel is inside.
            return direction * inner_angle, direction * outer_angle  # Returns FL inner and FR outer angles.
        return direction * outer_angle, direction * inner_angle  # Returns FL outer and FR inner angles.

    def step(self, state: FourWheelVehicleState, action: VehicleAction, dt: float) -> FourWheelVehicleState:  # Advances the simulation by one time step.
        p = self.params  # Stores a short reference to the parameters.
        if dt <= 0.0:  # Checks for an invalid time step.
            raise ValueError("dt must be greater than zero")  # Stops the simulation with a clear error.

        throttle = clamp(float(action.throttle), 0.0, 1.0)  # Limits throttle input to 0..1.
        brake = clamp(float(action.brake), 0.0, 1.0)  # Limits brake input to 0..1.
        steer_input = clamp(float(action.steer), -1.0, 1.0)  # Limits steering input to -1..1.

        steer_target = -steer_input * p.max_steer_angle  # Converts normalized steering input into the desired steering angle.
        max_steer_change = p.steer_rate * dt  # Calculates the largest steering change allowed this frame.
        steer = state.steer_angle + clamp(steer_target - state.steer_angle, -max_steer_change, max_steer_change)  # Applies the steering-rate limit.

        vx = state.vx  # Reads the current longitudinal velocity.
        vy = state.vy  # Reads the current lateral velocity.
        yaw_rate = state.yaw_rate  # Reads the current yaw rate.
        speed = math.hypot(vx, vy)  # Calculates total vehicle speed.

        front_left_steer, front_right_steer = self._front_wheel_angles(steer)  # Calculates separate front-wheel Ackermann angles.
        wheel_steers = (front_left_steer, front_right_steer, 0.0, 0.0)  # Rear wheels remain unsteered.

        requested_drive_force = throttle * p.max_motor_torque / p.tire_radius  # Converts max motor torque to total drive force using F=T/r.
        requested_brake_force = brake * p.max_brake_force # Calculates requested total braking force.

        drag_force = 0.5 * p.air_density * p.drag_coeff * p.frontal_area * speed**2  # Calculates aerodynamic drag.
        rolling_resistance = p.C_rr * p.mass_kg * G  # Calculates rolling resistance.

        front_distance = (1.0 - p.weight_dist_front) * p.wheelbase  # Calculates CG-to-front-axle distance a.
        rear_distance = p.weight_dist_front * p.wheelbase  # Calculates CG-to-rear-axle distance b.
        wheel_positions = ((front_distance, p.track_width / 2.0), (front_distance, -p.track_width / 2.0), (-rear_distance, p.track_width / 2.0), (-rear_distance, -p.track_width / 2.0))  # Stores FL, FR, RL, RR positions relative to the CG.

        # Use the requested road forces for the first load-transfer estimate.
        ax_estimate = (requested_drive_force - requested_brake_force - drag_force - rolling_resistance) / p.mass_kg  # Estimates longitudinal acceleration from F=ma.
        ay_estimate = vx * yaw_rate  # Estimates lateral acceleration from the vehicle's current turning motion.

        longitudinal_transfer = p.mass_kg * ax_estimate * p.cog_height / p.wheelbase  # Calculates front/rear load transfer from ax.
        lateral_transfer = p.mass_kg * ay_estimate * p.cog_height / p.track_width  # Calculates left/right load transfer from ay.

        front_static_each = p.mass_kg * G * p.weight_dist_front / 2.0  # Calculates static load on one front tire.
        rear_static_each = p.mass_kg * G * (1.0 - p.weight_dist_front) / 2.0  # Calculates static load on one rear tire.

        loads = []  # Creates a list for the four instantaneous tire loads.
        for i in range(4):  # Calculates FL, FR, RL and RR loads separately.
            base_load = front_static_each if self._FRONT[i] else rear_static_each  # Selects the correct static front/rear load.
            long_change = -longitudinal_transfer / 2.0 if self._FRONT[i] else longitudinal_transfer / 2.0  # Acceleration shifts load rearward; braking shifts it forward.
            lat_change = -lateral_transfer / 2.0 if self._LEFT[i] else lateral_transfer / 2.0  # Positive ay shifts load toward the right in this coordinate convention.
            loads.append(max(0.0, base_load + long_change + lat_change))  # Prevents an individual tire load from becoming negative.

        total_load = sum(loads)  # Adds all four tire loads.
        target_load = p.mass_kg * G  # Total vertical load on a flat road.
        if total_load > EPSILON:  # Checks that the load calculation produced usable values.
                correction = target_load / total_load  # Calculates a small correction for numerical clipping.
                loads = [load * correction for load in loads]  # Restores the total normal load to approximately m*g.

        wheel_fx = [0.0, 0.0, 0.0, 0.0]  # Stores each tire's final longitudinal force in body coordinates.
        wheel_fy = [0.0, 0.0, 0.0, 0.0]  # Stores each tire's final lateral force in body coordinates.
        wheel_states = []  # Stores the simplified tire states.

        # Distribute the requested drive force to the rear wheels according to their normal load.
        rear_load = loads[2] + loads[3]  # Adds the normal load carried by the two rear tires.
        drive_forces = [0.0, 0.0, 0.0, 0.0]  # Starts with zero drive force at every wheel.
        if rear_load > EPSILON:  # Checks that the rear tires are actually loaded.
                drive_forces[2] = requested_drive_force * loads[2] / rear_load  # Gives RL its load-proportional share of drive force.
                drive_forces[3] = requested_drive_force * loads[3] / rear_load  # Gives RR its load-proportional share of drive force.

        # Distribute requested braking according to normal load.
        brake_forces = [requested_brake_force * load / max(total_load, EPSILON) for load in loads]  # Shares braking force among all four tires.

        for i in range(4):  # Solves the tire model independently for each wheel.
                x_wheel, y_wheel = wheel_positions[i]  # Reads this wheel's position relative to the CG.
                wheel_angle = wheel_steers[i]  # Reads this wheel's steering angle.

                wheel_vx_body = vx - yaw_rate * y_wheel  # Calculates wheel longitudinal velocity including yaw motion.
                wheel_vy_body = vy + yaw_rate * x_wheel  # Calculates wheel lateral velocity including yaw motion.

                cos_a = math.cos(wheel_angle)  # Calculates cosine of wheel steering angle.
                sin_a = math.sin(wheel_angle)  # Calculates sine of wheel steering angle.
                wheel_vx = cos_a * wheel_vx_body + sin_a * wheel_vy_body  # Rotates velocity into the wheel's forward direction.
                wheel_vy = -sin_a * wheel_vx_body + cos_a * wheel_vy_body  # Rotates velocity into the wheel's sideways direction.

                if math.hypot(wheel_vx, wheel_vy) < p.low_speed_cutoff:  # Checks for very low wheel speed.
                    slip_angle = 0.0  # Avoids unstable lateral slip calculations near standstill.
                else:  # Runs the normal slip-angle calculation when the car is moving.
                    slip_angle = -math.atan2(wheel_vy, max(abs(wheel_vx), p.low_speed_cutoff))  # Calculates lateral tire slip angle alpha.

                lateral_force_request = p.cornering_stiffness * slip_angle  # Calculates lateral tire force from Fy=C_alpha*alpha.
                if speed < p.low_speed_cutoff:  # Checks whether the whole car is moving very slowly.
                    lateral_force_request *= speed / p.low_speed_cutoff  # Fades lateral force smoothly toward zero near standstill.

                longitudinal_force_request = drive_forces[i] - brake_forces[i]  # Calculates the requested tire longitudinal force.

                # A simple force-to-slip relationship provides a useful longitudinal slip state without wheel rotation.
                w_wheel = vx / p.tire_radius  # Calmulates angular velocity of the wheels.
                slip_ratio = (vx - w_wheel * p.tire_radius) / max(vx, EPSILON)  # Calculates longitudinal slip ratio.
                slip_ratio = clamp(slip_ratio, -1.0, 1.0)  # Keeps the simplified slip ratio within a practical range.

                max_tire_force = p.tire_mu * loads[i]  # Calculates this tire's total available grip F_max=mu*N.
                requested_magnitude = math.hypot(longitudinal_force_request, lateral_force_request)  # Calculates combined tire-force demand.

                if requested_magnitude > max_tire_force and requested_magnitude > EPSILON:  # Checks whether the tire exceeds its traction circle.
                    scale = max_tire_force / requested_magnitude  # Calculates the amount by which both force components must be reduced.
                    longitudinal_force = longitudinal_force_request * scale  # Applies the traction limit to longitudinal force.
                    lateral_force = lateral_force_request * scale  # Applies the same traction limit to lateral force.
                else:  # Runs when the tire is below its grip limit.
                    scale = 1.0  # No force reduction is required.
                    longitudinal_force = longitudinal_force_request  # Keeps the requested longitudinal force.
                    lateral_force = lateral_force_request  # Keeps the requested lateral force.

                wheel_fx[i] = longitudinal_force * cos_a - lateral_force * sin_a  # Rotates tire force into vehicle longitudinal coordinates.
                wheel_fy[i] = longitudinal_force * sin_a + lateral_force * cos_a  # Rotates tire force into vehicle lateral coordinates.

                grip_usage = math.hypot(longitudinal_force, lateral_force) / max(max_tire_force, EPSILON)  # Calculates how much of the tire's grip is being used.
                wheel_states.append(WheelState(slip_angle, slip_ratio, loads[i], longitudinal_force, lateral_force, clamp(grip_usage, 0.0, 1.0)))  # Stores the tire result.

        # Separate the final tire forces into actual drive and actual brake force.
        actual_drive_force = sum(max(wheel_states[i].longitudinal_force, 0.0) for i in range(4))  # Adds positive tire forces that drive the car forward.
        actual_brake_force = sum(max(-wheel_states[i].longitudinal_force, 0.0) for i in range(4))  # Adds negative tire forces that brake the car.
        ax_new = (actual_drive_force - actual_brake_force - drag_force - rolling_resistance) / p.mass_kg + vy * yaw_rate  # Calculates ax from F_drive-F_brake-F_drag-F_rr.
        ay_new = sum(wheel_fy) / p.mass_kg - vx * yaw_rate  # Calculates actual body-frame ay from tire lateral forces.

        ax_estimate = ax_new  # Uses the new acceleration for the next load-transfer iteration.
        ay_estimate = ay_new  # Uses the new lateral acceleration for the next load-transfer iteration.

        # Calculate yaw moment after the final tire-force solution.
        yaw_moment = 0.0  # Starts the total yaw moment at zero.
        for i in range(4):  # Adds the yaw moment from each tire.
            x_wheel, y_wheel = wheel_positions[i]  # Reads the tire position relative to the CG.
            yaw_moment += x_wheel * wheel_fy[i] - y_wheel * wheel_fx[i]  # Calculates Mz=x*Fy-y*Fx for this tire.

        actual_drive_force = sum(max(wheel_states[i].longitudinal_force, 0.0) for i in range(4))  # Final total drive force after tire-grip limits.
        actual_brake_force = sum(max(-wheel_states[i].longitudinal_force, 0.0) for i in range(4))  # Final total brake force after tire-grip limits.
        ax = (actual_drive_force - actual_brake_force - drag_force - rolling_resistance) / p.mass_kg + vy * yaw_rate  # Final ax from F_drive-F_brake-F_drag-F_rr.
        ay = sum(wheel_fy) / p.mass_kg - vx * yaw_rate  # Final lateral acceleration from tire lateral forces and body-frame rotation.
        yaw_accel = yaw_moment / max(p.yaw_inertia, EPSILON)  # Calculates yaw acceleration from Mz=Iz*alpha.

        new_vx = vx + ax * dt  # Updates longitudinal velocity using vx_new=vx+ax*dt.
        new_vy = vy + ay * dt  # Updates lateral velocity using vy_new=vy+ay*dt.
        new_yaw_rate = yaw_rate + yaw_accel * dt  # Updates yaw rate using r_new=r+alpha*dt.

        if vx >= 0.0 and new_vx < 0.0:  # Prevents the forward-only simulator from entering reverse.
            new_vx = 0.0  # Stops longitudinal velocity at zero.        

        new_heading = state.heading + new_yaw_rate * dt  # Updates vehicle heading from yaw rate.
        cos_h = math.cos(new_heading)  # Calculates heading cosine for world-coordinate conversion.
        sin_h = math.sin(new_heading)  # Calculates heading sine for world-coordinate conversion.
        world_vx = new_vx * cos_h - new_vy * sin_h  # Converts body velocity into world X velocity.
        world_vy = new_vx * sin_h + new_vy * cos_h  # Converts body velocity into world Y velocity.
        new_x = state.x + world_vx * dt  # Updates world X position.
        new_y = state.y + world_vy * dt  # Updates world Y position.

        return FourWheelVehicleState(new_x, new_y, new_heading, new_vx, new_vy, new_yaw_rate, steer, tuple(wheel_states))  # Returns the updated vehicle state.