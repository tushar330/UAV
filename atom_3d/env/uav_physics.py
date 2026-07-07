"""
UAV Physics Simulator — 2D and 3D models.


"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


class RotaryWingPower:
    """Rotary-wing propulsion power-speed curve P(V) (Zeng & Zhang 2019).

    P(V) = P0(1 + 3V^2/U_tip^2)
         + Pi(sqrt(1 + V^4/(4 v0^4)) - V^2/(2 v0^2))^(1/2)
         + 0.5 d0 rho s A V^3

    P(0) = P0 + Pi is the hover power — the *peak* of the curve. Replacing the
    old constant ``P_flight`` with this makes hovering correctly the most
    expensive state, which is what gives the dive-to-serve altitude decision a
    real interior optimum (PROBLEM_FORMULATION §5a).
    """

    def __init__(self, P0=79.86, Pi=88.63, U_tip=120.0, v0=4.03,
                 d0=0.6, rho=1.225, s=0.05, A=0.503):
        self.P0, self.Pi, self.U_tip, self.v0 = P0, Pi, U_tip, v0
        self.d0, self.rho, self.s, self.A = d0, rho, s, A
        self.P_hover = P0 + Pi
        # cache the energy-optimal (max-range) cruise speed and its J/m cost
        self._v_star, self._j_per_m = self._solve_optimal_speed()

    def power(self, V):
        """Instantaneous propulsion power at forward speed V (m/s)."""
        V = np.maximum(np.asarray(V, dtype=float), 0.0)
        blade = self.P0 * (1.0 + 3.0 * V ** 2 / self.U_tip ** 2)
        induced = self.Pi * np.sqrt(
            np.sqrt(1.0 + V ** 4 / (4.0 * self.v0 ** 4)) - V ** 2 / (2.0 * self.v0 ** 2)
        )
        parasite = 0.5 * self.d0 * self.rho * self.s * self.A * V ** 3
        return blade + induced + parasite

    def _solve_optimal_speed(self, v_lo=1.0, v_hi=60.0, n=600):
        """V* = argmin_V P(V)/V (max-range speed) — energy per metre travelled."""
        grid = np.linspace(v_lo, v_hi, n)
        jpm = self.power(grid) / grid
        i = int(np.argmin(jpm))
        return float(grid[i]), float(jpm[i])

    @property
    def optimal_speed(self) -> float:
        return self._v_star

    def energy_per_metre(self) -> float:
        """min_V P(V)/V — horizontal flight energy cost per metre (J/m)."""
        return self._j_per_m

    def leg_energy(self, horizontal_distance: float) -> float:
        """Energy to cruise a horizontal leg at the energy-optimal speed."""
        return self._j_per_m * max(horizontal_distance, 0.0)


@dataclass
class FlightSegment:
    start_pos: np.ndarray
    end_pos: np.ndarray
    start_altitude: float
    end_altitude: float
    distance_3d: float
    flight_time: float
    energy_consumed: float
    max_speed_used: float
    feasible: bool
    vert_energy: float = 0.0   # gravitational climb + small descent term (NOVEL, 3D only)


@dataclass
class TrajectoryResult:
    segments: List[FlightSegment] = field(default_factory=list)
    total_distance: float = 0.0
    total_flight_time: float = 0.0
    total_flight_energy: float = 0.0
    total_hover_energy: float = 0.0
    total_vert_energy: float = 0.0   # NOVEL altitude-change energy (Σ over legs)
    total_energy: float = 0.0
    data_collected: float = 0.0
    nodes_visited: List[int] = field(default_factory=list)
    feasible: bool = True


class UavPhysics2D:
    """
    2D UAV physics (Paper A baseline). Fixed altitude, fixed speed.

    Args:
        fly_speed:              Fixed speed (m/s).
        P_flight:               Propulsion power (W).
        P_hover:                Hover power (W).
        flight_height:          Fixed altitude (m).
        battery_capacity_mah:   mAh.
        battery_voltage:        V.
    """

    def __init__(
        self,
        fly_speed: float = 10.0,
        P_flight: float = 75.0,
        P_hover: float = 50.0,
        flight_height: float = 20.0,
        battery_capacity_mah: float = 2000.0,
        battery_voltage: float = 22.2,
    ):
        self.fly_speed      = fly_speed
        self.P_flight       = P_flight
        self.P_hover        = P_hover
        self.flight_height  = flight_height
        self.battery_capacity_j = battery_capacity_mah * battery_voltage * 3.6  # J

    def compute_flight_energy(self, distance: float) -> float:
        if distance < 1e-6:
            return 0.0
        return self.P_flight * (distance / self.fly_speed)

    def compute_flight_time(self, distance: float) -> float:
        if distance < 1e-6:
            return 0.0
        return distance / self.fly_speed

    def simulate_trajectory(
        self,
        waypoints: np.ndarray,
        depot: np.ndarray,
        hover_times: Optional[np.ndarray] = None,
    ) -> TrajectoryResult:
        """
        Simulate 2D trajectory.

        Args:
            waypoints:   (K, 2) node positions.
            depot:       (2,) depot position.
            hover_times: (K,) hover time per node (s). If None, hover energy = 0.
        """
        result      = TrajectoryResult()
        current_pos = depot.copy()

        for i in range(len(waypoints)):
            dist        = float(np.linalg.norm(current_pos[:2] - waypoints[i][:2]))
            flight_time = self.compute_flight_time(dist)
            energy      = self.compute_flight_energy(dist)

            seg = FlightSegment(
                start_pos=current_pos.copy(),
                end_pos=waypoints[i].copy(),
                start_altitude=self.flight_height,
                end_altitude=self.flight_height,
                distance_3d=dist,
                flight_time=flight_time,
                energy_consumed=energy,
                max_speed_used=self.fly_speed if dist > 0 else 0.0,
                feasible=True,
            )
            result.segments.append(seg)
            result.total_distance     += dist
            result.total_flight_time  += flight_time
            result.total_flight_energy += energy

            # Accumulate hover energy
            if hover_times is not None and i < len(hover_times):
                result.total_hover_energy += self.P_hover * float(hover_times[i])

            current_pos = waypoints[i].copy()

        # Return to depot
        ret_dist = float(np.linalg.norm(current_pos[:2] - depot[:2]))
        ret_time = self.compute_flight_time(ret_dist)
        ret_en   = self.compute_flight_energy(ret_dist)

        result.segments.append(FlightSegment(
            start_pos=current_pos.copy(), end_pos=depot.copy(),
            start_altitude=self.flight_height, end_altitude=self.flight_height,
            distance_3d=ret_dist, flight_time=ret_time,
            energy_consumed=ret_en,
            max_speed_used=self.fly_speed if ret_dist > 0 else 0.0,
            feasible=True,
        ))
        result.total_distance      += ret_dist
        result.total_flight_time   += ret_time
        result.total_flight_energy += ret_en

        # total_energy now includes hover energy
        result.total_energy = result.total_flight_energy + result.total_hover_energy
        result.feasible     = (result.total_energy <= self.battery_capacity_j)

        return result


class UavPhysics3D:
    """
    3D UAV physics with variable speed and altitude (Paper B mobility model).

    Enforces all Paper B mobility constraints:
        l_u(q+1) = l_u(q) + v_xy(q)·Δt
        H_u(q+1) = H_u(q) + v_z(q)·Δt
        ||v_xy|| ≤ v_xy_max,  |v_z| ≤ v_z_max
        ||Δv_xy|| ≤ a_xy_max·Δt,  |Δv_z| ≤ a_z_max·Δt
        H_min ≤ H_u ≤ H_max


    """

    def __init__(
        self,
        v_xy_max: float = 40.0,
        v_z_max: float = 20.0,
        a_xy_max: float = 5.0,
        a_z_max: float = 5.0,
        H_min: float = 20.0,
        H_max: float = 150.0,
        delta_t: float = 0.5,
        P_flight: float = 75.0,
        P_hover: float = 50.0,
        battery_capacity_mah: float = 2000.0,
        battery_voltage: float = 22.2,
        mass_kg: float = 2.0,
        motor_efficiency: float = 0.5,
        descent_coeff_cd: float = 4.0,
        gravity: float = 9.81,
        power_model: Optional["RotaryWingPower"] = None,
    ):
        self.v_xy_max  = v_xy_max
        self.v_z_max   = v_z_max
        self.a_xy_max  = a_xy_max
        self.a_z_max   = a_z_max
        self.H_min     = H_min
        self.H_max     = H_max
        self.delta_t   = delta_t
        # Rotary-wing power-speed model (Zeng-Zhang). When supplied, horizontal
        # legs cost energy_per_metre (energy-optimal cruise) and hover power is
        # the curve's peak P0+Pi — superseding the constant P_flight/P_hover.
        self.power_model = power_model
        self.P_flight  = P_flight
        self.P_hover   = power_model.P_hover if power_model is not None else P_hover
        self.battery_capacity_j = battery_capacity_mah * battery_voltage * 3.6
        # --- NOVEL altitude-change energy coefficients (spec §5) ---
        # climbing 1 m costs m*g/eta J of gravitational work; descending 1 m
        # dissipates a small c_d J (energy cannot be recovered, but is cheap).
        self.mass_kg          = mass_kg
        self.motor_efficiency = motor_efficiency
        self.descent_coeff_cd = descent_coeff_cd
        self.gravity          = gravity
        self.climb_coeff      = mass_kg * gravity / max(motor_efficiency, 1e-6)  # J per metre climbed

    def vertical_energy(self, delta_h: float) -> float:
        """E_vert for a single leg: (m*g/eta)*max(ΔH,0) + c_d*max(-ΔH,0).

        Climbing pays gravitational work; descending pays a small dissipation.
        """
        climb   = max(delta_h, 0.0)
        descent = max(-delta_h, 0.0)
        return self.climb_coeff * climb + self.descent_coeff_cd * descent

    def _plan_1d_motion(self, displacement: float, v_max: float, a_max: float) -> float:
        """
        Compute minimum travel time for a 1D displacement under
        trapezoidal velocity profile constraints.
        """
        dist = abs(displacement)
        if dist < 1e-6:
            return 0.0

        t_accel  = v_max / a_max
        d_accel  = v_max ** 2 / a_max   # total accel+decel distance

        if dist <= d_accel:
            return 2.0 * np.sqrt(dist / a_max)
        else:
            return 2.0 * t_accel + (dist - d_accel) / v_max

    def simulate_leg(
        self,
        start_pos: np.ndarray,
        start_alt: float,
        end_pos: np.ndarray,
        end_alt: float,
    ) -> FlightSegment:
        """Simulate one flight leg with 3D physics constraints."""
        h_disp = end_pos[:2] - start_pos[:2]
        h_dist = float(np.linalg.norm(h_disp))
        v_disp = end_alt - start_alt

        h_time = self._plan_1d_motion(h_dist, self.v_xy_max, self.a_xy_max)
        v_time = self._plan_1d_motion(v_disp, self.v_z_max,  self.a_z_max)

        total_time  = max(h_time, v_time, self.delta_t)
        dist_3d     = float(np.sqrt(h_dist ** 2 + v_disp ** 2))
        # Horizontal flight energy: power-speed curve at the energy-optimal cruise
        # speed if a power model is set, else the legacy constant-power estimate.
        if self.power_model is not None:
            energy = self.power_model.leg_energy(h_dist)
        else:
            energy = self.P_flight * total_time
        clamped_alt = float(np.clip(end_alt, self.H_min, self.H_max))
        feasible    = (abs(clamped_alt - end_alt) < 0.1)
        # NOVEL: gravitational climb / descent energy for the actual altitude change.
        vert_e      = self.vertical_energy(clamped_alt - start_alt)

        return FlightSegment(
            start_pos=start_pos.copy(),
            end_pos=end_pos.copy(),
            start_altitude=start_alt,
            end_altitude=clamped_alt,
            distance_3d=dist_3d,
            flight_time=total_time,
            energy_consumed=energy,
            max_speed_used=dist_3d / total_time if total_time > 0 else 0.0,
            feasible=feasible,
            vert_energy=vert_e,
        )

    def simulate_trajectory(
        self,
        waypoints: np.ndarray,
        altitudes: np.ndarray,
        depot: np.ndarray,
        depot_altitude: float,
        hover_times: Optional[np.ndarray] = None,
    ) -> TrajectoryResult:
        """
        Simulate full 3D trajectory.

        Args:
            waypoints:      (K, 2) horizontal positions.
            altitudes:      (K,)   UAV altitudes at each waypoint.
            depot:          (2,)   depot position.
            depot_altitude: Depot altitude (m).
            hover_times:    (K,)   hover time per node (s).
        """
        result      = TrajectoryResult()
        current_pos = depot.copy()
        current_alt = depot_altitude

        for i in range(len(waypoints)):
            target_alt = float(np.clip(altitudes[i], self.H_min, self.H_max))
            seg = self.simulate_leg(current_pos, current_alt, waypoints[i], target_alt)

            result.segments.append(seg)
            result.total_distance      += seg.distance_3d
            result.total_flight_time   += seg.flight_time
            result.total_flight_energy += seg.energy_consumed
            result.total_vert_energy   += seg.vert_energy
            result.feasible             = result.feasible and seg.feasible

            # Accumulate hover energy
            if hover_times is not None and i < len(hover_times):
                result.total_hover_energy += self.P_hover * float(hover_times[i])

            current_pos = waypoints[i].copy()
            current_alt = target_alt

        # Return to depot
        seg = self.simulate_leg(current_pos, current_alt, depot, depot_altitude)
        result.segments.append(seg)
        result.total_distance      += seg.distance_3d
        result.total_flight_time   += seg.flight_time
        result.total_flight_energy += seg.energy_consumed
        result.total_vert_energy   += seg.vert_energy
        result.feasible             = result.feasible and seg.feasible

        # total_energy = horizontal flight + NOVEL vertical + hover
        result.total_energy = (result.total_flight_energy
                               + result.total_vert_energy
                               + result.total_hover_energy)
        result.feasible     = result.feasible and (result.total_energy <= self.battery_capacity_j)

        return result

    def compute_energy_for_route(
        self,
        route_positions: np.ndarray,
        route_altitudes: np.ndarray,
    ) -> float:
        """Quick route energy estimate using trapezoidal motion model."""
        total_energy = 0.0
        for i in range(len(route_positions) - 1):
            h_dist = float(np.linalg.norm(
                route_positions[i + 1, :2] - route_positions[i, :2]
            ))
            v_dist = abs(route_altitudes[i + 1] - route_altitudes[i])

            h_time = self._plan_1d_motion(h_dist, self.v_xy_max, self.a_xy_max)
            v_time = self._plan_1d_motion(v_dist, self.v_z_max,  self.a_z_max)
            leg_time = max(h_time, v_time, 0.0)

            total_energy += self.P_flight * leg_time
            total_energy += self.vertical_energy(
                route_altitudes[i + 1] - route_altitudes[i]
            )

        return total_energy
