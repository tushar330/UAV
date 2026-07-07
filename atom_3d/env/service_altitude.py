"""
Energy-optimal service-altitude allocation — the dive-to-serve sub-problem.

This is the headline novelty (PROBLEM_FORMULATION §2.1, §5a): the altitude analog
of Paper A's KKT *time* allocation. When a UAV services a node/cluster ``S`` from
a hover altitude ``H_s`` directly above it, the per-service energy

    E_serve(H_s) = P_H * ( T_E(H_s) + T_C(H_s) )         # hover/charge, grows ∝ (H_s - z)^2
                 + (c_climb + c_d) * (H_cruise - H_s)     # dive down + climb back, grows with depth

has a genuine interior minimum, because lowering ``H_s`` strengthens the link
(less WPT/collect hover time at the *expensive* hover power) but costs more climb
energy to return to cruise. The optimum

    H_s*  ≈  z + (c_climb + c_d) * eta_L * beta * P_T / (2 * P_H * E_S)

*decreases* (dives deeper) as the served energy demand ``E_S`` grows, so a field
of heterogeneous demands produces an altitude profile that varies stop-to-stop.

We expose both the closed-form estimate and an exact 1-D numerical minimum (which
also accounts for the data-collection term ``T_C`` and the coverage constraint
``H_s >= z + r_S / tan(theta)`` so the cluster stays inside the footprint).
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ServiceAltitudeAllocator:
    bandwidth: float          # B (Hz)
    sigma2: float             # noise power (W)
    beta: float               # path-loss reference at carrier freq
    P_T: float                # transmit power (W)
    eta_linear: float         # linear EH efficiency eta_L
    P_hover: float            # hover power P_H (W) — the expensive peak
    c_climb: float            # m*g/eta (J per metre climbed)
    c_descent: float          # c_d (J per metre descended)
    tan_theta: float          # tan(half-beamwidth) for the coverage radius
    H_min: float = 20.0
    H_max: float = 150.0
    h_safe: float = 10.0

    # ------------------------------------------------------------------
    def _lower_bound(self, z_S: float, r_S: float) -> float:
        """Smallest feasible H_s: clearance + altitude floor + footprint covers r_S."""
        return max(self.H_min, z_S + self.h_safe, z_S + r_S / max(self.tan_theta, 1e-6))

    def closed_form(self, z_S: float, E_S: float, r_S: float = 0.0) -> float:
        """Closed-form H_s* estimate (ignores the slowly-varying T_C term)."""
        if E_S <= 0:
            hs = self.H_max
        else:
            hs = z_S + (self.c_climb + self.c_descent) * self.eta_linear * self.beta * self.P_T \
                 / (2.0 * self.P_hover * E_S)
        return float(np.clip(hs, self._lower_bound(z_S, r_S), self.H_max))

    # ------------------------------------------------------------------
    def _serve_energy(self, H_s, z_S, E_S, D_bits, r_S, H_cruise):
        """E_serve(H_s) including the data-collection term, vectorised over H_s."""
        d = np.sqrt(r_S ** 2 + (H_s - z_S) ** 2)        # worst-case node in the cluster
        d = np.maximum(d, 1e-3)
        P_R = self.eta_linear * self.beta * self.P_T / d ** 2
        T_E = E_S / np.maximum(P_R, 1e-30)               # WPT time to deliver E_S
        snr = self.P_T * self.beta / (d ** 2 * self.sigma2)
        rate = self.bandwidth * np.log2(1.0 + snr)
        T_C = D_bits / np.maximum(rate, 1e-6)
        hover_e = self.P_hover * (T_E + T_C)
        vert_e = (self.c_climb + self.c_descent) * np.maximum(H_cruise - H_s, 0.0)
        return hover_e + vert_e, T_E, T_C

    def optimal(self, z_S: float, E_S: float, data_mb: float, r_S: float,
                H_cruise: float, n_grid: int = 200):
        """Exact 1-D minimum of E_serve over the feasible altitude band.

        Returns (H_s*, E_serve*, T_E, T_C) for the chosen altitude.
        """
        lo = self._lower_bound(z_S, r_S)
        hi = self.H_max
        if hi <= lo:
            hi = lo + 1e-3
        D_bits = max(data_mb, 0.0) * 8e6
        grid = np.linspace(lo, hi, n_grid)
        E, T_E, T_C = self._serve_energy(grid, z_S, E_S, D_bits, r_S, H_cruise)
        i = int(np.argmin(E))
        return float(grid[i]), float(E[i]), float(T_E[i]), float(T_C[i])
