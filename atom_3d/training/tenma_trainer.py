"""
TENMA trainer for ATOM-3D (and the 2D-AUTO baseline).

Implements the REINFORCE-with-critic-baseline loop of Paper A, extended so that

* the rollout uses the footprint decoder (one hover serves a whole coverage cone),
* multi-UAV segmentation is driven by cumulative storage ``C_max`` and battery
  ``E_max`` (spec §6),
* the per-hover charge+collect time comes from the Paper-A KKT closed form with
  full 3D distance, capped by ``tau_max``,
* the policy gradient includes **both** the anchor log-prob *and* the altitude
  log-prob, so the altitude head finally receives a reward gradient (spec §8).

The heavy lifting is in :meth:`TENMATrainer._partition_and_evaluate`, which turns
a :class:`DecodePlan` into per-instance total energy + reward.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import lambertw

from ..models import GraphEncoder, GNNEncoder, CriticNetwork, TrajectoryDecoder
from ..env.iot_env import IoTEnvironment2D, IoTEnvironment3D
from ..env.channel_model import ChannelModel3D
from ..env.uav_physics import UavPhysics2D, UavPhysics3D, RotaryWingPower
from ..env.service_altitude import ServiceAltitudeAllocator
from ..utils.routing import nn_2opt_tour
from .reward_function import RewardConfig, energy_reward, priority_penalty
from .cmdp import CMDPConfig, LagrangianDuals


# ----------------------------------------------------------------------
# Vectorised KKT time allocation (Paper A eqs. 17-18) over many node visits.
# ----------------------------------------------------------------------
def kkt_times(
    distances: np.ndarray,     # (K,) 3D distance UAV->node (m)
    demands_mb: np.ndarray,    # (K,) data demand (MB)
    *,
    bandwidth: float,
    sigma2: float,
    beta: float,
    P_T: float,
    eta_linear: float,
    tau_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T_collect, T_energy) arrays, each clipped so T_C+T_E <= tau_max."""
    K = len(distances)
    if K == 0:
        return np.zeros(0), np.zeros(0)

    d = np.maximum(distances, 1e-3)
    D_bits = np.maximum(demands_mb, 0.0) * 8e6
    g2 = beta / d ** 2
    phi = eta_linear * g2 * P_T                       # phi(P^R) = eta * |g|^2 * P_T
    phi = np.maximum(phi, 1e-30)

    lambert_arg = phi / (np.e * max(sigma2, 1e-30))
    W = np.real(lambertw(lambert_arg, k=0))
    W = np.maximum(W, -0.999)

    denom = bandwidth * (W + 1.0) / math.log(2.0)
    t_collect = np.where(np.abs(denom) < 1e-12, tau_max * 0.9, D_bits / denom)
    t_collect = np.clip(t_collect, 1e-6, tau_max * 0.9)

    rate_ratio = D_bits / (t_collect * bandwidth + 1e-12)
    power2 = np.power(2.0, np.minimum(rate_ratio, 40.0))
    t_energy = np.where(power2 > 1.0, t_collect / (power2 - 1.0) * sigma2 / phi, 0.0)
    t_energy = np.clip(t_energy, 0.0, np.maximum(tau_max - t_collect, 0.0))
    return t_collect, t_energy


@dataclass
class TrainConfig:
    mode: str = "3d"               # "3d" or "2d"
    encoder: str = "attention"     # "attention" or "gnn"
    embed_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    lr_actor: float = 1e-5
    lr_critic: float = 1e-3
    lr_decay: float = 0.98
    max_grad_norm: float = 1.0
    entropy_coef: float = 0.0
    freeze_altitude: Optional[float] = None   # 3D frozen-altitude ablation (spec §6)
    priority_feature: bool = False            # feed per-node priority as an extra encoder input
    cmdp: bool = False                        # primal-dual CMDP: per-class QoS dual multipliers
    device: str = "cpu"


class TENMATrainer:
    """Encoder + decoder + critic with the TENMA REINFORCE loop."""

    def __init__(self, params: dict, train_cfg: TrainConfig):
        self.params = params
        self.cfg = train_cfg
        self.is_3d = (train_cfg.mode.lower() == "3d")
        self.device = torch.device(train_cfg.device)

        sc = params["scenario"]; uav = params["uav"]; mob = params["mobility_3d"]
        cov = params["coverage"]; ch = params["channel"]; qos = params["qos"]

        # --- physical / channel constants (shared by env + KKT) ---
        c = 3e8
        self.beta = (c / (4 * math.pi * ch["carrier_frequency"])) ** 2
        self.sigma2 = 10 ** ((ch["noise_power_dbm"] - 30) / 10)
        self.bandwidth = ch["bandwidth"]
        # path-loss exponent alpha (|g|^2 = beta / d^alpha). Used everywhere SNR is
        # computed inline so changing alpha in the config actually propagates (the
        # de-saturation knob; see channel.path_loss_exponent).
        self.path_loss_exp = float(ch["path_loss_exponent"])
        self.eta_linear = 0.8
        self.tau_max = cov["tau_max"]
        self.beamwidth_deg = cov["beamwidth_deg"]
        self.tan_theta = math.tan(math.radians(self.beamwidth_deg))
        self.h_safe = cov["h_safe"]

        self.H_min, self.H_max = mob["H_min"], mob["H_max"]
        self.fixed_alt = uav["flight_height_fixed"]
        self.depot = np.array([sc["data_center"]["x"], sc["data_center"]["y"]], dtype=np.float32)
        self.depot_alt = sc["data_center"]["z"]
        self.C_max = uav["C_max"]
        # E_max is a battery capacity in mAh; convert to the Joules budget used
        # for the per-UAV energy constraint (E_j <= E_max, Paper A eq. 13).
        self.E_max = uav["E_max"] * uav["battery_voltage"] * 3.6
        self.P_T, self.P_C = uav["P_transmit"], uav["P_collect"]

        # --- rotary-wing propulsion power-speed model (Zeng-Zhang); hover = peak ---
        prop = params.get("propulsion")
        if prop is not None:
            self.power_model = RotaryWingPower(
                P0=prop["P0"], Pi=prop["Pi"], U_tip=prop["U_tip"], v0=prop["v0"],
                d0=prop["d0"], rho=prop["rho"], s=prop["s"], A=prop["A"],
            )
            self.P_H = self.power_model.P_hover           # hover power = P0+Pi (expensive peak)
        else:
            self.power_model = None
            self.P_H = uav["P_hover"]
        # WPT (wireless power transfer) — feature-flagged. When OFF (the priority
        # contribution), no harvest energy is required (E_harvest_coeff forced to 0 =>
        # t_e=0 in _hover_energy) AND the closed-form service-altitude override is
        # disabled, so the decoder's SAMPLED altitude drives reward (a true learned action).
        self.wpt_enabled = bool(uav.get("wpt_enabled", True))
        self.E_harvest_coeff = float(uav.get("E_harvest_coeff_j_per_mb", 0.0)) if self.wpt_enabled else 0.0
        # use the dive-to-serve H_s* altitude allocation ONLY in the WPT variant; never
        # in the priority/no-WPT contribution (otherwise the altitude head gets no gradient).
        self.use_service_altitude = bool(
            self.wpt_enabled and self.is_3d and prop is not None and self.E_harvest_coeff > 0
            and train_cfg.freeze_altitude is None
        )
        self.climb_coeff = uav["mass_kg"] * 9.81 / max(uav["motor_efficiency"], 1e-6)
        self.descent_cd = uav["descent_coeff_cd"]

        # feature-normalisation scales (applied to encoder input only)
        self.xy_scale = max(sc["area_width"], sc["area_height"]) / 2.0
        self.z_scale = max(params["nodes"]["zi_max"], 1.0)
        self.d_scale = max(params["nodes"]["Di_max"], 1.0)

        # --- channel model (used for the R_min rate test inside the cone) ---
        self.channel = ChannelModel3D(
            beamwidth_deg=self.beamwidth_deg, bandwidth=self.bandwidth,
            noise_power_dbm=ch["noise_power_dbm"], carrier_freq=ch["carrier_frequency"],
            path_loss_exp=ch["path_loss_exponent"], transmit_power=self.P_T,
        )

        # --- physics ---
        self.phys3d = UavPhysics3D(
            v_xy_max=mob["v_xy_max"], v_z_max=mob["v_z_max"],
            a_xy_max=mob["a_xy_max"], a_z_max=mob["a_z_max"],
            H_min=self.H_min, H_max=self.H_max, delta_t=mob["delta_t"],
            P_flight=uav["P_flight"], P_hover=self.P_H,
            battery_capacity_mah=self.E_max, battery_voltage=uav["battery_voltage"],
            mass_kg=uav["mass_kg"], motor_efficiency=uav["motor_efficiency"],
            descent_coeff_cd=uav["descent_coeff_cd"],
            power_model=self.power_model,
        )
        # service-altitude allocator (the novel dive-to-serve sub-problem, §5a)
        self.allocator = ServiceAltitudeAllocator(
            bandwidth=self.bandwidth, sigma2=self.sigma2, beta=self.beta, P_T=self.P_T,
            eta_linear=self.eta_linear, P_hover=self.P_H,
            c_climb=self.climb_coeff, c_descent=self.descent_cd,
            tan_theta=self.tan_theta, H_min=self.H_min, H_max=self.H_max, h_safe=self.h_safe,
        )
        self.phys2d = UavPhysics2D(
            fly_speed=uav["fly_speed_max"], P_flight=uav["P_flight"], P_hover=self.P_H,
            flight_height=self.fixed_alt, battery_capacity_mah=self.E_max,
            battery_voltage=uav["battery_voltage"],
        )

        # --- networks ---
        # When priority_feature is on, the encoder also ingests a per-node priority
        # channel (normalised importance weight), so the policy can condition anchor
        # and altitude choices on which sensors are critical.
        self.priority_feature = bool(train_cfg.priority_feature)
        _pri = params.get("priority", {}) or {}
        _wmap = _pri.get("weights", {}) or {}
        self.w_scale = max([float(v) for v in _wmap.values()], default=1.0) if _wmap else 1.0
        input_dim = (4 if self.is_3d else 3) + (1 if self.priority_feature else 0)
        EncoderCls = GraphEncoder if train_cfg.encoder == "attention" else GNNEncoder
        self.encoder = EncoderCls(
            input_dim=input_dim, embed_dim=train_cfg.embed_dim,
            num_heads=train_cfg.num_heads, ff_dim=train_cfg.ff_dim,
            num_layers=train_cfg.num_layers, dropout=train_cfg.dropout,
        ).to(self.device)
        self.decoder = TrajectoryDecoder(
            embed_dim=train_cfg.embed_dim, num_heads=train_cfg.num_heads,
            dim_3d=self.is_3d, H_min=self.H_min, H_max=self.H_max,
            fixed_altitude=self.fixed_alt, tan_theta=self.tan_theta,
            clip_logits=params["decoder"]["clip_logits"],
            freeze_altitude=train_cfg.freeze_altitude,
        ).to(self.device)
        self.critic = CriticNetwork(
            embed_dim=train_cfg.embed_dim, hidden_dim=max(train_cfg.embed_dim // 2, 64),
        ).to(self.device)

        self.reward_cfg = RewardConfig(
            energy_scale=float(params["reward"].get("energy_scale", 1.0e4)),
            unserved_penalty=float(params["reward"].get("unserved_penalty", 5.0)),
            lambda_priority=float(params["reward"].get("lambda_priority", 0.0)),
            freshness_coeff=float(params["reward"].get("freshness_penalty_coeff", 0.0)),
        )

        # --- primal-dual CMDP solver state (per-class QoS dual multipliers) ---
        # Opt-in: when train_cfg.cmdp is False, self.duals stays None and train_step
        # uses the unchanged fixed-lambda reward. The CONSTRAINED classes are exactly
        # those carrying a positive per-class R_min floor (low has no floor => no dual),
        # each mapped to the scorer's existing per-instance satisfaction metric.
        self.cmdp_cfg = CMDPConfig.from_params(params, enabled=bool(train_cfg.cmdp))
        _pr = (params.get("priority", {}) or {}).get("R_min", {}) or {}
        self._cmdp_class_keys: Dict[str, str] = {}
        if float(_pr.get("high", 0.0)) > 0:
            self._cmdp_class_keys["high"] = "high_qos_satisfaction"
        if float(_pr.get("medium", 0.0)) > 0:
            self._cmdp_class_keys["medium"] = "med_qos_satisfaction"
        self.duals = (LagrangianDuals(list(self._cmdp_class_keys.keys()), self.cmdp_cfg)
                      if self.cmdp_cfg.enabled else None)

        # Single uniform optimizer/schedule for the whole actor. (Option D — a
        # separate non-decaying LR group for the altitude head — was reverted on
        # 2026-07-02: the plateau's root cause was a non-detached rsample() in the
        # decoder that made the altitude score-function gradient identically zero,
        # not the LR schedule. See trajectory_decoder.py altitude-action comment.)
        actor_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.opt_actor = torch.optim.Adam(actor_params, lr=train_cfg.lr_actor)
        self.sched_actor = torch.optim.lr_scheduler.ExponentialLR(self.opt_actor, train_cfg.lr_decay)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=train_cfg.lr_critic)
        self.sched_critic = torch.optim.lr_scheduler.ExponentialLR(self.opt_critic, train_cfg.lr_decay)

    # ------------------------------------------------------------------
    def _normalize(self, node_features: torch.Tensor) -> torch.Tensor:
        """Scale raw [x,y,(z),D] features into an O(1) range for the encoder."""
        f = node_features.clone()
        f[..., 0] /= self.xy_scale
        f[..., 1] /= self.xy_scale
        if self.is_3d:
            f[..., 2] /= self.z_scale
            f[..., 3] /= self.d_scale
        else:
            f[..., 2] /= self.d_scale
        return f

    def _encode_decode(
        self, node_features: torch.Tensor, R_min: float, greedy: bool,
        node_rmin: Optional[torch.Tensor] = None,
        node_weights: Optional[torch.Tensor] = None,
    ):
        """Run encoder + decoder; returns (plan, h_graph)."""
        norm = self._normalize(node_features).to(self.device)
        if self.priority_feature:
            # append normalised priority importance as an extra encoder channel
            if node_weights is not None:
                wch = (node_weights.float() / max(self.w_scale, 1e-6)).unsqueeze(-1).to(self.device)
            else:
                wch = torch.zeros(norm.shape[0], norm.shape[1], 1, device=self.device)
            norm = torch.cat([norm, wch], dim=-1)
        h_nodes, h_graph = self.encoder(norm)

        node_xy = node_features[..., :2].to(self.device)
        if self.is_3d:
            node_z = node_features[..., 2].to(self.device)
            node_d = node_features[..., 3].to(self.device)
        else:
            node_z = torch.zeros_like(node_features[..., 0]).to(self.device)
            node_d = node_features[..., 2].to(self.device)

        nr = node_rmin.to(self.device) if node_rmin is not None else None
        plan = self.decoder(
            h_nodes, h_graph, node_xy, node_z, node_d,
            R_min=R_min, channel=self.channel, greedy=greedy, node_rmin=nr,
        )
        return plan, h_graph

    # ------------------------------------------------------------------
    def _hover_energy(self, d3d, demands):
        """Charge+collect energy and time for a set of served nodes at distances d3d.

        Uses the explicit physical model when the rotary-wing power model is set:
        WPT charge time T_E = E_S / P_R (deliver required harvested energy), and
        collection time T_C = D_bits / rate. Falls back to Paper-A KKT otherwise.
        Returns (per-node T_E+T_C, e_hover).
        """
        if self.power_model is not None:
            d = np.maximum(d3d, 1e-3)
            E_S = self.E_harvest_coeff * np.maximum(demands, 0.0)         # required harvest (J)
            P_R = self.eta_linear * self.beta * self.P_T / d ** 2
            t_e = E_S / np.maximum(P_R, 1e-30)
            snr = self.P_T * self.beta / (d ** self.path_loss_exp * self.sigma2)
            rate = self.bandwidth * np.log2(1.0 + snr)
            t_c = (np.maximum(demands, 0.0) * 8e6) / np.maximum(rate, 1e-6)
            return t_c, t_e
        return kkt_times(
            d3d, demands, bandwidth=self.bandwidth, sigma2=self.sigma2,
            beta=self.beta, P_T=self.P_T, eta_linear=self.eta_linear, tau_max=self.tau_max,
        )

    def _partition_and_evaluate(
        self,
        plan,
        node_features: torch.Tensor,
        R_min: float = 0.0,
        use_service_altitude: bool = False,
        node_weights: Optional[torch.Tensor] = None,
        node_rmin: Optional[torch.Tensor] = None,
        refine_routing: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Turn the decoder plan into per-instance total energy + reward.

        For every instance we walk the ordered hover anchors, fit each hover's
        charge+collect time (capped by tau_max), and segment the route into
        multiple UAVs whenever cumulative storage (C_max) or battery (E_max)
        would be exceeded. When ``use_service_altitude`` is set, each hover's
        altitude is overridden with the energy-optimal H_s* for its served
        cluster (the novel dive-to-serve allocation, §5a). Returns (B,) arrays.
        """
        B = node_features.shape[0]
        N = node_features.shape[1]
        nf = node_features.detach().cpu().numpy()
        nw = node_weights.detach().cpu().numpy() if node_weights is not None else None
        nr = node_rmin.detach().cpu().numpy() if node_rmin is not None else None
        anchors = plan.anchors.detach().cpu().numpy()
        altitudes = plan.altitudes.detach().cpu().numpy()
        served = plan.served.detach().cpu().numpy()          # (B, T, N) bool
        active = plan.step_active.detach().cpu().numpy()      # (B, T)
        T = anchors.shape[1]

        rewards = np.zeros(B, dtype=np.float32)
        energies = np.zeros(B, dtype=np.float32)
        n_uavs = np.zeros(B, dtype=np.int32)
        unserved_fracs = np.zeros(B, dtype=np.float32)
        data_collected = np.zeros(B, dtype=np.float32)
        penalties = np.zeros(B, dtype=np.float32)
        prio_sat = np.ones(B, dtype=np.float32)   # frac of QoS-floored nodes served within floor
        # per-class QoS satisfaction (high / medium); NaN for an instance with no such nodes
        high_sat = np.full(B, np.nan, dtype=np.float32)
        med_sat = np.full(B, np.nan, dtype=np.float32)
        # reference per-class floors from config (class<->floor is bijective here)
        _pr = (self.params.get("priority", {}) or {}).get("R_min", {}) or {}
        hi_floor = float(_pr.get("high", 0.0)); md_floor = float(_pr.get("medium", 0.0))

        for b in range(B):
            xy = nf[b, :, :2]
            if self.is_3d:
                z = nf[b, :, 2]; dem = nf[b, :, 3]
            else:
                z = np.zeros(N); dem = nf[b, :, 2]

            # per-node achieved rate + collection order, for the priority penalty
            rate_arr = np.zeros(N, dtype=np.float64)
            order_arr = np.zeros(N, dtype=np.float64)
            serve_counter = 0

            # ---- build the ordered list of real hovers for this instance ----
            hovers = []  # each: (anchor_xy, H, served_idx, t_hover, e_hover, demand)
            for t in range(T):
                if not active[b, t]:
                    continue
                s_idx = np.where(served[b, t])[0]
                if s_idx.size == 0:
                    continue
                a_xy = xy[anchors[b, t]]
                H = float(altitudes[b, t]) if self.is_3d else self.fixed_alt

                rho = np.linalg.norm(xy[s_idx] - a_xy, axis=1)

                # NOVEL: override altitude with the energy-optimal dive depth H_s*
                # for this served cluster (must still cover it: H_s >= z + r/tanθ).
                if use_service_altitude and self.is_3d:
                    z_S = float(z[s_idx].mean())
                    r_S = float(rho.max()) if rho.size else 0.0
                    D_S = float(dem[s_idx].sum())
                    E_S = self.E_harvest_coeff * D_S
                    H, _, _, _ = self.allocator.optimal(
                        z_S=z_S, E_S=E_S, data_mb=D_S, r_S=r_S, H_cruise=self.H_max,
                    )

                # 3D distance from hover to each served node
                if self.is_3d:
                    d3d = np.sqrt(rho ** 2 + (H - z[s_idx]) ** 2)
                else:
                    d3d = np.sqrt(rho ** 2 + self.fixed_alt ** 2)

                t_c, t_e = self._hover_energy(d3d, dem[s_idx])
                node_time = t_c + t_e
                # Serve-all (A4, hard): every cone node the decoder assigned to this
                # hover IS collected -- no node is dropped. tau_max is NOT a constraint
                # of the optimization problem (P0-3D, §6); it is only a per-decode-step
                # serving device in the MDP transition (§8). The hover energy is the
                # node-indexed sum E^C=(P^C+P^H)ΣT^C_i, E^T=(P^T+P^H)ΣT^E_i (§3,§5),
                # which is invariant to how nodes are grouped into hover stops, so the
                # scorer simply accounts every assigned node. Nearest-first ordering is
                # kept only for the freshness/collection-order term.
                order = np.argsort(d3d)     # nearest-first (collection order / freshness)
                keep = order                # serve ALL nodes in the cluster (no silent drop)
                kept_idx = s_idx[keep]
                t_c_k, t_e_k = t_c[keep], t_e[keep]

                # achieved rate (for the priority R_min test) + collection order
                # (for the freshness term) at each node served by this hover
                d_kept = np.maximum(d3d[keep], 1e-3)
                snr_kept = self.P_T * self.beta / (d_kept ** self.path_loss_exp * self.sigma2)
                rate_arr[kept_idx] = self.bandwidth * np.log2(1.0 + snr_kept)
                order_arr[kept_idx] = serve_counter
                serve_counter += int(kept_idx.size)

                # E^T = (P_T+P_H) sum T_E ; E^C = (P_C+P_H) sum T_C
                e_hover = (self.P_T + self.P_H) * float(t_e_k.sum()) + \
                          (self.P_C + self.P_H) * float(t_c_k.sum())
                t_hover = float((t_c_k + t_e_k).sum())
                hovers.append((a_xy, H, kept_idx, t_hover, e_hover, float(dem[kept_idx].sum())))

            # nodes never served (footprint missed them at every hover)
            served_any = np.zeros(N, dtype=bool)
            for h in hovers:
                served_any[h[2]] = True
            unserved = int((~served_any).sum())

            # Phase 1: identical NN+2opt routing refinement for EVERY evaluated
            # solution (learned policy + all deterministic baselines), so energy
            # comparisons isolate clustering/altitude/assignment from routing luck.
            # Eval-only (off during training, both for speed and to keep routing a
            # learned action). Reordering the hover sequence changes the route AND
            # the order-dependent capacity/battery segmentation below; the freshness
            # collection-order is recomputed to follow the refined route.
            if refine_routing and len(hovers) > 1:
                # keep the native (float32) anchor dtype: the baselines fed nn_2opt
                # the raw node positions, and forcing float64 here would cross the
                # 2-opt improvement threshold differently and diverge from them.
                pts = np.array([h[0] for h in hovers])
                route = nn_2opt_tour(pts, self.depot)
                hovers = [hovers[i] for i in route]
                if nw is not None or nr is not None:
                    sc2 = 0
                    for h in hovers:
                        order_arr[h[2]] = sc2
                        sc2 += int(h[2].size)

            # ---- multi-UAV segmentation by capacity + battery ----
            total_energy = 0.0
            uav_count = 0
            cur_pos = self.depot.copy(); cur_alt = self.depot_alt
            cur_cap = 0.0; cur_energy = 0.0
            open_uav = False

            def leg_energy(p0, a0, p1, a1):
                if self.is_3d:
                    seg = self.phys3d.simulate_leg(p0, a0, p1, a1)
                    return seg.energy_consumed + seg.vert_energy
                d = float(np.linalg.norm(p0[:2] - p1[:2]))
                return self.phys2d.compute_flight_energy(d)

            for (a_xy, H, kept_idx, t_hover, e_hover, dem_sum) in hovers:
                in_e = leg_energy(cur_pos, cur_alt, a_xy, H)
                ret_e = leg_energy(a_xy, H, self.depot, self.depot_alt)
                would_cap = cur_cap + dem_sum
                would_energy = cur_energy + in_e + e_hover + ret_e

                if open_uav and (would_cap > self.C_max or would_energy > self.E_max):
                    # close current UAV: fly home from previous position
                    total_energy += leg_energy(cur_pos, cur_alt, self.depot, self.depot_alt)
                    uav_count += 1
                    cur_pos = self.depot.copy(); cur_alt = self.depot_alt
                    cur_cap = 0.0; cur_energy = 0.0
                    open_uav = False
                    in_e = leg_energy(cur_pos, cur_alt, a_xy, H)

                cur_energy += in_e + e_hover
                cur_cap += dem_sum
                total_energy += in_e + e_hover
                cur_pos = a_xy.copy(); cur_alt = H
                open_uav = True

            if open_uav:
                total_energy += leg_energy(cur_pos, cur_alt, self.depot, self.depot_alt)
                uav_count += 1

            frac_unserved = unserved / max(N, 1)

            # priority-weighted quality/freshness penalty (0.0 when priority is off)
            if nw is not None or nr is not None:
                w_b = nw[b] if nw is not None else np.ones(N, dtype=np.float64)
                r_b = nr[b] if nr is not None else np.zeros(N, dtype=np.float64)
                ord_norm = order_arr / max(serve_counter - 1, 1)
                pen = priority_penalty(
                    w_b, r_b, served_any, rate_arr, ord_norm, self.reward_cfg.freshness_coeff)
                # priority-satisfaction: among nodes carrying a floor, fraction
                # actually served AND meeting their R_min (the QoS guarantee metric)
                floor_mask = r_b > 0
                if floor_mask.any():
                    met = served_any & (rate_arr >= r_b)
                    prio_sat[b] = float(met[floor_mask].sum() / floor_mask.sum())
                    # per-class satisfaction (high = strictest floor, medium = next)
                    hi_mask = floor_mask & np.isclose(r_b, hi_floor) if hi_floor > 0 else np.zeros(N, bool)
                    md_mask = floor_mask & np.isclose(r_b, md_floor) if md_floor > 0 else np.zeros(N, bool)
                    if hi_mask.any():
                        high_sat[b] = float(met[hi_mask].sum() / hi_mask.sum())
                    if md_mask.any():
                        med_sat[b] = float(met[md_mask].sum() / md_mask.sum())
            else:
                pen = 0.0

            rewards[b] = energy_reward(total_energy, frac_unserved, pen, self.reward_cfg)
            energies[b] = total_energy
            n_uavs[b] = uav_count
            unserved_fracs[b] = frac_unserved
            data_collected[b] = float(dem[served_any].sum())
            penalties[b] = pen

        return {
            "reward": rewards, "energy": energies, "num_uavs": n_uavs,
            "unserved_frac": unserved_fracs, "data_collected": data_collected,
            "priority_penalty": penalties, "priority_satisfaction": prio_sat,
            "high_qos_satisfaction": high_sat, "med_qos_satisfaction": med_sat,
        }

    # ------------------------------------------------------------------
    def _lagrangian_reward(self, metrics: Dict[str, np.ndarray]):
        """CMDP primal reward and per-class mean constraint violation.

        Assembled ONLY from the frozen scorer's outputs (energy, unserved, per-class
        satisfaction) — the scorer is not modified. Per instance b:

            R_b = -E_b/scale - mu*unserved_b - Σ_c λ_c · g_{c,b},  g_{c,b}=target - sat_{c,b}

        Instances with no class-c node have sat=NaN -> g=0 there (constraint inactive,
        no dual contribution). Returns (reward (B,) float32, mean_violation per class
        averaged over the instances where the class is active).
        """
        cfg = self.reward_cfg
        r = -(metrics["energy"] / cfg.energy_scale) \
            - cfg.unserved_penalty * metrics["unserved_frac"]
        mean_viol: Dict[str, float] = {}
        for c, key in self._cmdp_class_keys.items():
            g = self.cmdp_cfg.target_satisfaction - metrics[key]      # (B,), NaN where inactive
            r = r - self.duals.lmbda[c] * np.nan_to_num(g, nan=0.0)
            active = g[~np.isnan(g)]
            if active.size:
                mean_viol[c] = float(active.mean())
        return r.astype(np.float32), mean_viol

    def train_step(
        self,
        node_features: torch.Tensor,
        R_min: float = 0.0,
        node_weights: Optional[torch.Tensor] = None,
        node_rmin: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """One REINFORCE update on a batch of instances."""
        self.encoder.train(); self.decoder.train(); self.critic.train()

        plan, h_graph = self._encode_decode(
            node_features, R_min, greedy=False, node_rmin=node_rmin, node_weights=node_weights)
        metrics = self._partition_and_evaluate(
            plan, node_features, R_min, use_service_altitude=self.use_service_altitude,
            node_weights=node_weights, node_rmin=node_rmin)
        # CMDP: replace the fixed-lambda reward with the augmented-Lagrangian reward
        # (per-class duals). When CMDP is off, use the scorer's reward unchanged.
        mean_viol: Dict[str, float] = {}
        if self.duals is not None:
            r_np, mean_viol = self._lagrangian_reward(metrics)
            reward = torch.from_numpy(r_np).float().to(self.device)
        else:
            reward = torch.from_numpy(metrics["reward"]).float().to(self.device)

        # critic baseline
        value = self.critic(h_graph).squeeze(-1)               # (B,)
        advantage = (reward - value).detach()

        # sum log-probs over the real decode steps (anchor + altitude)
        active = plan.step_active.float()
        logp = ((plan.log_p_anchor + plan.log_p_alt) * active).sum(dim=1)  # (B,)

        actor_loss = -(advantage * logp).mean()
        if self.cfg.entropy_coef > 0:
            actor_loss = actor_loss - self.cfg.entropy_coef * plan.entropy.mean()
        critic_loss = nn.functional.mse_loss(value, reward)

        self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.cfg.max_grad_norm,
        )
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_actor.step(); self.opt_critic.step()

        # CMDP dual step: projected ascent on this batch's mean violation (after the
        # primal step at the current multipliers — standard primal-dual ordering).
        if self.duals is not None:
            self.duals.ascent(mean_viol)

        out = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "reward": float(reward.mean().item()),
            "energy": float(metrics["energy"].mean()),
            "num_uavs": float(metrics["num_uavs"].mean()),
            "unserved_frac": float(metrics["unserved_frac"].mean()),
            "data_collected": float(metrics["data_collected"].mean()),
            "priority_penalty": float(metrics["priority_penalty"].mean()),
        }
        if self.duals is not None:
            out["duals"] = self.duals.as_dict()
            out["cmdp_violation"] = mean_viol
            for c, key in self._cmdp_class_keys.items():
                sat = metrics[key]
                out[f"{c}_qos_satisfaction"] = (float(np.nanmean(sat))
                                                if not np.all(np.isnan(sat)) else float("nan"))
        return out

    @torch.no_grad()
    def evaluate(
        self,
        node_features: torch.Tensor,
        R_min: float = 0.0,
        greedy: bool = True,
        node_weights: Optional[torch.Tensor] = None,
        node_rmin: Optional[torch.Tensor] = None,
        qos_decode: bool = True,
    ) -> Dict[str, float]:
        """Greedy (or sampled) rollout for evaluation; returns averaged metrics.

        ``qos_decode`` controls ONLY the decoder's footprint QoS gate, decoupled from
        the metric. Set it to match how the model was TRAINED: True for a priority-aware
        model (it gated on per-node floors during training), False for a priority-BLIND
        model (so we measure its real, ungated policy — passing floors to its decoder
        would make a blind policy behave priority-aware and invalidate the baseline).
        The satisfaction/penalty metrics ALWAYS use the true ``node_rmin``.
        """
        self.encoder.eval(); self.decoder.eval(); self.critic.eval()
        decode_rmin = node_rmin if qos_decode else None
        plan, _ = self._encode_decode(
            node_features, R_min, greedy=greedy, node_rmin=decode_rmin, node_weights=node_weights)
        m = self._partition_and_evaluate(
            plan, node_features, R_min, use_service_altitude=self.use_service_altitude,
            node_weights=node_weights, node_rmin=node_rmin, refine_routing=True)
        return {
            "reward": float(m["reward"].mean()),
            "energy": float(m["energy"].mean()),
            "energy_wh": float(m["energy"].mean() / 3600.0),
            "num_uavs": float(m["num_uavs"].mean()),
            "unserved_frac": float(m["unserved_frac"].mean()),
            "data_collected": float(m["data_collected"].mean()),
            "priority_penalty": float(m["priority_penalty"].mean()),
            "priority_satisfaction": float(m["priority_satisfaction"].mean()),
            "high_qos_satisfaction": float(np.nanmean(m["high_qos_satisfaction"]))
                if not np.all(np.isnan(m["high_qos_satisfaction"])) else float("nan"),
            "med_qos_satisfaction": float(np.nanmean(m["med_qos_satisfaction"]))
                if not np.all(np.isnan(m["med_qos_satisfaction"])) else float("nan"),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_routes(
        self, node_features: torch.Tensor, R_min: float = 0.0,
        node_weights: Optional[torch.Tensor] = None,
        node_rmin: Optional[torch.Tensor] = None,
    ):
        """Reconstruct per-UAV anchor routes for one instance (index 0).

        Returns (routes, altitudes) where each entry is one UAV: ``routes[j]`` is
        an (K_j+2, 2) array of horizontal waypoints (depot -> anchors -> depot)
        and ``altitudes[j]`` the matching (K_j+2,) altitudes. Used for plotting.

        Pass ``node_weights`` / ``node_rmin`` to plot a priority model faithfully
        (so the encoder sees priority and the decoder applies per-node QoS gating);
        omit them for an energy-only model.
        """
        self.encoder.eval(); self.decoder.eval()
        nw = node_weights[:1] if node_weights is not None else None
        nr = node_rmin[:1] if node_rmin is not None else None
        plan, _ = self._encode_decode(
            node_features[:1], R_min, greedy=True, node_rmin=nr, node_weights=nw)
        nf = node_features[0].cpu().numpy()
        xy = nf[:, :2]
        anchors = plan.anchors[0].cpu().numpy()
        alts = plan.altitudes[0].cpu().numpy()
        served = plan.served[0].cpu().numpy()
        active = plan.step_active[0].cpu().numpy()

        # ordered hovers (anchor xy, altitude, demand summed)
        hovers = []
        if self.is_3d:
            dem = nf[:, 3]
        else:
            dem = nf[:, 2]
        for t in range(len(anchors)):
            if not active[t] or served[t].sum() == 0:
                continue
            H = float(alts[t]) if self.is_3d else self.fixed_alt
            hovers.append((xy[anchors[t]], H, float(dem[served[t]].sum())))

        # segment by capacity + battery, mirroring _partition_and_evaluate
        routes, altitudes = [], []
        cur_xy = [self.depot.copy()]; cur_alt = [self.depot_alt]
        cur_cap = 0.0; cur_energy = 0.0; cur_pos = self.depot.copy(); cur_a = self.depot_alt

        def leg_e(p0, a0, p1, a1):
            if self.is_3d:
                seg = self.phys3d.simulate_leg(p0, a0, p1, a1)
                return seg.energy_consumed + seg.vert_energy
            return self.phys2d.compute_flight_energy(float(np.linalg.norm(p0[:2] - p1[:2])))

        for (a_xy, H, dem_sum) in hovers:
            in_e = leg_e(cur_pos, cur_a, a_xy, H)
            ret_e = leg_e(a_xy, H, self.depot, self.depot_alt)
            if len(cur_xy) > 1 and (cur_cap + dem_sum > self.C_max or
                                    cur_energy + in_e + ret_e > self.E_max):
                cur_xy.append(self.depot.copy()); cur_alt.append(self.depot_alt)
                routes.append(np.array(cur_xy)); altitudes.append(np.array(cur_alt))
                cur_xy = [self.depot.copy()]; cur_alt = [self.depot_alt]
                cur_cap = 0.0; cur_energy = 0.0; cur_pos = self.depot.copy(); cur_a = self.depot_alt
                in_e = leg_e(cur_pos, cur_a, a_xy, H)
            cur_xy.append(a_xy); cur_alt.append(H)
            cur_cap += dem_sum; cur_energy += in_e
            cur_pos = a_xy; cur_a = H
        if len(cur_xy) > 1:
            cur_xy.append(self.depot.copy()); cur_alt.append(self.depot_alt)
            routes.append(np.array(cur_xy)); altitudes.append(np.array(cur_alt))
        return routes, altitudes

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "sched_actor": self.sched_actor.state_dict(),
            "sched_critic": self.sched_critic.state_dict(),
            "cfg": self.cfg.__dict__,
            "duals": self.duals.state_dict() if self.duals is not None else None,
        }

    def load_state_dict(self, sd: dict):
        # Tolerate the epoch-wrapped checkpoint format {"sd": <trainer state>, "epoch": k}
        # so any loader (run_eval, --resume) accepts both new and legacy bare checkpoints.
        if "sd" in sd and "encoder" not in sd:
            sd = sd["sd"]
        self.encoder.load_state_dict(sd["encoder"])
        self.decoder.load_state_dict(sd["decoder"])
        self.critic.load_state_dict(sd["critic"])
        if "opt_actor" in sd:
            self.opt_actor.load_state_dict(sd["opt_actor"])
            self.opt_critic.load_state_dict(sd["opt_critic"])
        # LR scheduler state (guarded: old checkpoints predate these keys)
        if "sched_actor" in sd:
            self.sched_actor.load_state_dict(sd["sched_actor"])
            self.sched_critic.load_state_dict(sd["sched_critic"])
        if self.duals is not None and sd.get("duals"):
            self.duals.load_state_dict(sd["duals"])
