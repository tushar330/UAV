"""
Autoregressive Trajectory Decoder (Paper A ATOM decoder, extended to 3D).

Given the per-node embeddings ``h_nodes`` and the graph summary ``h_sa`` produced
by :class:`GraphEncoder`, the decoder builds a route one *anchor* at a time using
a pointer-attention mechanism (Kool et al. attention model / Paper A Sec. III-C).

Two modes:

* **2D** (Paper A baseline): visit every node individually. Each decode step
  selects the next unvisited node; the UAV flies there at the fixed altitude and
  collects its data. The rollout ends when every node has been visited.

* **3D** (ATOM-3D, novel): each decode step picks an anchor node **and an
  altitude** ``H`` (a *stochastic* Gaussian action — this is what gives the
  altitude head a reward gradient, spec §8). The UAV hovers at ``(x, y, H)`` and
  serves *every* still-unserved node inside the coverage cone
  ``rho <= (H - z) * tan(theta)`` that also meets the QoS floor ``R_min``. Those
  footprint-collected nodes are masked out, so the next anchor is chosen only
  from the remainder. The rollout ends when every node is served.

The decoder returns a *plan* (anchors, altitudes, per-step served masks) plus the
log-probabilities needed for the REINFORCE policy gradient. Energy accounting,
multi-UAV segmentation and the reward live in the trainer's
``_partition_and_evaluate`` so this module stays a pure policy network.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecodePlan:
    """Output of a decoder rollout for a whole batch.

    All tensors are padded to ``T`` = the max number of decode steps used by any
    instance in the batch; ``step_active`` flags the real (non-padding) steps.
    """
    anchors: torch.Tensor          # (B, T) long — chosen anchor node index per step
    altitudes: torch.Tensor        # (B, T) float — hover altitude per step (3D); fixed alt in 2D
    served: torch.Tensor           # (B, T, N) bool — nodes collected at each step's hover
    log_p_anchor: torch.Tensor     # (B, T) — log prob of the anchor choice
    log_p_alt: torch.Tensor        # (B, T) — log prob of the altitude choice (0 in 2D)
    step_active: torch.Tensor      # (B, T) bool — True for real steps, False for padding
    entropy: torch.Tensor          # (B,) — summed policy entropy (for optional regularisation)


class TrajectoryDecoder(nn.Module):
    """Pointer-attention decoder with an optional stochastic altitude head.

    Args:
        embed_dim:   node embedding dimension D (must match the encoder).
        num_heads:   heads for the glimpse attention.
        dim_3d:      if True, enable the altitude head and footprint rollout.
        H_min/H_max: altitude bounds (m), used to squash the altitude action.
        fixed_altitude: altitude used in 2D mode (m).
        tan_theta:   tan of the coverage half-beamwidth (footprint radius factor).
        clip_logits: tanh clipping value C for the pointer logits (Paper A).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dim_3d: bool = False,
        H_min: float = 20.0,
        H_max: float = 150.0,
        fixed_altitude: float = 20.0,
        tan_theta: float = math.tan(math.radians(60.0)),
        clip_logits: float = 10.0,
        freeze_altitude: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_3d = dim_3d
        self.H_min = H_min
        self.H_max = H_max
        self.fixed_altitude = fixed_altitude
        self.tan_theta = tan_theta
        self.clip_logits = clip_logits
        # Ablation (spec §6): if set, the 3D decoder still covers footprints but
        # hovers at this ONE altitude — the altitude head is disabled, isolating
        # the benefit of *varying* altitude from the benefit of footprint coverage.
        self.freeze_altitude = freeze_altitude

        # Number of scalar context features appended to the query.
        #   2D: [frac_served, frac_capacity_left]
        #   3D: [frac_served, frac_capacity_left, current_altitude_norm]
        n_scalar = 3 if dim_3d else 2

        # Context = [h_graph, h_prev_anchor, scalars] -> query of dim D.
        self.context_proj = nn.Linear(2 * embed_dim + n_scalar, embed_dim)

        # Glimpse: multi-head attention of the context query over node embeddings.
        self.glimpse = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Projections for the final single-head pointer compatibility logits.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # Learned starting "previous anchor" embedding (depot placeholder).
        self.start_token = nn.Parameter(torch.randn(embed_dim) * 0.1)

        if dim_3d:
            # Gaussian altitude head: query -> (mean, log_std) of a pre-squash normal.
            self.alt_mean = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
            )
            self.alt_log_std = nn.Parameter(torch.zeros(1) - 0.5)  # std ~ 0.6 initially

    # ------------------------------------------------------------------
    def forward(
        self,
        h_nodes: torch.Tensor,       # (B, N, D)
        h_graph: torch.Tensor,       # (B, D)
        node_xy: torch.Tensor,       # (B, N, 2) real coords
        node_z: torch.Tensor,        # (B, N) ground elevation (zeros in 2D)
        node_demand: torch.Tensor,   # (B, N) data demand (MB)
        R_min: float = 0.0,
        channel: Optional[object] = None,   # ChannelModel3D for R_min rate test
        greedy: bool = False,
        max_steps: Optional[int] = None,
        node_rmin: Optional[torch.Tensor] = None,   # (B, N) per-node QoS floor (priority)
    ) -> DecodePlan:
        """Run an autoregressive rollout over the batch.

        ``channel`` is only needed when a rate floor is active to evaluate the
        per-node rate inside the footprint; its ``compute_data_rate`` is reused.
        ``node_rmin`` (priority mode) gives each node its own floor and takes
        precedence over the scalar ``R_min`` (uniform mode). Nodes whose link
        cannot clear their floor are not footprint-served; the anchor itself is
        always collected (serve-all + progress), so a critical node served too
        high is still collected but incurs the reward's quality-miss penalty.
        """
        B, N, D = h_nodes.shape
        device = h_nodes.device
        if max_steps is None:
            max_steps = N  # worst case (2D); 3D usually stops far earlier

        keys = self.k_proj(h_nodes)               # (B, N, D) pointer keys
        served = torch.zeros(B, N, dtype=torch.bool, device=device)  # collected nodes
        prev = self.start_token.unsqueeze(0).expand(B, D)            # (B, D)

        # Per-instance running capacity proxy (fraction of total demand left).
        total_demand = node_demand.sum(dim=1).clamp(min=1e-6)        # (B,)
        cur_alt = torch.full((B,), self.fixed_altitude, device=device)

        anchors_l, alts_l, served_l = [], [], []
        logpa_l, logp_alt_l, active_l = [], [], []
        entropy = torch.zeros(B, device=device)

        finished = served.all(dim=1)              # (B,) all nodes served?

        for _ in range(max_steps):
            if bool(finished.all()):
                break

            # ---- build the query context ----
            served_frac = served.float().mean(dim=1, keepdim=True)            # (B,1)
            demand_left = (node_demand * (~served).float()).sum(dim=1, keepdim=True)
            cap_frac = (demand_left / total_demand.unsqueeze(1)).clamp(0, 1)  # (B,1)
            if self.dim_3d:
                alt_norm = ((cur_alt - self.H_min) /
                            max(self.H_max - self.H_min, 1e-6)).unsqueeze(1)
                scalars = torch.cat([served_frac, cap_frac, alt_norm], dim=1)
            else:
                scalars = torch.cat([served_frac, cap_frac], dim=1)

            ctx = torch.cat([h_graph, prev, scalars], dim=1)     # (B, 2D+s)
            query = self.context_proj(ctx).unsqueeze(1)          # (B, 1, D)

            # Mask: True = not selectable (already served). Keep finished
            # instances fully masked except one slot to avoid NaN softmax.
            mask = served.clone()                                 # (B, N)
            # For instances with everything served, unmask node 0 (it is padding).
            all_served = mask.all(dim=1)
            if all_served.any():
                mask[all_served, 0] = False

            # ---- glimpse attention (query attends over unserved nodes) ----
            attn_mask = mask                                      # (B, N) key padding
            glimpsed, _ = self.glimpse(
                query, h_nodes, h_nodes, key_padding_mask=attn_mask
            )                                                     # (B, 1, D)
            q = self.q_proj(glimpsed)                             # (B, 1, D)

            # ---- pointer compatibility logits ----
            logits = torch.matmul(q, keys.transpose(1, 2)).squeeze(1)  # (B, N)
            logits = logits / math.sqrt(D)
            logits = self.clip_logits * torch.tanh(logits)
            logits = logits.masked_fill(mask, float('-inf'))

            log_probs = F.log_softmax(logits, dim=1)              # (B, N)
            probs = log_probs.exp()

            if greedy:
                anchor = probs.argmax(dim=1)                      # (B,)
            else:
                anchor = torch.multinomial(probs, 1).squeeze(1)   # (B,)

            log_p_anchor = log_probs.gather(1, anchor.unsqueeze(1)).squeeze(1)
            step_entropy = -(probs * log_probs.clamp(min=-1e9)).nan_to_num().sum(dim=1)

            anchor_xy = node_xy.gather(
                1, anchor.view(B, 1, 1).expand(B, 1, 2)
            ).squeeze(1)                                          # (B, 2)

            # ---- altitude action ----
            if self.dim_3d and self.freeze_altitude is not None:
                # frozen-altitude ablation: constant H, no altitude gradient
                H = torch.full((B,), float(self.freeze_altitude), device=device)
                log_p_alt = torch.zeros(B, device=device)
            elif self.dim_3d:
                mean = self.alt_mean(glimpsed.squeeze(1)).squeeze(1)   # (B,)
                std = self.alt_log_std.exp().clamp(0.05, 2.0)
                dist = torch.distributions.Normal(mean, std)
                # score-function REINFORCE: the sample must NOT carry gradient,
                # otherwise log_prob(mean + std*eps) is constant in mean and the
                # altitude policy gradient is identically zero (audit 2026-07-02).
                raw = mean if greedy else dist.sample()
                log_p_alt = dist.log_prob(raw)                    # (B,)
                # squash to [H_min, H_max]
                H = self.H_min + (self.H_max - self.H_min) * torch.sigmoid(raw)
            else:
                H = torch.full((B,), self.fixed_altitude, device=device)
                log_p_alt = torch.zeros(B, device=device)

            # ---- footprint serving ----
            if self.dim_3d:
                rho = torch.linalg.norm(node_xy - anchor_xy.unsqueeze(1), dim=2)  # (B,N)
                radius = torch.clamp(H.unsqueeze(1) - node_z, min=0.0) * self.tan_theta
                in_cone = rho <= radius
                # per-node QoS gate: a node is footprint-served only if its link
                # clears its own R_min floor (priority) or the scalar R_min (uniform).
                if channel is not None and (node_rmin is not None or R_min > 0.0):
                    d3d = torch.sqrt(rho ** 2 + (H.unsqueeze(1) - node_z) ** 2 + 1e-10)
                    rate = channel.compute_data_rate(d3d)
                    floor = node_rmin if node_rmin is not None else R_min
                    in_cone = in_cone & (rate >= floor)
                step_served = in_cone & (~served)
                # the anchor node itself is always collected at its own hover
                step_served.scatter_(1, anchor.unsqueeze(1), True)
                step_served = step_served & (~served)
            else:
                # 2D: only the chosen node is collected this step
                step_served = torch.zeros_like(served)
                step_served.scatter_(1, anchor.unsqueeze(1), True)
                step_served = step_served & (~served)

            active = ~finished                                    # (B,) real this step

            # Mask out contributions from already-finished instances.
            log_p_anchor = log_p_anchor * active.float()
            log_p_alt = log_p_alt * active.float()
            step_served = step_served & active.unsqueeze(1)
            entropy = entropy + step_entropy * active.float()

            # ---- commit step ----
            served = served | step_served
            prev = torch.where(
                active.unsqueeze(1),
                h_nodes.gather(1, anchor.view(B, 1, 1).expand(B, 1, D)).squeeze(1),
                prev,
            )
            cur_alt = torch.where(active, H, cur_alt)

            anchors_l.append(anchor)
            alts_l.append(H)
            served_l.append(step_served)
            logpa_l.append(log_p_anchor)
            logp_alt_l.append(log_p_alt)
            active_l.append(active)

            finished = served.all(dim=1)

        if not anchors_l:                                         # nothing to do (N==0)
            T = 1
            z = torch.zeros(B, T, device=device)
            return DecodePlan(
                anchors=torch.zeros(B, T, dtype=torch.long, device=device),
                altitudes=torch.full((B, T), self.fixed_altitude, device=device),
                served=torch.zeros(B, T, N, dtype=torch.bool, device=device),
                log_p_anchor=z, log_p_alt=z.clone(),
                step_active=torch.zeros(B, T, dtype=torch.bool, device=device),
                entropy=entropy,
            )

        return DecodePlan(
            anchors=torch.stack(anchors_l, dim=1),
            altitudes=torch.stack(alts_l, dim=1),
            served=torch.stack(served_l, dim=1),
            log_p_anchor=torch.stack(logpa_l, dim=1),
            log_p_alt=torch.stack(logp_alt_l, dim=1),
            step_active=torch.stack(active_l, dim=1),
            entropy=entropy,
        )
