"""
Channel Model for UAV-IoT communication.

Implements line-of-sight (LoS) path loss and Shannon data rate computation
for both 2D (fixed altitude) and 3D (variable altitude) scenarios.

Paper A: |g_ij|² = β / d_ij²  (free-space path loss)
Data rate: R_ij = B · log2(1 + P·|g_ij|² / σ²)
"""

import numpy as np
import torch
from typing import Union


class ChannelModel:
    """
    2D Channel model with fixed UAV altitude (Paper A baseline).

    Computes path loss and achievable data rate between UAV j and IoT node i
    assuming free-space LoS propagation and fixed altitude.

    Args:
        bandwidth: Channel bandwidth in Hz (default: 2 MHz).
        noise_power_dbm: Noise power in dBm (default: -110 dBm).
        carrier_freq: Carrier frequency in Hz (default: 2.4 GHz).
        path_loss_exp: Path loss exponent (default: 2 for free-space).
        fixed_altitude: UAV altitude in meters (default: 20 m).
        transmit_power: UAV transmit power in W (default: 0.5 W).
    """

    def __init__(
        self,
        bandwidth: float = 2e6,
        noise_power_dbm: float = -110.0,
        carrier_freq: float = 2.4e9,
        path_loss_exp: float = 2.0,
        fixed_altitude: float = 20.0,
        transmit_power: float = 0.5,
    ):
        self.bandwidth = bandwidth
        self.noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)  # convert dBm to W
        self.carrier_freq = carrier_freq
        self.path_loss_exp = path_loss_exp
        self.fixed_altitude = fixed_altitude
        self.transmit_power = transmit_power

        # Reference path loss constant β (at 1m reference distance)
        # β = (c / (4π f_c))^2 — free-space path loss at reference distance
        c = 3e8  # speed of light
        self.beta = (c / (4 * np.pi * carrier_freq)) ** 2

    def compute_distance(
        self,
        uav_pos: Union[np.ndarray, torch.Tensor],
        node_pos: Union[np.ndarray, torch.Tensor],
    ) -> Union[float, torch.Tensor]:
        """
        Compute distance between UAV and IoT node (2D + fixed altitude).

        Args:
            uav_pos: UAV horizontal position [x, y] or batch (..., 2).
            node_pos: Node horizontal position [x, y] or batch (..., 2).

        Returns:
            3D distance (including fixed altitude offset).
        """
        if isinstance(uav_pos, torch.Tensor):
            horizontal_dist_sq = torch.sum((uav_pos[..., :2] - node_pos[..., :2]) ** 2, dim=-1)
            return torch.sqrt(horizontal_dist_sq + self.fixed_altitude ** 2)
        else:
            horizontal_dist_sq = np.sum((uav_pos[..., :2] - node_pos[..., :2]) ** 2, axis=-1)
            return np.sqrt(horizontal_dist_sq + self.fixed_altitude ** 2)

    def compute_path_loss(
        self, distance: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute channel power gain: |g|² = β / d^α

        Args:
            distance: Distance in meters (scalar, array, or tensor).

        Returns:
            Channel power gain (linear scale).
        """
        if isinstance(distance, torch.Tensor):
            return self.beta / (distance ** self.path_loss_exp + 1e-10)
        else:
            return self.beta / (np.maximum(distance, 1e-5) ** self.path_loss_exp)

    def compute_data_rate(
        self, distance: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute achievable data rate: R = B · log2(1 + P·|g|²/σ²)

        Args:
            distance: Distance in meters.

        Returns:
            Data rate in bits/second.
        """
        gain = self.compute_path_loss(distance)
        snr = self.transmit_power * gain / self.noise_power_w

        if isinstance(snr, torch.Tensor):
            return self.bandwidth * torch.log2(1 + snr)
        else:
            return self.bandwidth * np.log2(1 + snr)

    def compute_data_rate_matrix(
        self,
        uav_positions: np.ndarray,
        node_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute data rate matrix between all UAVs and all nodes.

        Args:
            uav_positions: (M, 2) UAV positions.
            node_positions: (N, 2) node positions.

        Returns:
            (M, N) matrix of data rates in bits/second.
        """
        M = uav_positions.shape[0]
        N = node_positions.shape[0]

        # Broadcast: (M, 1, 2) - (1, N, 2) → (M, N, 2)
        diff = uav_positions[:, None, :2] - node_positions[None, :, :2]
        horizontal_dist_sq = np.sum(diff ** 2, axis=-1)  # (M, N)
        distances = np.sqrt(horizontal_dist_sq + self.fixed_altitude ** 2)  # (M, N)

        return self.compute_data_rate(distances)


class ChannelModel3D(ChannelModel):
    """
    3D Channel model with variable UAV altitude (ATOM-3D extension).

    Uses full 3D Euclidean distance:
        d_ij(t) = sqrt((x_j - x_i)² + (y_j - y_i)² + (H_j - z_i)²)

    where H_j is the UAV altitude and z_i is the node elevation.
    """

    def __init__(self, beamwidth_deg: float = 60.0, **kwargs):
        # Don't use fixed_altitude — altitude is variable
        kwargs.pop('fixed_altitude', None)
        super().__init__(fixed_altitude=0.0, **kwargs)
        # Half-beamwidth theta of the downward antenna cone (Paper B eq. 23e).
        self.beamwidth_deg = beamwidth_deg
        self.tan_theta = float(np.tan(np.radians(beamwidth_deg)))

    # ------------------------------------------------------------------
    # Coverage footprint (Paper B eq. 23e) and QoS serve test (spec A3/A4)
    # ------------------------------------------------------------------
    def coverage_radius(
        self,
        uav_altitude: Union[float, np.ndarray, torch.Tensor],
        node_elevation: Union[float, np.ndarray, torch.Tensor] = 0.0,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Footprint radius r = (H - z) * tan(theta) of the coverage cone.

        A negative vertical clearance (UAV at/below the node) yields radius 0.
        """
        clearance = uav_altitude - node_elevation
        if isinstance(clearance, torch.Tensor):
            return torch.clamp(clearance, min=0.0) * self.tan_theta
        return np.maximum(clearance, 0.0) * self.tan_theta

    def covered(
        self,
        horizontal_dist: Union[float, np.ndarray, torch.Tensor],
        uav_altitude: Union[float, np.ndarray, torch.Tensor],
        node_elevation: Union[float, np.ndarray, torch.Tensor] = 0.0,
    ) -> Union[bool, np.ndarray, torch.Tensor]:
        """Is a node inside the coverage cone?  rho <= (H - z)*tan(theta).

        Args:
            horizontal_dist: ground-plane distance rho between hover (x,y) and node.
            uav_altitude:    hover altitude H.
            node_elevation:  node ground elevation z_i.

        Returns bool / boolean array matching the broadcast shape.
        """
        return horizontal_dist <= self.coverage_radius(uav_altitude, node_elevation)

    def serves(
        self,
        horizontal_dist: Union[float, np.ndarray, torch.Tensor],
        uav_altitude: Union[float, np.ndarray, torch.Tensor],
        node_elevation: Union[float, np.ndarray, torch.Tensor] = 0.0,
        R_min: float = 0.0,
    ) -> Union[bool, np.ndarray, torch.Tensor]:
        """Node is served iff it is inside the cone AND meets the QoS floor R_min.

        Combines the geometric footprint test (A3) with the per-node rate floor
        (Pareto knob). With R_min = 0 this reduces to ``covered``.
        """
        in_cone = self.covered(horizontal_dist, uav_altitude, node_elevation)
        if R_min <= 0.0:
            return in_cone
        # 3D distance straight from the ground offset rho: d = sqrt(rho^2 + (H-z)^2).
        clearance = uav_altitude - node_elevation
        if isinstance(horizontal_dist, torch.Tensor):
            dist_3d = torch.sqrt(horizontal_dist ** 2 + clearance ** 2 + 1e-10)
            rate = self.compute_data_rate(dist_3d)
            return in_cone & (rate >= R_min)
        dist_3d = np.sqrt(np.asarray(horizontal_dist) ** 2 + np.asarray(clearance) ** 2 + 1e-10)
        rate = self.compute_data_rate(dist_3d)
        meets = rate >= R_min
        return np.logical_and(in_cone, meets) if np.ndim(meets) else bool(in_cone and meets)

    def compute_distance_3d(
        self,
        uav_pos: Union[np.ndarray, torch.Tensor],
        uav_altitude: Union[float, np.ndarray, torch.Tensor],
        node_pos: Union[np.ndarray, torch.Tensor],
        node_elevation: Union[float, np.ndarray, torch.Tensor] = 0.0,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute full 3D distance between UAV and IoT node.

        Args:
            uav_pos: UAV horizontal position (..., 2).
            uav_altitude: UAV altitude H_j (scalar or matching batch).
            node_pos: Node horizontal position (..., 2).
            node_elevation: Node ground elevation z_i (scalar or matching batch).

        Returns:
            3D Euclidean distance.
        """
        if isinstance(uav_pos, torch.Tensor):
            horizontal_dist_sq = torch.sum((uav_pos[..., :2] - node_pos[..., :2]) ** 2, dim=-1)
            vertical_dist_sq = (uav_altitude - node_elevation) ** 2
            return torch.sqrt(horizontal_dist_sq + vertical_dist_sq + 1e-10)
        else:
            horizontal_dist_sq = np.sum((uav_pos[..., :2] - node_pos[..., :2]) ** 2, axis=-1)
            vertical_dist_sq = (uav_altitude - node_elevation) ** 2
            return np.sqrt(horizontal_dist_sq + vertical_dist_sq + 1e-10)

    def compute_data_rate_3d(
        self,
        uav_pos: Union[np.ndarray, torch.Tensor],
        uav_altitude: Union[float, np.ndarray, torch.Tensor],
        node_pos: Union[np.ndarray, torch.Tensor],
        node_elevation: Union[float, np.ndarray, torch.Tensor] = 0.0,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute data rate using full 3D distance."""
        distance = self.compute_distance_3d(uav_pos, uav_altitude, node_pos, node_elevation)
        return self.compute_data_rate(distance)

    def compute_data_rate_matrix_3d(
        self,
        uav_positions: np.ndarray,
        uav_altitudes: np.ndarray,
        node_positions: np.ndarray,
        node_elevations: np.ndarray,
    ) -> np.ndarray:
        """
        Compute 3D data rate matrix.

        Args:
            uav_positions: (M, 2) UAV horizontal positions.
            uav_altitudes: (M,) UAV altitudes.
            node_positions: (N, 2) node horizontal positions.
            node_elevations: (N,) node ground elevations.

        Returns:
            (M, N) data rate matrix in bits/second.
        """
        # (M, 1, 2) - (1, N, 2) → (M, N)
        diff = uav_positions[:, None, :2] - node_positions[None, :, :2]
        horizontal_dist_sq = np.sum(diff ** 2, axis=-1)

        # (M, 1) - (1, N) → (M, N)
        vertical_dist_sq = (uav_altitudes[:, None] - node_elevations[None, :]) ** 2

        distances = np.sqrt(horizontal_dist_sq + vertical_dist_sq + 1e-10)
        return self.compute_data_rate(distances)
