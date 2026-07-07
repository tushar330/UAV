"""
IoT Environment for Multi-UAV Data Collection.

Implements both 2D (Paper A baseline) and 3D (ATOM-3D extension) environments.
Generates random IoT node placements with data demands, manages UAV state
(position, remaining capacity, remaining energy), and tracks visit history.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class UAVState:
    """Tracks the dynamic state of a single UAV during trajectory execution."""
    position: np.ndarray          # [x, y] or [x, y, z]
    remaining_energy: float       # mAh remaining
    remaining_capacity: float     # MB remaining
    visited_nodes: List[int] = field(default_factory=list)
    total_distance: float = 0.0   # cumulative flight distance (m)
    total_data_collected: float = 0.0  # cumulative data collected (MB)
    current_altitude: float = 20.0     # current altitude (m), used in 3D


class IoTEnvironment2D:
    """
    2D IoT environment (Paper A baseline).

    Generates N IoT nodes at random (x, y) positions within a rectangular area.
    Each node has a data demand D_i ~ Uniform(Di_min, Di_max).
    UAVs operate at a fixed altitude.

    Args:
        N: Number of IoT nodes.
        area_width: Deployment area width in meters.
        area_height: Deployment area height in meters.
        Di_min: Minimum data demand per node (MB).
        Di_max: Maximum data demand per node (MB).
        data_center: (x, y) position of the data center / depot.
        flight_height: Fixed flight altitude (m).
        E_max: UAV battery capacity (mAh).
        C_max: UAV storage capacity (MB).
        M: Number of UAVs (0 = auto-determine).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        N: int = 100,
        area_width: float = 1000.0,
        area_height: float = 1000.0,
        Di_min: float = 0.2,
        Di_max: float = 1.5,
        data_center: Tuple[float, float] = (0.0, 0.0),
        flight_height: float = 20.0,
        E_max: float = 2000.0,
        C_max: float = 500.0,
        M: int = 3,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.area_width = area_width
        self.area_height = area_height
        self.Di_min = Di_min
        self.Di_max = Di_max
        self.data_center = np.array(data_center, dtype=np.float32)
        self.flight_height = flight_height
        self.E_max = E_max
        self.C_max = C_max
        self.M = M
        self.rng = np.random.RandomState(seed)

        # Will be populated by reset()
        self.node_positions: Optional[np.ndarray] = None  # (N, 2)
        self.node_demands: Optional[np.ndarray] = None    # (N,)
        self.uav_states: List[UAVState] = []
        self.visited_mask: Optional[np.ndarray] = None    # (N,) bool

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Generate a new random scenario.

        Returns:
            dict with keys:
                'node_features': torch.Tensor (N, 3) — [x, y, D] per node
                'depot': torch.Tensor (2,) — data center position
                'num_nodes': int
                'num_uavs': int
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Generate random node positions centered around the area
        self.node_positions = np.stack([
            self.rng.uniform(-self.area_width / 2, self.area_width / 2, self.N),
            self.rng.uniform(-self.area_height / 2, self.area_height / 2, self.N),
        ], axis=1).astype(np.float32)  # (N, 2)

        # Generate random data demands
        self.node_demands = self.rng.uniform(
            self.Di_min, self.Di_max, self.N
        ).astype(np.float32)  # (N,)

        # Initialize UAV states — all start at data center
        self.uav_states = []
        for _ in range(self.M):
            self.uav_states.append(UAVState(
                position=self.data_center.copy(),
                remaining_energy=self.E_max,
                remaining_capacity=self.C_max,
                current_altitude=self.flight_height,
            ))

        # Track which nodes have been visited
        self.visited_mask = np.zeros(self.N, dtype=bool)

        return self._get_observation()

    def _get_observation(self) -> Dict:
        """Build the observation dict for the current state."""
        # Node features: [x, y, D_i]
        features = np.column_stack([
            self.node_positions,        # (N, 2)
            self.node_demands[:, None], # (N, 1)
        ])  # (N, 3)

        return {
            'node_features': torch.from_numpy(features),          # (N, 3)
            'depot': torch.from_numpy(self.data_center),          # (2,)
            'num_nodes': self.N,
            'num_uavs': self.M,
            'visited_mask': torch.from_numpy(self.visited_mask),  # (N,)
            'node_positions': torch.from_numpy(self.node_positions),
            'node_demands': torch.from_numpy(self.node_demands),
        }

    def compute_distance_2d(self, pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """Euclidean distance between two 2D points, including fixed altitude."""
        horizontal = np.linalg.norm(pos_a[:2] - pos_b[:2])
        return float(np.sqrt(horizontal ** 2 + self.flight_height ** 2))

    def step_uav(self, uav_idx: int, node_idx: int) -> Dict:
        """
        Move UAV to visit a specific node.

        Args:
            uav_idx: Which UAV to move.
            node_idx: Which node to visit.

        Returns:
            dict with step results: distance, data_collected, energy_used, feasible
        """
        uav = self.uav_states[uav_idx]
        node_pos = self.node_positions[node_idx]
        node_demand = self.node_demands[node_idx]

        # Compute flight distance (2D horizontal + fixed altitude)
        distance = float(np.linalg.norm(uav.position[:2] - node_pos[:2]))

        # Update UAV state
        uav.position[:2] = node_pos.copy()
        uav.total_distance += distance
        uav.visited_nodes.append(node_idx)

        # Data collection (limited by remaining capacity)
        data_collected = min(node_demand, uav.remaining_capacity)
        uav.remaining_capacity -= data_collected
        uav.total_data_collected += data_collected

        # Mark node as visited
        self.visited_mask[node_idx] = True

        return {
            'distance': distance,
            'data_collected': data_collected,
            'feasible': uav.remaining_capacity >= 0,
        }

    def get_return_distance(self, uav_idx: int) -> float:
        """Distance for UAV to return to data center."""
        uav = self.uav_states[uav_idx]
        return float(np.linalg.norm(uav.position[:2] - self.data_center[:2]))

    @staticmethod
    def generate_batch(
        batch_size: int, N: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of random IoT scenarios for training.

        Positions are generated in real coordinates [-area_width/2, area_width/2]
        to match the distribution used by reset() during evaluation.

        Returns:
            node_features: (batch_size, N, 3) — [x, y, D] in real coordinates
            depot: (batch_size, 2) — depot position (data center at origin)
        """
        rng = np.random.RandomState(kwargs.get('seed', None))
        Di_min = kwargs.get('Di_min', 0.2)
        Di_max = kwargs.get('Di_max', 1.5)
        area_width = kwargs.get('area_width', 1000.0)
        area_height = kwargs.get('area_height', 1000.0)

        # Positions in real coordinate range (matches env.reset())
        x = rng.uniform(-area_width / 2, area_width / 2, (batch_size, N, 1)).astype(np.float32)
        y = rng.uniform(-area_height / 2, area_height / 2, (batch_size, N, 1)).astype(np.float32)
        demands = rng.uniform(Di_min, Di_max, (batch_size, N, 1)).astype(np.float32)

        node_features = np.concatenate([x, y, demands], axis=-1)  # (B, N, 3)
        depot = np.zeros((batch_size, 2), dtype=np.float32)       # origin

        return torch.from_numpy(node_features), torch.from_numpy(depot)


class IoTEnvironment3D(IoTEnvironment2D):
    """
    3D IoT environment (ATOM-3D extension).

    Extends the 2D environment with:
    - Node elevation z_i ~ Uniform(zi_min, zi_max)
    - UAV altitude tracking and constraints [H_min, H_max]
    - 3D distance computation for channel model

    Args:
        zi_min: Minimum node elevation (m).
        zi_max: Maximum node elevation (m).
        H_min: Minimum UAV altitude (m).
        H_max: Maximum UAV altitude (m).
        All other args inherited from IoTEnvironment2D.
    """

    def __init__(
        self,
        zi_min: float = 0.0,
        zi_max: float = 50.0,
        H_min: float = 20.0,
        H_max: float = 150.0,
        priority_enabled: bool = False,
        priority_class_probs: Optional[List[float]] = None,
        priority_weights: Optional[List[float]] = None,
        priority_rmin: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.zi_min = zi_min
        self.zi_max = zi_max
        self.H_min = H_min
        self.H_max = H_max

        # Priority / criticality model (config: priority.*). Class order: [high, medium, low].
        # Defaults preserve uniform behaviour (weight 1, no R_min floor) when disabled.
        self.priority_enabled = priority_enabled
        self.priority_class_probs = np.asarray(
            priority_class_probs if priority_class_probs is not None else [0.1, 0.3, 0.6],
            dtype=np.float64,
        )
        self.priority_weights = np.asarray(
            priority_weights if priority_weights is not None else [5.0, 2.0, 1.0],
            dtype=np.float32,
        )
        self.priority_rmin = np.asarray(
            priority_rmin if priority_rmin is not None else [4.2e7, 2.0e7, 0.0],
            dtype=np.float32,
        )

        self.node_elevations: Optional[np.ndarray] = None       # (N,)
        self.node_priority_class: Optional[np.ndarray] = None   # (N,) int in {0,1,2}
        self.node_weights: Optional[np.ndarray] = None          # (N,) importance w_i
        self.node_rmin: Optional[np.ndarray] = None             # (N,) per-node QoS floor (bits/s)

    def _assign_priorities(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a criticality class per node and map it to (class_idx, weight, R_min).

        When priority is disabled, every node is class 0 with weight 1 and a zero R_min
        floor — identical to the pre-priority environment (serve-all, no QoS constraint).
        """
        if not self.priority_enabled:
            cls = np.zeros(N, dtype=np.int64)
            return cls, np.ones(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
        probs = self.priority_class_probs / self.priority_class_probs.sum()
        cls = self.rng.choice(len(probs), size=N, p=probs).astype(np.int64)
        return cls, self.priority_weights[cls].astype(np.float32), self.priority_rmin[cls].astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Generate a new 3D random scenario."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Generate 2D positions
        self.node_positions = np.stack([
            self.rng.uniform(-self.area_width / 2, self.area_width / 2, self.N),
            self.rng.uniform(-self.area_height / 2, self.area_height / 2, self.N),
        ], axis=1).astype(np.float32)

        # Generate node elevations
        self.node_elevations = self.rng.uniform(
            self.zi_min, self.zi_max, self.N
        ).astype(np.float32)

        # Generate data demands
        self.node_demands = self.rng.uniform(
            self.Di_min, self.Di_max, self.N
        ).astype(np.float32)

        # Assign per-node priority class -> importance weight + QoS rate floor
        self.node_priority_class, self.node_weights, self.node_rmin = self._assign_priorities(self.N)

        # Initialize UAVs at data center position and altitude
        initial_altitude = float(self.data_center[1]) if len(self.data_center) > 2 else self.flight_height
        if hasattr(self, '_dc_z'):
            initial_altitude = self._dc_z
        else:
            self._dc_z = self.flight_height
            initial_altitude = self._dc_z

        self.uav_states = []
        for _ in range(self.M):
            self.uav_states.append(UAVState(
                position=np.array([self.data_center[0], self.data_center[1]], dtype=np.float32),
                remaining_energy=self.E_max,
                remaining_capacity=self.C_max,
                current_altitude=initial_altitude,
            ))

        self.visited_mask = np.zeros(self.N, dtype=bool)

        return self._get_observation()

    def _get_observation(self) -> Dict:
        """Build 3D observation with node elevations."""
        # Node features: [x, y, z, D_i]
        features = np.column_stack([
            self.node_positions,            # (N, 2)
            self.node_elevations[:, None],  # (N, 1)
            self.node_demands[:, None],     # (N, 1)
        ])  # (N, 4)

        return {
            'node_features': torch.from_numpy(features),          # (N, 4)
            'depot': torch.from_numpy(self.data_center),
            'num_nodes': self.N,
            'num_uavs': self.M,
            'visited_mask': torch.from_numpy(self.visited_mask),
            'node_positions': torch.from_numpy(self.node_positions),
            'node_elevations': torch.from_numpy(self.node_elevations),
            'node_demands': torch.from_numpy(self.node_demands),
            'node_priority_class': torch.from_numpy(self.node_priority_class),
            'node_weights': torch.from_numpy(self.node_weights),
            'node_rmin': torch.from_numpy(self.node_rmin),
        }

    def compute_distance_3d(
        self, pos_a: np.ndarray, alt_a: float,
        pos_b: np.ndarray, alt_b: float
    ) -> float:
        """Full 3D Euclidean distance between two positions."""
        horizontal = np.linalg.norm(pos_a[:2] - pos_b[:2])
        vertical = abs(alt_a - alt_b)
        return float(np.sqrt(horizontal ** 2 + vertical ** 2))

    def step_uav(self, uav_idx: int, node_idx: int, target_altitude: Optional[float] = None) -> Dict:
        """
        Move UAV to visit a node, optionally adjusting altitude.

        Args:
            uav_idx: Which UAV.
            node_idx: Which node to visit.
            target_altitude: UAV altitude when visiting this node.
                If None, maintains current altitude.

        Returns:
            dict with: distance, data_collected, altitude_change, feasible
        """
        uav = self.uav_states[uav_idx]
        node_pos = self.node_positions[node_idx]
        node_demand = self.node_demands[node_idx]

        # Determine target altitude
        if target_altitude is None:
            target_altitude = uav.current_altitude
        target_altitude = np.clip(target_altitude, self.H_min, self.H_max)

        # 3D distance
        distance = self.compute_distance_3d(
            uav.position, uav.current_altitude,
            node_pos, target_altitude
        )
        altitude_change = abs(target_altitude - uav.current_altitude)

        # Update state
        uav.position[:2] = node_pos.copy()
        uav.current_altitude = target_altitude
        uav.total_distance += distance
        uav.visited_nodes.append(node_idx)

        # Data collection
        data_collected = min(node_demand, uav.remaining_capacity)
        uav.remaining_capacity -= data_collected
        uav.total_data_collected += data_collected

        self.visited_mask[node_idx] = True

        return {
            'distance': distance,
            'data_collected': data_collected,
            'altitude_change': altitude_change,
            'feasible': uav.remaining_capacity >= 0,
        }

    @staticmethod
    def generate_batch(
        batch_size: int, N: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of 3D IoT scenarios.

        Positions and elevations are generated in real coordinates to match
        the distribution used by reset() during evaluation.

        Returns:
            node_features: (batch_size, N, 4) — [x, y, z, D] in real coordinates
            depot: (batch_size, 3) — depot position (data center at origin)
        """
        rng = np.random.RandomState(kwargs.get('seed', None))
        Di_min = kwargs.get('Di_min', 0.2)
        Di_max = kwargs.get('Di_max', 1.5)
        zi_min = kwargs.get('zi_min', 0.0)
        zi_max = kwargs.get('zi_max', 50.0)
        area_width = kwargs.get('area_width', 1000.0)
        area_height = kwargs.get('area_height', 1000.0)

        # Positions in real coordinate range (matches env.reset())
        x = rng.uniform(-area_width / 2, area_width / 2, (batch_size, N, 1)).astype(np.float32)
        y = rng.uniform(-area_height / 2, area_height / 2, (batch_size, N, 1)).astype(np.float32)
        elevations = rng.uniform(zi_min, zi_max, (batch_size, N, 1)).astype(np.float32)
        demands = rng.uniform(Di_min, Di_max, (batch_size, N, 1)).astype(np.float32)

        node_features = np.concatenate([x, y, elevations, demands], axis=-1)
        depot = np.zeros((batch_size, 3), dtype=np.float32)

        return torch.from_numpy(node_features), torch.from_numpy(depot)

    @staticmethod
    def generate_priorities(
        batch_size: int, N: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate per-node priority class / weight / R_min for a training batch.

        Non-breaking: call alongside generate_batch() only when priority is enabled.
        When disabled, returns uniform tensors (class 0, weight 1, zero R_min) so the
        trainer behaves exactly as before.

        Returns three (batch_size, N) tensors:
            class_idx (long), weight w_i (float), R_min in bits/s (float).
        """
        rng = np.random.RandomState(kwargs.get('seed', None))
        enabled = kwargs.get('priority_enabled', False)
        if not enabled:
            cls = np.zeros((batch_size, N), dtype=np.int64)
            w = np.ones((batch_size, N), dtype=np.float32)
            r = np.zeros((batch_size, N), dtype=np.float32)
            return torch.from_numpy(cls), torch.from_numpy(w), torch.from_numpy(r)

        class_probs = np.asarray(kwargs.get('priority_class_probs', [0.1, 0.3, 0.6]), dtype=np.float64)
        weights_map = np.asarray(kwargs.get('priority_weights', [5.0, 2.0, 1.0]), dtype=np.float32)
        rmin_map = np.asarray(kwargs.get('priority_rmin', [4.2e7, 2.0e7, 0.0]), dtype=np.float32)
        probs = class_probs / class_probs.sum()
        cls = rng.choice(len(probs), size=(batch_size, N), p=probs).astype(np.int64)
        w = weights_map[cls].astype(np.float32)
        r = rmin_map[cls].astype(np.float32)
        return torch.from_numpy(cls), torch.from_numpy(w), torch.from_numpy(r)
