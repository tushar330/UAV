"""
Visualizer for ATOM-3D Trajectories, Altitude Profiles, and Training Progress.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


class TrajectoryVisualizer:
    """
    Visualization tools for 2D and 3D UAV routes.
    """
    def __init__(self, output_dir: str = "figures/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_2d_trajectories(
        self,
        node_positions: np.ndarray,
        routes: List[np.ndarray],
        depot: np.ndarray,
        filename: str = "trajectory_2d.png"
    ):
        """
        Plot 2D trajectories of UAVs.

        Args:
            node_positions: (N, 2) array of node coordinates.
            routes: List of (K_j, 2) paths taken by each UAV.
            depot: (2,) depot position.
            filename: file to save plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot nodes
        plt.scatter(node_positions[:, 0], node_positions[:, 1], c='blue', alpha=0.6, label='IoT Nodes', edgecolors='k')
        
        # Plot depot
        plt.scatter(depot[0], depot[1], c='red', marker='D', s=100, label='Depot/Data Center', edgecolors='k')

        # Plot each UAV route
        colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
        for idx, route in enumerate(routes):
            color = colors[idx % len(colors)]
            plt.plot(route[:, 0], route[:, 1], color=color, linewidth=2, marker='o', label=f'UAV {idx+1}')
            
            # Draw arrows to show direction
            for i in range(len(route) - 1):
                dx = route[i+1, 0] - route[i, 0]
                dy = route[i+1, 1] - route[i, 1]
                plt.arrow(route[i, 0], route[i, 1], dx*0.5, dy*0.5, head_width=15, head_length=15, fc=color, ec=color)

        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.title('Multi-UAV 2D Route Plan')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 2D trajectory plot to {out_path}")

    def plot_3d_trajectories(
        self,
        node_positions: np.ndarray,
        node_elevations: np.ndarray,
        routes: List[np.ndarray],
        altitudes: List[np.ndarray],
        depot: np.ndarray,
        filename: str = "trajectory_3d.png"
    ):
        """
        Plot 3D trajectories of UAVs.

        Args:
            node_positions: (N, 2) horizontal positions.
            node_elevations: (N,) ground elevations.
            routes: List of (K_j, 2) horizontal paths.
            altitudes: List of (K_j,) flight altitudes.
            depot: (2,) depot position.
            filename: file to save plot.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_elevations, c='blue', alpha=0.6, label='IoT Nodes', edgecolors='k')
        
        # Plot depot (assumed DC altitude = 20)
        ax.scatter(depot[0], depot[1], 20.0, c='red', marker='D', s=100, label='Depot/Data Center', edgecolors='k')

        colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
        for idx, (route, alts) in enumerate(zip(routes, altitudes)):
            color = colors[idx % len(colors)]
            ax.plot(route[:, 0], route[:, 1], alts, color=color, linewidth=2.5, marker='o', label=f'UAV {idx+1}')

        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_zlabel('Altitude / Elevation (m)')
        ax.set_title('Multi-UAV 3D Altitude-Variable Routes')
        ax.legend()
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D trajectory plot to {out_path}")

    def plot_altitude_profile(
        self,
        altitudes: List[np.ndarray],
        filename: str = "altitude_profile.png"
    ):
        """
        Plot UAV flight altitude profiles over route visit indexes.
        """
        plt.figure(figsize=(10, 5))
        colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
        
        for idx, alts in enumerate(altitudes):
            plt.plot(alts, marker='o', linestyle='-', color=colors[idx % len(colors)], label=f'UAV {idx+1}')
            
        plt.xlabel('Route Sequence Index')
        plt.ylabel('Altitude (m)')
        plt.title('UAV Altitude Profiles')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved altitude profile plot to {out_path}")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_curves.png"
    ):
        """
        Plot actor/critic losses, reward, and data collections vs training epochs.
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['reward']) + 1)

        # Reward
        axs[0, 0].plot(epochs, history['reward'], color='blue')
        axs[0, 0].set_title('Average Reward vs Epoch')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)

        # Actor Loss
        axs[0, 1].plot(epochs, history['actor_loss'], color='red')
        axs[0, 1].set_title('Actor Loss vs Epoch')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)

        # Critic Loss
        axs[1, 0].plot(epochs, history['critic_loss'], color='purple')
        axs[1, 0].set_title('Critic Loss vs Epoch')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True)

        # Data vs Energy
        axs[1, 1].plot(epochs, history['avg_data'], label='Data Collected (MB)', color='green')
        ax2 = axs[1, 1].twinx()
        ax2.plot(epochs, np.array(history['avg_energy'])/3600.0, label='Energy Spent (Wh)', color='orange', linestyle='--')
        axs[1, 1].set_title('Data Collected & Energy Expended')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Data (MB)', color='green')
        ax2.set_ylabel('Energy (Wh)', color='orange')
        axs[1, 1].grid(True)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved training curves to {out_path}")

    def plot_pareto_curve(
        self,
        data_collected_list: List[float],
        energy_wh_list: List[float],
        lambdas: List[float],
        filename: str = "pareto_curve.png"
    ):
        """
        Plot data throughput vs energy Pareto efficiency curve.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(energy_wh_list, data_collected_list, color='red', s=80, edgecolors='k', zorder=3)
        plt.plot(energy_wh_list, data_collected_list, color='blue', linestyle='-', alpha=0.5)

        for i, txt in enumerate(lambdas):
            plt.annotate(f"λ={txt}", (energy_wh_list[i], data_collected_list[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

        plt.xlabel('Total Energy Consumption (Wh)')
        plt.ylabel('Total Collected Data (MB)')
        plt.title('Pareto Frontier (Data Collected vs Energy Spent)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Pareto curve to {out_path}")
