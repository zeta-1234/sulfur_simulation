from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection

    from sulfur_simulation.scattering_calculation import SimulationParameters


def animate_particle_positions_square(
    all_positions: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
    lattice_dimension: tuple[int, int],
    timesteps: np.ndarray,
    lattice_spacing: float = 2.5,
) -> animation.FuncAnimation:
    """Animate particle positions with lattice sites as blue stars."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lattice_spacing, lattice_dimension[0] * lattice_spacing)
    ax.set_ylim(-lattice_spacing, lattice_dimension[1] * lattice_spacing)
    ax.set_aspect("equal")
    ax.set_title("Particle Simulation")

    # Draw lattice sites as blue stars once
    lattice_x, lattice_y = np.meshgrid(
        np.arange(lattice_dimension[0]) * lattice_spacing,
        np.arange(lattice_dimension[1]) * lattice_spacing,
    )
    ax.scatter(
        lattice_x.ravel(),
        lattice_y.ravel(),
        color="aqua",
        marker=".",
        s=5,
        zorder=0,
        label="Sites",
    )

    # Particle scatter on top of lattice stars
    particle_scatter: PathCollection = ax.scatter(
        [], [], color="red", s=20, edgecolors="black", zorder=1
    )
    ax.legend(["Particles", "Sites"], loc="lower right")

    def update(frame: int) -> tuple[PathCollection]:
        occupancy = all_positions[frame]
        rows, cols = np.nonzero(occupancy)

        # Convert lattice indices to physical positions
        x = cols * lattice_spacing
        y = rows * lattice_spacing

        particle_scatter.set_offsets(np.c_[x, y])
        ax.set_title(f"Timestep: {frame}")
        return (particle_scatter,)

    return animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=5000 / len(timesteps),
        blit=False,
        repeat=True,
    )


def animate_particle_positions_hexagonal(
    all_positions: np.ndarray,  # shape (timesteps, rows, cols), dtype=bool
    lattice_dimension: tuple[int, int],
    timesteps: np.ndarray,
    lattice_spacing: float = 2.5,
) -> animation.FuncAnimation:
    """Animate particle positions on a hexagonal close-packed (HCP) lattice."""
    fig, ax = plt.subplots(figsize=(6, 6))

    dx = lattice_spacing
    dy = lattice_spacing * np.sqrt(3) / 2

    # Precompute lattice coordinates
    lattice_x = np.zeros(lattice_dimension[0] * lattice_dimension[1])
    lattice_y = np.zeros_like(lattice_x)

    index = 0
    for row in range(lattice_dimension[1]):
        for col in range(lattice_dimension[0]):
            lattice_x[index] = col * dx + (dx / 2) * row
            lattice_y[index] = row * dy
            index += 1

    # Set axis limits based on lattice extent
    ax.set_xlim(lattice_x.min() - dx, lattice_x.max() + dx)
    ax.set_ylim(lattice_y.min() - dy, lattice_y.max() + dy)
    ax.set_aspect("equal")

    # Hide tick labels but keep axis lines
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw lattice sites
    ax.scatter(
        lattice_x, lattice_y, color="aqua", marker=".", s=5, zorder=0, label="Sites"
    )

    # Particle scatter
    particle_scatter: PathCollection = ax.scatter(
        [], [], color="red", s=20, edgecolors="black", zorder=1
    )

    # Legend
    ax.legend(["Particles", "Sites"], loc="upper right")

    def update(frame: int) -> tuple[PathCollection]:
        occupancy = all_positions[frame]
        rows, cols = np.nonzero(occupancy)
        x = cols * dx + (dx / 2) * rows
        y = rows * dy
        particle_scatter.set_offsets(np.c_[x, y])
        ax.set_title(f"Timestep: {frame}")
        return (particle_scatter,)

    return animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=5000 / len(timesteps),
        blit=False,
        repeat=True,
    )


def get_timeframe_str(
    positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    timestep: int,
    params: SimulationParameters,
) -> str:
    """Print a preview of positions at specified timestep. Works for small grids."""
    dimension = params.lattice_dimension
    max_dimension = 100
    if dimension[0] > max_dimension or dimension[1] > max_dimension:
        return "Lattice too large to print"
    grid = positions[timestep].reshape(params.lattice_dimension)

    green = "\033[92m"
    gray = "\033[0m"

    def colorize(val: bool) -> str:  # noqa: FBT001
        return f"{green}o{gray}" if val else "."

    lines: list[str] = []
    for row in grid:
        colored_row = [colorize(v) for v in row]
        lines.append(" ".join(colored_row))

    return "\n".join(lines)
