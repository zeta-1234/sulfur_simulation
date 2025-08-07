from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection

    from sulfur_simulation.scattering_calculation import SimulationParameters


def animate_particle_positions(
    all_positions: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
    lattice_dimension: tuple[int, int],
    timesteps: np.ndarray,
    lattice_spacing: float = 2.5,
) -> animation.FuncAnimation:
    """Animate particle positions from boolean occupancy arrays over time."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-lattice_spacing, lattice_dimension[0] * lattice_spacing)
    ax.set_ylim(-lattice_spacing, lattice_dimension[1] * lattice_spacing)
    ax.set_aspect("equal")
    ax.set_title("Particle Simulation")

    # Initially no points
    particle_scatter: PathCollection = ax.scatter(
        [], [], color="red", s=40, edgecolors="black"
    )
    ax.legend(["Particles"], loc="upper right")

    def update(frame: int) -> tuple[PathCollection]:
        occupancy = all_positions[frame]
        rows, cols = np.nonzero(occupancy)

        # Convert lattice indices to physical positions (x=cols, y=rows)
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


def print_timeframe(
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
