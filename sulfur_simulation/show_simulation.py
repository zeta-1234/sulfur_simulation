from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from sulfur_simulation.scattering_calculation import JUMP_DIRECTIONS, SimulationResult
from sulfur_simulation.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import NDArray

    from sulfur_simulation.scattering_calculation import SimulationParameters


def animate_particle_positions(
    all_positions: NDArray[np.bool_],
    lattice_dimension: tuple[int, int],
    timesteps: NDArray[np.float64],
    lattice_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
    defect_locations: NDArray[np.integer] | None = None,
) -> animation.FuncAnimation:
    """Animate particle positions on a skewed lattice defined by two lattice vectors."""
    n_cols, n_rows = lattice_dimension
    v0, v1 = lattice_vectors

    lattice_coords = np.array(
        [col * v0 + row * v1 for row in range(n_rows) for col in range(n_cols)]
    )
    lattice_x: NDArray[np.float64] = lattice_coords[:, 0]
    lattice_y: NDArray[np.float64] = lattice_coords[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_title("Particle Simulation")

    ax.set_xlim(lattice_x.min() - 1, lattice_x.max() + 1)
    ax.set_ylim(lattice_y.min() - 1, lattice_y.max() + 1)
    ax.set_xlabel("x distance")
    ax.set_ylabel("y distance")

    ax.scatter(
        lattice_x, lattice_y, color="aqua", marker=".", s=5, zorder=0, label="Sites"
    )

    if defect_locations is not None and defect_locations.size > 0:
        defect_coords = np.array([col * v0 + row * v1 for row, col in defect_locations])
        ax.scatter(
            defect_coords[:, 0],
            defect_coords[:, 1],
            facecolors="none",
            edgecolors="gray",
            marker="o",
            s=40,
            linewidths=2.0,
            zorder=2,
            label="Defects",
        )

    particle_scatter: PathCollection = ax.scatter(
        [], [], color="red", s=20, edgecolors="black", zorder=3, label="Particles"
    )
    ax.legend(loc="lower right")

    def update(frame: int) -> tuple[PathCollection]:
        occupancy = all_positions[frame]
        rows, cols = np.nonzero(occupancy)
        coords = np.array(
            [col * v0 + row * v1 for row, col in zip(rows, cols, strict=False)]
        )
        if coords.size > 0:
            particle_scatter.set_offsets(coords)
        else:
            particle_scatter.set_offsets(np.empty((0, 2)))
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


def plot_mean_jump_rates(
    results: list[SimulationResult], ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    """Plot attempted and successful jump counts."""
    delta = JUMP_DIRECTIONS
    labels = [f"{d}" for d in delta]

    width = 0.35
    fig, ax = get_figure(ax=ax)

    jump_counts = np.array([result.jump_count for result in results])
    mean_jump_count = jump_counts.mean(axis=0)

    indices = np.arange(mean_jump_count.shape[0], dtype=np.int_)
    ax.bar(indices - width / 2, mean_jump_count, width, label="Successful jumps")
    attempted_jump_count = np.array(
        [result.attempted_jump_counter for result in results]
    )
    mean_attempted_jump_count = attempted_jump_count.mean(axis=0)
    ax.bar(
        indices + width / 2, mean_attempted_jump_count, width, label="Attempted jumps"
    )
    ax.set_xlabel("Direction (delta row, delta col)")
    ax.set_ylabel("Count")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45)

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False):
        if labels[int(tick)] == "(0, 0)":
            label.set_color("gray")

    ax.legend()
    return fig, ax
