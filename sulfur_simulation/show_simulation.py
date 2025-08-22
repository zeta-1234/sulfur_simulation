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
    lattice_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
    timesteps: NDArray[np.int_],
    max_ticks: int = 10,
) -> animation.FuncAnimation:
    """Animate particle positions on a skewed 2D lattice defined by two lattice vectors."""
    v0, v1 = lattice_vectors
    n_rows, n_cols = all_positions.shape[1:3]

    rows, cols = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_cols, dtype=np.float64),
        indexing="ij",
    )
    lattice_positions: NDArray[np.float64] = (
        cols[..., None] * v0 + rows[..., None] * v1
    ).reshape(-1, 2)

    x_min, y_min = lattice_positions.min(axis=0)
    x_max, y_max = lattice_positions.max(axis=0)
    fig, ax = plt.subplots(figsize=(6, 6 * ((y_max - y_min) / (x_max - x_min))))

    ax.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        color="aqua",
        s=5,
        marker=".",
        zorder=0,
    )

    particles: PathCollection = ax.scatter(
        [], [], color="red", s=20, edgecolors="black", zorder=1
    )

    ax.set_aspect("equal")
    ax.set_title("Particle Simulation")
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.legend(["Sites", "Particles"], loc="lower right")

    def nice_ticks(n_points: int, max_ticks: int = 10) -> NDArray[np.int_]:
        step = max(1, int(np.ceil(n_points / max_ticks)))
        return np.arange(0, n_points, step)

    ax.set_xticks(nice_ticks(n_cols, max_ticks))
    ax.set_xticklabels([str(i) for i in nice_ticks(n_cols, max_ticks)])
    ax.set_yticks(nice_ticks(n_rows, max_ticks))
    ax.set_yticklabels([str(i) for i in nice_ticks(n_rows, max_ticks)])

    def update(frame: int) -> tuple[PathCollection]:
        r_idx, c_idx = np.nonzero(all_positions[frame])
        positions = c_idx[:, None] * v0 + r_idx[:, None] * v1
        particles.set_offsets(positions)
        ax.set_title(f"Timestep: {frame}")
        return (particles,)

    return animation.FuncAnimation(
        fig,
        update,
        frames=timesteps,
        interval=5000 / len(timesteps),
        blit=False,
        repeat=True,
    )


# TODO: need to make axes scale properly to figure


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

    indices: NDArray[np.int_] = np.arange(mean_jump_count.shape[0], dtype=np.int_)
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
