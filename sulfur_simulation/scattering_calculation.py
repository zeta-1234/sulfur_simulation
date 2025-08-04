from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

if TYPE_CHECKING:
    from numpy.random import Generator


def _update_position(
    position: np.ndarray,
    hopping_probability: float,
    lattice_spacing: float,
    rng: Generator,
) -> np.ndarray:  # temp function to enable simple random diffusion
    """Update the position of the particle."""
    n_directions = 4
    directions = {
        0: np.array([lattice_spacing, 0]),  # RIGHT
        1: np.array([-1 * lattice_spacing, 0]),  # LEFT
        2: np.array([0, lattice_spacing]),  # UP
        3: np.array([0, -1 * lattice_spacing]),  # DOWN
    }

    if rng.random() < hopping_probability:
        direction = int(rng.integers(low=0, high=n_directions))
        return position + directions[direction]
    return position


def run_simulation(
    params: SimulationParameters, *, seed: int | None
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Run a single simulation of particle diffusion."""
    position = params.initial_position
    positions = np.empty((params.n_timesteps, 2))
    rng = np.random.default_rng(seed)
    for i in range(1, params.n_timesteps):
        position = _update_position(
            position=position,
            hopping_probability=params.hopping_probability,
            lattice_spacing=params.lattice_spacing,
            rng=rng,
        )
        positions[i] = position
    return positions


def run_multiple_simulations(
    params: SimulationParameters, *, n_runs: int, stable_seed: bool = False
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """Run multiple simulations and returns all positions."""
    all_positions = np.empty((n_runs, params.n_timesteps, 2))
    for i in trange(n_runs, desc="Running simulations"):
        position = run_simulation(
            params,
            seed=i if stable_seed else None,
        )
        all_positions[i] = position
    return all_positions


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    initial_position: np.ndarray
    """Initial position of particle"""
    lattice_spacing: float = 2.5
    "Spacing of lattice in Angstroms"
    hopping_probability: float = 0.01
    """The probability of hopping to a new position at each step."""

    @property
    def times(self) -> np.ndarray:
        """Times for simulation."""
        return np.arange(0, self.n_timesteps)
