from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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


def _run_simulation(
    initial_position: np.ndarray,
    hopping_probability: float,
    lattice_spacing: float,
    rng: Generator,
    n_timesteps: int,
) -> np.ndarray:
    position = initial_position
    positions = np.empty((n_timesteps, 2))
    for i in range(1, n_timesteps):
        position = _update_position(
            position=position,
            hopping_probability=hopping_probability,
            lattice_spacing=lattice_spacing,
            rng=rng,
        )
        positions[i] = position
    return positions


def run_multiple_simulations(params: SimulationParameters, n_runs: int) -> np.ndarray:
    """Run multiple simulations and returns all positions."""
    all_positions = np.empty((n_runs, params.n_timesteps, 2))
    for i in range(n_runs):
        rng = np.random.default_rng(seed=i)
        position = _run_simulation(
            initial_position=params.initial_position,
            hopping_probability=params.hopping_probability,
            lattice_spacing=params.lattice_spacing,
            rng=rng,
            n_timesteps=params.n_timesteps,
        )
        all_positions[i] = position
    return all_positions


def get_amplitude(
    form_factor: float, delta_k: np.ndarray, position: np.ndarray
) -> np.ndarray:
    """Calculate the complex amplitude for a given delta_k and position."""
    r, t, _ = position.shape
    m = delta_k.shape[0]

    amplitudes = np.empty((r, t, m), dtype=np.complex128)

    for i in range(r):
        phase = position[i] @ delta_k.T  # shape (t, m)
        amplitudes[i] = form_factor * np.exp(-1j * phase)

    return amplitudes


def get_delta_k(
    n_points: int,
    delta_k_range: tuple[float, float],
    direction: tuple[float, float] = (1, 0),
) -> np.ndarray:
    """Return a matrix of delta_k values in the [1,0] direction."""
    abs_delta_k = np.linspace(
        start=delta_k_range[0], stop=delta_k_range[1], num=n_points, endpoint=True
    )

    return np.asarray(direction)[np.newaxis, :] * abs_delta_k[:, np.newaxis]


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
    form_factor: float = 1
    """Prefactor for scattered amplitude"""

    @property
    def times(self) -> np.ndarray:
        """Times for simulation."""
        return np.arange(0, self.n_timesteps)


@dataclass(kw_only=True, frozen=True)
class ResultParameters:
    """Parameters for plotting results of simulation."""

    n_delta_k_intervals: int
    """The number of delta_k values to use"""
    delta_k_max: float
    """The max value of delta_k"""
    delta_k_min: float = 0.1
    """The min value of delta_k"""

    def __post_init__(self) -> None:
        if self.delta_k_min == 0:
            msg = "delta_k_min should not be zero"
            raise ValueError(msg)

    @property
    def delta_k_array(self) -> np.ndarray:
        """All delta_k values."""
        return get_delta_k(
            n_points=self.n_delta_k_intervals,
            delta_k_range=(self.delta_k_min, self.delta_k_max),
        )
