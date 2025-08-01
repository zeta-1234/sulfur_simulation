from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def update_position(
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


def get_amplitude(
    form_factor: float,
    delta_k: np.ndarray,
    position: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray:
    """Calculate the complex amplitude for a given delta_k and position."""
    return form_factor * np.exp(-1j * delta_k @ position.T)


def get_10_delta_k(
    n_delta_k: int, max_delta_k: float, min_delta_k: float
) -> np.ndarray:
    """Return a matrix of delta_k values in the [1,0] direction."""
    delta_k_x = np.linspace(start=min_delta_k, stop=max_delta_k, num=n_delta_k)
    delta_k_y = np.zeros(n_delta_k)
    return np.stack([delta_k_x, delta_k_y], axis=1)


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    lattice_spacing: float = 2.5
    "Spacing of lattice in Angstroms"
    step: int = 1
    """Step size in simulation"""
    hopping_probability: float = 0.01
    """The probability of hopping to a new position at each step."""
    form_factor: float = 1
    """Prefactor for scattered amplitude"""


@dataclass(kw_only=True, frozen=True)
class ResultParameters:
    """Parameters for plotting results of simulation."""

    n_delta_k_intervals: int
    """The number of delta_k values to use"""
    delta_k_max: float
    """The max value of delta_k"""
    delta_k_min: float = 0.1
    """The min value of delta_k"""
