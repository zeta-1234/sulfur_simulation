from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

n_directions = 4
directions = {
    0: np.array([1, 0]),  # RIGHT
    1: np.array([-1, 0]),  # LEFT
    2: np.array([0, 1]),  # UP
    3: np.array([0, -1]),  # DOWN
}
rng = np.random.default_rng()


def get_delta_k(
    angle_of_incidence: float, total_scattering_angle: float, wavevector: np.ndarray
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate delta_k for given angles and incident wavevector."""
    return wavevector * (
        np.sin(total_scattering_angle - angle_of_incidence) - np.sin(angle_of_incidence)
    )


def update_position(
    position: np.ndarray, hopping_probability: float
) -> np.ndarray:  # temp function to enable simple random diffusion
    """Update the position of the particle."""
    if rng.random() < hopping_probability:
        direction = int(rng.integers(low=0, high=n_directions))
        return position + directions[direction]
    return position


def get_amplitude(
    delta_k: np.ndarray,
    position: np.ndarray[Any, np.dtype[np.float64]],
) -> complex:
    """Calculate the complex amplitude for a given delta_k and position."""
    return np.exp(-1j * np.dot(delta_k, position))


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    incident_wavevector: np.ndarray
    "The incident wavevector of the helium beam"
    total_scattering_angle: float
    """Total angle scattered through"""
    n_timesteps: int
    """Number of timesteps"""
    angle_of_incidence: float
    """Angle of incidence of helium"""
    lattice_spacing: float = 2.5
    "Spacing of lattice in Angstroms"
    step: float = 1
    """Step size in simulation"""
    hopping_probability: float = 0.01
    """The probability of hopping to a new position at each step."""
    form_factor: float = 1
    """Prefactor for scattered amplitude"""
