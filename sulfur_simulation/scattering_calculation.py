from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def get_amplitude(
    delta_k: np.ndarray,
    position: np.ndarray[Any, np.dtype[np.float64]],
) -> complex:
    """Calculate the complex amplitude for a given delta_k and position."""
    return np.exp(-1j * np.dot(delta_k, position))


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    delta: float  # TODO: docs
    n_timesteps: int
    gamma: float  # TODO: docs

    step: float = 1  # TODO: docs
    hopping_probability: float = 0.01  # TODO: docs
    """The probability of hopping to a new position at each step."""
    form_factor: float = 1  # TODO: docs
