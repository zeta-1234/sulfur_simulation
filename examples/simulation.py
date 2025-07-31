from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    delta: float
    duration: float
    gamma: float

    step: float = 1
    Phop: float = 0.01  # default probability of hopping
    form_factor: float = 1  # modifier for scattered amplitude


params = SimulationParameters(delta=1.2, duration=5000, gamma=1, step=2)

# use params.gamma to get variable gamma from the parameters
