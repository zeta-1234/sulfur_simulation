from __future__ import annotations

import numpy as np

from examples.simulation import SimulationParameters

params = SimulationParameters(delta=1.2, duration=5000, gamma=1, step=2)

A = np.array([params.form_factor + 0j])


def a_value(del_k: np.ndarray, r: np.ndarray) -> complex:  # cspell:ignore ndarray
    """Calculate the complex amplitude for a given del_k and position.

    Args:
        del_k (np.ndarray): Change in momentum
        r (np.ndarray): Position

    Returns
    -------
        float: complex amplitude
    """
    return np.exp(-1j * np.dot(del_k, r))
