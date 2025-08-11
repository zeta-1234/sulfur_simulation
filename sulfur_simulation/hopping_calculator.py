from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from sulfur_simulation.scattering_calculation import (
        SimulationParameters,  # type: ignore library types
    )


class HoppingCalculator(ABC):
    """Abstract base class for calculating hopping probabilities."""

    @abstractmethod
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """Get hopping probabilities."""


class SquareHoppingCalculator(HoppingCalculator):
    """Class for calculating hopping probabilities in a square lattice."""

    def __init__(self, baserate: float, temperature: float) -> None:
        self._baserate = baserate
        self._temperature = temperature

    @override
    def get_hopping_probabilities(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        energies = self._get_energy_landscape(positions=positions)
        rows, cols = np.nonzero(positions)

        delta = np.array(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        )

        neighbor_rows = (rows[:, None] + delta[:, 0]) % positions.shape[0]
        neighbor_cols = (cols[:, None] + delta[:, 1]) % positions.shape[1]
        np.ravel_multi_index((neighbor_rows, neighbor_cols), positions.shape)

        neighbor_energies = energies[neighbor_rows, neighbor_cols]
        current_energies = energies[rows, cols][:, None]

        # Calculate the rate based on the boltzmann factor
        # beta = delta_e / (k_B * T)  # noqa: ERA001
        beta = 1 / (2 * Boltzmann * self._temperature)
        energy_difference = neighbor_energies - current_energies
        rates = np.exp(-beta * energy_difference) * self._baserate

        # Prevent self-jumps
        rates[:, 4] = 0

        return rates

    @classmethod
    def _get_energy_landscape(
        cls, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        return np.full(positions.shape, 3.2e-19)


class LineDefectHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator for a line defect in a square lattice."""

    @classmethod
    @override
    def _get_energy_landscape(
        cls, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions=positions)
        energies[positions.shape[0] // 2, :] = 0
        return energies


class LennardJonesHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator with a Leonard-Jones potential in a square lattice."""

    def __init__(
        self,
        sigma: float,
        epsilon: float,
        cutoff: int,
        params: SimulationParameters,
        *args: tuple[Any, ...],
        **kwargs: tuple[Any, ...],
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore RuffANN002

        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.params = params

        self.lj_table = self.lj_lookup_table()

    def lj_lookup_table(self) -> dict[tuple[int, int], float]:
        """Generate a lookup table of Lennard-Jones potential values."""
        lookup: dict[tuple[int, int], float] = {}

        for dx in range(-self.cutoff, self.cutoff + 1):
            for dy in range(-self.cutoff, self.cutoff + 1):
                if dx == 0 and dy == 0:
                    continue

                r = self.params.lattice_spacing * np.sqrt(dx**2 + dy**2)

                if r <= self.cutoff * self.params.lattice_spacing:
                    sr6 = (self.sigma / r) ** 6
                    sr12 = sr6**2
                    potential = 4 * self.epsilon * (sr12 - sr6)
                    lookup[dx, dy] = potential

        return lookup
