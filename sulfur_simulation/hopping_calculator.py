from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, override

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]
from scipy.optimize import brentq  # type: ignore[reportMissingTypeStubs]


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
        max_exp_arg = np.log(0.51 / self._baserate)
        beta = 1 / (2 * Boltzmann * self._temperature)
        energy_difference = neighbor_energies - current_energies
        exponent = np.clip(-beta * energy_difference, a_min=None, a_max=max_exp_arg)
        rates = np.exp(exponent) * self._baserate

        # Prevent self-jumps
        rates[:, 4] = 0

        return rates

    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        _ = self
        return np.full(positions.shape, 3.2e-19)


class LineDefectHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator for a line defect in a square lattice."""

    @override
    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions=positions)
        energies[positions.shape[0] // 2, :] = 0
        return energies


class LennardJonesHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator with a Lennard Jones potential between particles."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        baserate: float,
        temperature: float,
        lattice_spacing: float,
        sigma: float,
        epsilon: float,
        cutoff_energy: float = 1.6e-22,
    ) -> None:
        super().__init__(baserate, temperature)
        self.lj_table = self._get_lj_table(
            sigma=sigma,
            epsilon=epsilon,
            lattice_spacing=lattice_spacing,
            cutoff_energy=cutoff_energy,
        )

    @classmethod
    def _get_lj_table(
        cls, sigma: float, epsilon: float, lattice_spacing: float, cutoff_energy: float
    ) -> dict[tuple[int, int], float]:
        """Create lookup table for Lennard Jones values."""
        max_cutoff = 5

        def lj_potential(r: float) -> float:
            sr6 = (sigma / r) ** 6
            sr12 = sr6**2
            return 4 * epsilon * (sr12 - sr6)

        def energy_difference(r: float) -> float:
            return abs(lj_potential(r)) - cutoff_energy

        try:
            r_cut = cast(
                "float", brentq(energy_difference, sigma * 1.01, 5 * lattice_spacing)
            )
        except ValueError:
            r_cut = 5 * lattice_spacing

        cutoff = min(int(np.ceil(r_cut / lattice_spacing)), max_cutoff)
        lookup: dict[tuple[int, int], float] = {}
        for dx in range(-cutoff, cutoff + 1):
            for dy in range(-cutoff, cutoff + 1):
                if dx == 0 and dy == 0:
                    continue
                r = lattice_spacing * np.sqrt(dx**2 + dy**2)
                if r <= cutoff * lattice_spacing:
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6**2
                    lookup[dx, dy] = 4 * epsilon * (sr12 - sr6)

        return lookup

    @override
    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions)

        n_rows, n_columns = positions.shape
        occupied_coordinates = np.argwhere(positions)

        # Add Lennard-Jones contributions from each occupied site
        for particle_row, particle_col in occupied_coordinates:
            for (delta_row, delta_col), lj_potential in self.lj_table.items():
                target_row = (particle_row + delta_row) % n_rows
                target_col = (particle_col + delta_col) % n_columns
                energies[target_row, target_col] += lj_potential

        return energies
