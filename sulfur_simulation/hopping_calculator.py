from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, cast, override

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]
from scipy.optimize import (  # type: ignore[reportMissingTypeStubs]
    RootResults,
    root_scalar,  # type: ignore[reportMissingTypeStubs]
)

if TYPE_CHECKING:
    from collections.abc import Callable


class HoppingCalculator(ABC):
    """Abstract base class for calculating hopping probabilities."""

    @abstractmethod
    def get_hopping_probabilities(self, positions: np.ndarray) -> np.ndarray:
        """Get hopping probabilities."""


class SquareHoppingCalculator(HoppingCalculator):
    """Class for calculating hopping probabilities in a square lattice."""

    def __init__(self, baserate: tuple[float, float], temperature: float) -> None:
        self._straight_baserate = baserate[0]
        self._diagonal_baserate = baserate[1]
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
        neighbor_energies = energies[neighbor_rows, neighbor_cols]
        current_energies = energies[rows, cols][:, None]

        max_exp_arg = np.log(1 / min(self._straight_baserate, self._diagonal_baserate))
        beta = 1 / (2 * Boltzmann * self._temperature)
        energy_difference = neighbor_energies - current_energies
        exponent = np.clip(-beta * energy_difference, a_min=None, a_max=max_exp_arg)
        if np.any(np.isclose(exponent, max_exp_arg)):
            warnings.warn(
                "Some energy differences are too large, the resulting distribution may be inaccurate.",
                stacklevel=2,
            )

        base_rates = np.full(delta.shape[0], self._straight_baserate)
        base_rates[[0, 2, 6, 8]] = self._diagonal_baserate
        base_rates[4] = 0

        rates = np.exp(exponent) * base_rates

        if np.sum(rates) > 1.0:
            rates /= np.sum(rates)

        if np.sum(rates) > 1.0:
            rates /= np.sum(rates)

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


class InteractingHoppingCalculator(SquareHoppingCalculator):
    """Hopping Calculator with a Lennard Jones potential between particles."""

    def __init__(
        self,
        baserate: tuple[float, float],
        temperature: float,
        lattice_spacing: float,
        interaction: Callable[[float], float],
        cutoff_radius_potential: float = 1.6e-22,
    ) -> None:
        super().__init__(baserate, temperature)

        self._lattice_spacing = lattice_spacing
        self._interaction = interaction
        self._cutoff_radius_potential = cutoff_radius_potential

    @cached_property
    def _potential_table(self) -> dict[tuple[int, int], float]:
        """Create lookup table for interactin potential values."""
        max_cutoff_radius = 6

        def _energy_difference(r: float) -> float:
            # subtract potential at infinity
            return (
                abs(
                    self._interaction(r)
                    - self._interaction(1000 * self._lattice_spacing)
                )
                - self._cutoff_radius_potential
            )

        cutoff_radius = _find_cutoff_radius(
            energy_difference=_energy_difference,
            r_max=max_cutoff_radius * self._lattice_spacing,
            r_min=self._lattice_spacing,
        )

        cutoff_radius = int(np.ceil(cutoff_radius / self._lattice_spacing))

        lookup: dict[tuple[int, int], float] = {}
        for dx in range(-cutoff_radius, cutoff_radius + 1):
            for dy in range(-cutoff_radius, cutoff_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                r = self._lattice_spacing * np.sqrt(dx**2 + dy**2)
                if r <= cutoff_radius * self._lattice_spacing:
                    lookup[dx, dy] = self._interaction(r)

        return lookup

    @override
    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions)

        n_rows, n_columns = positions.shape
        occupied_coordinates = np.argwhere(positions)

        # Add potential contributions from each occupied site
        for particle_row, particle_col in occupied_coordinates:
            for (delta_row, delta_col), lj_potential in self._potential_table.items():
                target_row = (particle_row + delta_row) % n_rows
                target_col = (particle_col + delta_col) % n_columns
                energies[target_row, target_col] += lj_potential

        return energies


def get_lennard_jones_potential(
    sigma: float,
    epsilon: float,
    cutoff_energy: float = 1.9e-20,
) -> Callable[[float], float]:
    """
    Take float and return Lennard Jones potential.

    ...math:::
        V(r) = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    Sigma = radius where potential = 0
    Epsilon = minimum potential reached
    """

    def _potential(r: float) -> float:
        sr6 = (sigma / r) ** 6
        sr12 = sr6**2
        return np.clip(4 * epsilon * (sr12 - sr6), a_min=None, a_max=cutoff_energy)

    return _potential


def _find_cutoff_radius(
    energy_difference: Callable[[float], float],
    r_max: float,
    r_min: float,
    num_points: int = 1000,
) -> float:
    """
    Find the largest root of `energy_difference(r)` by scanning inward from r_max.

    Raises
    ------
    ValueError
        If no root is found in the given range.
    """
    r_values = np.linspace(r_max, r_min, num_points)
    energy_values = [energy_difference(r) for r in r_values]

    # Scan inward for the first sign change
    for i in range(len(r_values) - 1):
        f_current = energy_values[i]
        f_next = energy_values[i + 1]

        if f_current * f_next < 0:
            bracket_a = r_values[i + 1]
            bracket_b = r_values[i]

            # Find root in bracket using Brent's method
            solution = cast(
                "RootResults",
                root_scalar(
                    energy_difference, bracket=[bracket_a, bracket_b], method="brentq"
                ),
            )

            if solution.converged:
                return solution.root
    if energy_values[0] < 0:
        return 0
    msg = "No root found in the given range."
    raise ValueError(msg)
