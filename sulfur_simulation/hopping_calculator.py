from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, cast, override

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]
from scipy.optimize import (  # type: ignore[reportMissingTypeStubs]
    RootResults,
    root_scalar,  # type: ignore[reportMissingTypeStubs]
)

from sulfur_simulation.sulfur_data import JUMP_DIRECTIONS

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseRate(ABC):
    """Return grid of baserates."""

    @property
    @abstractmethod
    def grid(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        """Generate grid of baserates."""
        ...


@dataclass(kw_only=True, frozen=True)
class SquareBaseRate(BaseRate):
    """Baserates for hopping."""

    straight_rate: float
    "Baserate for moving horizontally or vertically in a square lattice"
    diagonal_rate: float = 0
    "Baserate for moving diagonally in a square lattice"

    @property
    @override
    def grid(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return np.array(
            [
                self.diagonal_rate,
                self.straight_rate,
                self.diagonal_rate,
                self.straight_rate,
                0,
                self.straight_rate,
                self.diagonal_rate,
                self.straight_rate,
                self.diagonal_rate,
            ]
        )


@dataclass(kw_only=True, frozen=True)
class HexagonalBaseRate(BaseRate):
    """Baserates for hopping."""

    rate: float

    @property
    @override
    def grid(self) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        return np.array(
            [
                0,
                self.rate,
                self.rate,
                self.rate,
                0,
                self.rate,
                self.rate,
                self.rate,
                0,
            ]
        )


class HoppingCalculator(ABC):
    """Abstract base class for calculating hopping probabilities."""

    @abstractmethod
    def get_hopping_probabilities_and_destinations(
        self, positions: np.ndarray, layers: np.ndarray | None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get hopping probabilities."""


class BaseRateHoppingCalculator(HoppingCalculator):
    """Class for calculating hopping probabilities in a square lattice."""

    def __init__(
        self,
        *,
        baserate: BaseRate,
        temperature: float,
        lattice_directions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._baserate = baserate
        self._temperature = temperature
        self._lattice_directions = lattice_directions

        a, b = self._lattice_directions
        if np.isclose(a[0] * b[1] - a[1] * b[0], 0.0):
            msg = "Lattice vectors cannot be parallel."
            raise ValueError(msg)

    @property
    def normalized_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalize lattice directions."""
        a, b = self._lattice_directions
        return a / np.linalg.norm(a), b / np.linalg.norm(b)

    def _get_rates(
        self,
        positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
        layer_access_sites: np.ndarray | None,
        blocked_sites: np.ndarray | None,
    ) -> np.ndarray:
        _ = layer_access_sites
        _ = blocked_sites
        energies = self._get_energy_landscape(positions=positions)

        rows, cols = np.nonzero(positions)

        delta = JUMP_DIRECTIONS

        beta = 1 / (2 * Boltzmann * self._temperature)
        max_exp_arg = np.log(1 / np.min(self._baserate.grid[self._baserate.grid > 0]))

        # Compute energy differences to neighbors
        neighbor_rows = (rows[:, None] + delta[:, 0]) % positions.shape[0]
        neighbor_cols = (cols[:, None] + delta[:, 1]) % positions.shape[1]

        energy_difference = (
            energies[neighbor_rows, neighbor_cols] - energies[rows, cols][:, None]
        )

        exponent = np.clip(-beta * energy_difference, a_min=None, a_max=max_exp_arg)
        if np.any(np.isclose(exponent, max_exp_arg)):
            warnings.warn(
                "Some energy differences are too large, the resulting distribution may be inaccurate.",
                stacklevel=2,
            )

        return np.exp(exponent) * self._baserate.grid

    @override
    def get_hopping_probabilities_and_destinations(
        self,
        positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
        layers: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        rates = self._get_rates(
            positions=positions, layer_access_sites=None, blocked_sites=None
        )
        row_sums = rates.sum(axis=1)
        over_rows = row_sums > 1.0
        rates[over_rows] /= row_sums[over_rows, None]

        warning_threshold = 0.5
        if np.any(row_sums > warning_threshold):
            warnings.warn("Some probabilities exceed 0.5", stacklevel=2)

        # Stationary probability
        rates[:, 4] = 1 - rates.sum(axis=1)
        rates = np.clip(rates, 0.0, 1.0)

        probabilities_list: list[np.ndarray] = []
        destinations_list: list[np.ndarray] = []

        shape = positions.shape
        rows, cols = np.nonzero(positions)
        for r, c, prob_row in zip(rows, cols, rates, strict=True):
            destinations = []
            for jump in JUMP_DIRECTIONS:  # includes stationary at index 4
                rr, cc = self._get_next_index((r, c), jump, shape)
                destinations.append((rr, cc, -1))  # ground level

            probabilities_list.append(prob_row)
            destinations_list.append(np.array(destinations, dtype=int))

        return probabilities_list, destinations_list

    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        _ = self
        return np.full(positions.shape, 3.2e-19)

    def _wrap_index(
        self, index: tuple[int, int], shape: tuple[int, int]
    ) -> tuple[int, int]:
        """Wrap index to stay within the bounds of the lattice."""
        _ = self
        return (index[0] % shape[0], index[1] % shape[1])

    def _get_next_index(
        self,
        initial_index: tuple[int, int],
        jump: tuple[int, int],
        shape: tuple[int, int],
    ) -> tuple[int, int]:
        return self._wrap_index(
            (initial_index[0] + jump[0], initial_index[1] + jump[1]), shape
        )


class LineDefectHoppingCalculator(BaseRateHoppingCalculator):
    """Hopping Calculator for a line defect in a square lattice."""

    def __init__(
        self,
        *,
        baserate: BaseRate,
        temperature: float,
        lattice_directions: tuple[np.ndarray, np.ndarray],
    ) -> None:
        super().__init__(
            baserate=baserate,
            temperature=temperature,
            lattice_directions=lattice_directions,
        )

    @override
    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Generate the energy landscape for the lattice."""
        energies = super()._get_energy_landscape(positions=positions)
        energies[positions.shape[0] // 2, :] = 0
        return energies


class InteractingHoppingCalculator(BaseRateHoppingCalculator):
    """Hopping Calculator with a Lennard Jones potential between particles in a square lattice."""

    def __init__(
        self,
        *,
        baserate: BaseRate,
        temperature: float,
        lattice_directions: tuple[np.ndarray, np.ndarray],
        interaction: Callable[[float], float],
        cutoff_potential: float = 1.6e-22,
    ) -> None:
        super().__init__(
            baserate=baserate,
            temperature=temperature,
            lattice_directions=(lattice_directions[0], lattice_directions[1]),
        )
        self._lattice_spacing = float(np.linalg.norm(lattice_directions[0]))
        self._interaction = interaction
        self.cutoff_potential = cutoff_potential

        a, b = self._lattice_directions
        if np.linalg.norm(a) != np.linalg.norm(b):
            msg = "Lattice vectors should have same magnitude"
            raise ValueError(msg)

    def _build_lj_lookup_table(
        self, cutoff_radius: int
    ) -> dict[tuple[int, int], float]:
        lookup: dict[tuple[int, int], float] = {}

        for dx in range(-cutoff_radius, cutoff_radius + 1):
            for dy in range(-cutoff_radius, cutoff_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                displacement = (
                    dx * self.normalized_directions[0]
                    + dy * self.normalized_directions[1]
                )
                r = float(self._lattice_spacing * np.linalg.norm(displacement))

                if r <= cutoff_radius * self._lattice_spacing:
                    lookup[dx, dy] = self._interaction(r)

        return lookup

    @cached_property
    def _potential_table(self) -> dict[tuple[int, int], float]:
        """Create lookup table for interaction potential values."""
        max_cutoff_radius = 6

        def _energy_difference(r: float) -> float:
            # subtract potential at infinity and the cutoff potential
            abs_difference = abs(
                self._interaction(r) - self._interaction(1000 * self._lattice_spacing)
            )
            return abs_difference - self.cutoff_potential

        cutoff_radius = _find_cutoff_radius(
            energy_difference=_energy_difference,
            r_max=max_cutoff_radius * self._lattice_spacing,
            r_min=self._lattice_spacing,
        )

        cutoff_radius = int(np.ceil(cutoff_radius / self._lattice_spacing))

        return self._build_lj_lookup_table(cutoff_radius=cutoff_radius)

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
