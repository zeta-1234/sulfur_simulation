from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

if TYPE_CHECKING:
    from hopping_calculator import HoppingCalculator
    from numpy.random import Generator


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    lattice_dimension: tuple[int, int]
    "Dimension of lattice"
    lattice_spacing: float = 2.5
    "Spacing of lattice in Angstroms"
    n_particles: int
    """The number of particles"""
    rng_seed: int
    """rng seed for reproducibility"""
    hopping_calculator: HoppingCalculator

    @property
    def times(self) -> np.ndarray:
        """Times for simulation."""
        return np.arange(0, self.n_timesteps)

    @property
    def initial_positions(self) -> np.ndarray:
        """
        Initial particle positions.

        Raises
        ------
        ValueError
            If the number of particles exceeds the number of lattice spaces.
        """
        if self.n_particles > np.prod(self.lattice_dimension):
            msg = "More particles than lattice spaces"
            raise ValueError(msg)

        rng = np.random.default_rng(seed=self.rng_seed)
        initial_positions = np.zeros(self.lattice_dimension, dtype=bool).ravel()
        initial_positions[: self.n_particles] = True
        return rng.permutation(initial_positions).reshape(self.lattice_dimension)


def _wrap_index(index: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    """Wrap index to stay within the bounds of the lattice."""
    return (index[0] % shape[0], index[1] % shape[1])


def _get_next_index(
    initial_index: tuple[int, int], jump: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    return _wrap_index((initial_index[0] + jump[0], initial_index[1] + jump[1]), shape)


def _make_jump(
    initial_position: int,
    jump_idx: int,
    particle_positions: np.ndarray,
) -> np.ndarray:
    # TODO: this should probably depend on the hopping calculator  # noqa: FIX002
    jump = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ][jump_idx]
    initial_index = np.unravel_index(initial_position, particle_positions.shape)
    final_idx = _get_next_index(initial_index, jump, particle_positions.shape)  # type: ignore[no-any-return]

    # if destination is full, don't do anything
    if particle_positions[final_idx]:
        return particle_positions

    particle_positions[final_idx] = True
    particle_positions[initial_index] = False
    return particle_positions


def _update_positions(
    particle_positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    jump_probabilities: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    true_locations = np.flatnonzero(particle_positions)

    # Go through the simulation one particle at a time
    # and make a jump based on jump_probabilities
    for particle_index, initial_location in enumerate(true_locations):
        cumulative_probabilities = np.cumsum(jump_probabilities[particle_index])
        rand_val = rng.random()

        for i, threshold in enumerate(cumulative_probabilities):
            if rand_val < threshold:
                particle_positions = _make_jump(
                    jump_idx=i,
                    particle_positions=particle_positions,
                    initial_position=initial_location,
                )
                break

    return particle_positions


def run_simulation(
    params: SimulationParameters,
) -> np.ndarray:
    """Run the simulation."""
    rng = np.random.default_rng(seed=params.rng_seed)
    all_positions = np.empty(
        (params.n_timesteps, *params.lattice_dimension), dtype=np.bool_
    )
    all_positions[0] = params.initial_positions

    for i in trange(1, params.n_timesteps):
        jump_probabilities = params.hopping_calculator.get_hopping_probabilities(
            all_positions[i - 1]
        )

        all_positions[i] = _update_positions(
            particle_positions=all_positions[i - 1],
            jump_probabilities=jump_probabilities,
            rng=rng,
        )

    return all_positions
