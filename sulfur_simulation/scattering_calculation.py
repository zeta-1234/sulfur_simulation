from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

if TYPE_CHECKING:
    from hopping_calculator import HoppingCalculator
    from numpy.random import Generator

JUMP_DIRECTIONS = np.array(
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


@dataclass(kw_only=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    lattice_dimension: tuple[int, int]
    "Dimension of lattice"
    n_particles: int
    """The number of particles"""
    hopping_calculator: HoppingCalculator
    results: SimulationResults = field(init=False)

    def __post_init__(self) -> None:
        self.results = SimulationResults(
            positions=np.empty((self.n_timesteps, *self.lattice_dimension), dtype=bool),
            jump_counter=np.zeros(9),
            attempted_jump_counter=np.zeros(9),
        )

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

        rng = np.random.default_rng()
        initial_positions = np.zeros(self.lattice_dimension, dtype=bool).ravel()
        initial_positions[: self.n_particles] = True
        return rng.permutation(initial_positions).reshape(self.lattice_dimension)


@dataclass(kw_only=True)
class SimulationResults:
    """Results of a simulation."""

    positions: np.ndarray
    "The particles' positions at each timestep"
    jump_counter: np.ndarray
    "The number of successful jumps in each direction"
    attempted_jump_counter: np.ndarray
    "The number of jumps attempted"


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
    params: SimulationParameters,
) -> np.ndarray:
    jump = JUMP_DIRECTIONS[jump_idx]

    initial_index = np.unravel_index(initial_position, particle_positions.shape)
    final_idx = _get_next_index(initial_index, jump, particle_positions.shape)  # type: ignore[no-any-return]

    # if destination is full, don't do anything
    if particle_positions[final_idx]:
        return particle_positions

    params.results.jump_counter[jump_idx] += 1
    particle_positions[final_idx] = True
    particle_positions[initial_index] = False
    return particle_positions


def _assert_cumulative_probability_valid(move_probabilities: np.ndarray) -> None:
    total_probability = np.sum(move_probabilities)
    if not np.isclose(total_probability, 1):
        msg = f"Invalid probability distribution, total probability ({total_probability}) != 1"
        raise ValueError(msg)


def _update_positions(
    particle_positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    jump_probabilities: np.ndarray,
    params: SimulationParameters,
    rng: Generator,
) -> np.ndarray:
    true_locations = np.flatnonzero(particle_positions)

    for idx in rng.permutation(len(true_locations)):
        initial_location = int(true_locations[idx])
        move_probabilities = jump_probabilities[idx]

        _assert_cumulative_probability_valid(move_probabilities)

        jump_idx = rng.choice(len(move_probabilities), p=move_probabilities)
        stationary_index = 4

        if jump_idx != stationary_index:
            params.results.attempted_jump_counter[jump_idx] += 1
            particle_positions = _make_jump(
                jump_idx=jump_idx,
                particle_positions=particle_positions,
                initial_position=initial_location,
                params=params,
            )

    return particle_positions


def run_simulation(params: SimulationParameters, rng_seed: int) -> np.ndarray:
    """Run the simulation."""
    rng = np.random.default_rng(seed=rng_seed)
    all_positions = params.results.positions
    all_positions[0] = params.initial_positions

    for i in trange(1, params.n_timesteps):
        jump_probabilities = params.hopping_calculator.get_hopping_probabilities(
            all_positions[i - 1]
        )

        all_positions[i] = _update_positions(
            params=params,
            particle_positions=all_positions[i - 1],
            jump_probabilities=jump_probabilities,
            rng=rng,
        )

    return all_positions
