from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(kw_only=True, frozen=True)
class SimulationParameters:
    """Parameters for simulating diffusion."""

    n_timesteps: int
    """Number of timesteps"""
    lattice_dimension: tuple[int, int]
    "Dimension of lattice"
    n_particles: int
    """The number of particles"""
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

        rng = np.random.default_rng()
        initial_positions = np.zeros(self.lattice_dimension, dtype=bool).ravel()
        initial_positions[: self.n_particles] = True
        return rng.permutation(initial_positions).reshape(self.lattice_dimension)


@dataclass(kw_only=True, frozen=True)
class SimulationResult:
    """Results of a simulation."""

    positions: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]
    "The particles' positions at each timestep"
    jump_counter: np.ndarray[tuple[int], np.dtype[np.int_]]
    "The number of successful jumps in each direction"
    attempted_jump_counter: np.ndarray[tuple[int], np.dtype[np.int_]]
    "The number of jumps attempted"


def _wrap_index(index: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    """Wrap index to stay within the bounds of the lattice."""
    return (index[0] % shape[0], index[1] % shape[1])


def _get_next_index(
    initial_index: tuple[int, int], jump: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    return _wrap_index((initial_index[0] + jump[0], initial_index[1] + jump[1]), shape)


def _make_jump(
    idx: int,
    result: SimulationResult,
    initial_location: int,
    jump_idx: int,
) -> None:
    initial_index = np.unravel_index(initial_location, result.positions[idx].shape)
    final_idx = _get_next_index(
        initial_index,  # type: ignore misc
        JUMP_DIRECTIONS[jump_idx],
        result.positions[idx].shape,
    )
    result.attempted_jump_counter[jump_idx] += 1
    # if destination is full, don't do anything
    if result.positions[idx][final_idx]:
        return

    result.jump_counter[jump_idx] += 1
    result.positions[idx][final_idx] = True
    result.positions[idx][initial_index] = False


def _assert_cumulative_probability_valid(move_probabilities: np.ndarray) -> None:
    total_probability = np.sum(move_probabilities)
    if not np.isclose(total_probability, 1):
        msg = f"Invalid probability distribution, total probability ({total_probability}) != 1"
        raise ValueError(msg)


def _update_result(
    idx: int,
    result: SimulationResult,
    jump_probabilities: np.ndarray,
    rng: Generator,
) -> None:
    true_locations = np.flatnonzero(result.positions[idx - 1])
    result.positions[idx] = result.positions[idx - 1]

    for loc_idx in rng.permutation(len(true_locations)):
        initial_location = int(true_locations[loc_idx])
        move_probabilities = jump_probabilities[loc_idx]

        _assert_cumulative_probability_valid(move_probabilities)

        jump_idx = rng.choice(len(move_probabilities), p=move_probabilities)
        stationary_index = 4

        if jump_idx != stationary_index:
            _make_jump(
                idx=idx,
                result=result,
                jump_idx=jump_idx,
                initial_location=initial_location,
            )


def run_simulation(params: SimulationParameters, rng_seed: int) -> SimulationResult:
    """Run the simulation."""
    rng = np.random.default_rng(seed=rng_seed)
    all_positions = np.empty(
        (params.n_timesteps, *params.lattice_dimension), dtype=np.bool_
    )
    all_positions[0] = params.initial_positions
    jump_counter = np.zeros(9, dtype=np.int_)
    attempted_jump_counter = np.zeros(9, dtype=np.int_)
    out = SimulationResult(
        positions=all_positions,
        jump_counter=jump_counter,
        attempted_jump_counter=attempted_jump_counter,
    )

    for i in trange(1, params.n_timesteps):
        jump_probabilities = params.hopping_calculator.get_hopping_probabilities(
            all_positions[i - 1]
        )

        _update_result(
            idx=i,
            result=out,
            jump_probabilities=jump_probabilities,
            rng=rng,
        )

    return out
