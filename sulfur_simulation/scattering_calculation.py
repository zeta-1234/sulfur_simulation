from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from tqdm import trange

from sulfur_simulation.sulfur_data import DEFECT_LOCATIONS
from sulfur_simulation.sulfur_nickel_calculator import SulfurNickelHoppingCalculator

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
    jump_count: np.ndarray[tuple[int], np.dtype[np.int_]]
    "The number of successful jumps in each direction"
    attempted_jump_counter: np.ndarray[tuple[int], np.dtype[np.int_]]
    "The number of jumps attempted"
    layers: np.ndarray[tuple[int, int, int, int], np.dtype[np.bool_]] | None


def _make_jump(
    idx: int,
    result: SimulationResult,
    jump_idx: int,
    move_destinations: np.ndarray,
    initial_location: np.ndarray,
) -> None:
    row, column, layer = move_destinations[jump_idx]
    old_row, old_column = np.unravel_index(
        initial_location, result.positions[idx].shape
    )

    if layer == -1:
        result.attempted_jump_counter[jump_idx] += 1
        if result.positions[idx][row, column]:
            return
        result.jump_count[jump_idx] += 1
        result.positions[idx][row, column] = True
    else:
        assert result.layers is not None
        result.attempted_jump_counter[9] += 1
        if result.layers[idx][layer][row, column]:
            return
        result.jump_count[9] += 1
        result.layers[idx][layer][row, column] = True
    result.positions[idx][old_row, old_column] = False


def _assert_cumulative_probability_valid(move_probabilities: np.ndarray) -> None:
    total_probability = np.sum(move_probabilities)
    if not np.isclose(total_probability, 1):
        msg = f"Invalid probability distribution, total probability ({total_probability}) != 1"
        raise ValueError(msg)


def _update_result(
    idx: int,
    result: SimulationResult,
    jump_probabilities: list[np.ndarray],
    jump_destinations: list[np.ndarray],
    rng: Generator,
) -> None:
    true_locations = np.flatnonzero(result.positions[idx - 1])
    result.positions[idx] = result.positions[idx - 1].copy()

    for particle_idx in rng.permutation(len(true_locations)):
        move_probabilities = cast("np.ndarray", jump_probabilities[particle_idx])
        move_destinations = cast("np.ndarray", jump_destinations[particle_idx])

        _assert_cumulative_probability_valid(move_probabilities)

        jump_idx = rng.choice(len(move_probabilities), p=move_probabilities)
        stationary_index = 4

        if jump_idx != stationary_index:
            _make_jump(
                idx=idx,
                result=result,
                jump_idx=jump_idx,
                move_destinations=move_destinations,
                initial_location=true_locations[particle_idx],
            )


def _run_single_simulation(
    params: SimulationParameters, rng: Generator
) -> SimulationResult:
    """Run the simulation."""
    all_positions = np.empty(
        (params.n_timesteps, *params.lattice_dimension), dtype=np.bool_
    )
    all_positions[0] = params.initial_positions
    jump_counter = np.zeros(10, dtype=np.int_)
    attempted_jump_counter = np.zeros(10, dtype=np.int_)

    if isinstance(params.hopping_calculator, SulfurNickelHoppingCalculator):
        all_layers = np.empty(
            (
                params.n_timesteps,
                len(DEFECT_LOCATIONS),
                *(
                    params.hopping_calculator.sulfur_nickel_data.max_layer_size,
                    params.hopping_calculator.sulfur_nickel_data.max_layer_size,
                ),
            ),
            dtype=np.bool_,
        )
        all_layers[0] = (
            params.hopping_calculator.sulfur_nickel_data.initial_sulfur_layers
        )

        out = SimulationResult(
            positions=all_positions,
            jump_count=jump_counter,
            attempted_jump_counter=attempted_jump_counter,
            layers=all_layers,
        )

        for i in trange(1, params.n_timesteps):
            jump_probabilities, jump_destinations = (
                params.hopping_calculator.get_hopping_probabilities_and_destinations(
                    all_positions[i - 1], layers=all_layers
                )
            )

            _update_result(
                idx=i,
                result=out,
                jump_probabilities=jump_probabilities,
                jump_destinations=jump_destinations,
                rng=rng,
            )

    else:
        out = SimulationResult(
            positions=all_positions,
            jump_count=jump_counter,
            attempted_jump_counter=attempted_jump_counter,
            layers=None,
        )

        for i in trange(1, params.n_timesteps):
            jump_probabilities, jump_destinations = (
                params.hopping_calculator.get_hopping_probabilities_and_destinations(
                    all_positions[i - 1], layers=None
                )
            )

            _update_result(
                idx=i,
                result=out,
                jump_probabilities=jump_probabilities,
                jump_destinations=jump_destinations,
                rng=rng,
            )

    return out


def run_simulation(
    n_runs: int, params: SimulationParameters, rng: Generator | None = None
) -> list[SimulationResult]:
    """Run multiple simulations and return the results as a list."""
    rng = np.random.default_rng() if rng is None else rng
    return [_run_single_simulation(params=params, rng=rng) for _ in range(n_runs)]
