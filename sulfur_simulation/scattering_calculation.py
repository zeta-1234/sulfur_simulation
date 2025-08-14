from __future__ import annotations

import warnings
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
    n_particles: int
    """The number of particles"""
    rng_seed: int | None = None
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


jump_counter = np.zeros(9)


def _make_jump(
    initial_position: int,
    jump_idx: int,
    particle_positions: np.ndarray,
) -> np.ndarray:
    # TODO: this list should probably depend on the hopping calculator  # noqa: FIX002
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
    jump_counter[jump_idx] += 1
    return particle_positions


CUMULATIVE_PROBABILITY_THRESHOLD = 0.5


def _assert_cumulative_probability_valid(cumulative_probabilities: np.ndarray) -> None:
    if cumulative_probabilities[-1] > 1:
        msg = f"Invalid probability distribution, total probability ({cumulative_probabilities[-1]}) exceeds 1"
        raise ValueError(msg)
    if cumulative_probabilities[-1] > CUMULATIVE_PROBABILITY_THRESHOLD:
        warnings.warn(
            f"Cumulative probability = {cumulative_probabilities[-1]}, "
            f"is larger than reccommended threshold {CUMULATIVE_PROBABILITY_THRESHOLD}",
            stacklevel=2,
        )


sampled_jumps = np.zeros(9)


def _update_positions(
    particle_positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    jump_probabilities: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    true_locations = np.flatnonzero(particle_positions)
    particle_indices = np.arange(len(true_locations))
    rng.shuffle(particle_indices)

    for idx in particle_indices:
        initial_location = true_locations[idx]
        move_probs = jump_probabilities[idx]
        stay_prob = max(0.0, 1.0 - move_probs.sum())
        choices = np.arange(len(move_probs) + 1)  # last index = stay put
        probs = np.append(move_probs, stay_prob)

        # Just for the check
        cumulative_probabilities = np.cumsum(move_probs)
        _assert_cumulative_probability_valid(cumulative_probabilities)

        jump_idx = rng.choice(choices, p=probs)

        if jump_idx < len(move_probs):
            sampled_jumps[jump_idx] += 1
            particle_positions = _make_jump(
                jump_idx=jump_idx,
                particle_positions=particle_positions,
                initial_position=initial_location,
            )

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
