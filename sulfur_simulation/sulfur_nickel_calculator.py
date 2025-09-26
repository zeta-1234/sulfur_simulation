from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from sulfur_simulation.hopping_calculator import BaseRate, InteractingHoppingCalculator

if TYPE_CHECKING:
    from collections.abc import Callable
import warnings

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]

from sulfur_simulation.sulfur_data import (
    DEFECT_LOCATIONS,
    JUMP_DIRECTIONS,
    blocked_hcp_sites,
    hcp_horizontal_vector,
    hcp_vertical_vector,
    relative_tile_positions,
)


@dataclass(kw_only=True, frozen=True)
class SulfurNickelData:
    """Parameters for the Sulfur on Nickel simulation."""

    max_layer_size: int
    "The maximum dimension a sulfur structure can grow to."
    layer_energy: float = -7e-21
    "The potential energy of empty, available spaces in the layers."

    def __post_init__(self) -> None:
        if self.max_layer_size % 2 == 0:
            msg = "max_layer_size must be an odd integer."
            raise ValueError(msg)

    @property
    def layer_tiling_indices(self) -> np.ndarray:
        """The indices of a sulfur layer."""
        layer_indices = np.zeros(
            (self.max_layer_size, self.max_layer_size), dtype=np.int32
        )

        tile_horizontal_vector = np.array([1, -3])
        tile_vertical_vector = np.array([3, 1])

        n_tiles = int((self.max_layer_size - 1) / 6) + 1
        for x_value in range(-n_tiles, n_tiles):
            for y_value in range(-n_tiles, n_tiles):
                for tile_index, tile_position in relative_tile_positions.items():
                    position = (
                        x_value * tile_horizontal_vector
                        + y_value * tile_vertical_vector
                        + tile_position
                    )
                    if (
                        np.abs(position[0]) <= (self.max_layer_size - 1) / 2
                        and np.abs(position[1]) <= (self.max_layer_size - 1) / 2
                    ):
                        layer_indices[int(position[0] + (self.max_layer_size - 1) / 2)][
                            int(position[1] + (self.max_layer_size - 1) / 2)
                        ] = int(tile_index)

        return layer_indices

    @property
    def layer_tiling_indices_mirrored(self) -> np.ndarray:
        """The mirrored indices of a sulfur layer."""
        return self.layer_tiling_indices[:, ::-1]

    @property
    def initial_sulfur_layers(self) -> np.ndarray:
        """Generate empty sulfur layers."""
        return np.zeros(
            (len(DEFECT_LOCATIONS), self.max_layer_size, self.max_layer_size),
            dtype=np.bool_,
        )

    @property
    def sulfur_layers_data(self) -> np.ndarray:
        """Generate data for sulfur layers in format (center position, orientation)."""
        dtype = np.dtype([("position", np.int32, (2,)), ("orientation", np.int32)])
        layer_data = np.empty((len(DEFECT_LOCATIONS)), dtype=dtype)
        rng = np.random.default_rng()
        for i, start_position in enumerate(DEFECT_LOCATIONS):
            layer_data[i] = (start_position, int(rng.integers(0, 6)))
        return layer_data


class SulfurNickelHoppingCalculator(InteractingHoppingCalculator):
    """Hopping Calculator for sulfur on nickel."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        baserate: BaseRate,
        temperature: float,
        lattice_directions: tuple[np.ndarray, np.ndarray],
        interaction: Callable[[float], float],
        cutoff_potential: float = 1.6e-22,
        sulfur_nickel_data: SulfurNickelData,
    ) -> None:
        super().__init__(
            baserate=baserate,
            temperature=temperature,
            lattice_directions=lattice_directions,
            interaction=interaction,
            cutoff_potential=cutoff_potential,
        )
        self._defect_indices = tuple(DEFECT_LOCATIONS.T)
        self.sulfur_nickel_data = sulfur_nickel_data
        self._layer_data = sulfur_nickel_data.sulfur_layers_data
        self._initial_layers = sulfur_nickel_data.initial_sulfur_layers
        self._layer_tiling_indices = sulfur_nickel_data.layer_tiling_indices
        self._layer_tiling_indices_mirrored = (
            sulfur_nickel_data.layer_tiling_indices_mirrored
        )

        self._cached_layers: np.ndarray | None = None
        self._cached_blocked_sites: np.ndarray = np.empty((0, 2), dtype=int)
        self._cached_layer_access_sites: np.ndarray = np.empty((0, 2), dtype=int)

    @override
    def _get_energy_landscape(
        self, positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        energies = super()._get_energy_landscape(positions)
        energies[self._defect_indices] -= 1.6e-20
        return energies

    def _get_rates_with_layers(  # noqa: PLR0914
        self,
        positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
        layer_access_sites: np.ndarray | None,
        blocked_sites: np.ndarray | None,
        layer_edge_sites: np.ndarray,
    ) -> list[np.ndarray]:
        energies = self._get_energy_landscape(positions=positions)
        rows, cols = np.nonzero(positions)

        _ = layer_edge_sites  # filler to stop error, will need to use this array to generate probabilities and destinations

        delta = JUMP_DIRECTIONS
        beta = 1 / (2 * Boltzmann * self._temperature)
        max_exp_arg = np.log(1 / np.min(self._baserate.grid[self._baserate.grid > 0]))

        # --- Base lattice moves ---
        neighbor_rows = (rows[:, None] + delta[:, 0]) % positions.shape[0]
        neighbor_cols = (cols[:, None] + delta[:, 1]) % positions.shape[1]

        energy_difference = (
            energies[neighbor_rows, neighbor_cols] - energies[rows, cols][:, None]
        )
        exponent = np.clip(-beta * energy_difference, a_min=None, a_max=max_exp_arg)
        rates = np.exp(exponent) * self._baserate.grid  # shape (n_particles, 9)

        # --- Blocked sites ---
        if blocked_sites is not None and len(blocked_sites) > 0:
            blocked_mask = np.zeros_like(positions, dtype=bool)
            blocked_mask[blocked_sites[:, 0], blocked_sites[:, 1]] = True
            blocked_neighbors = blocked_mask[neighbor_rows, neighbor_cols]
            rates[blocked_neighbors] = 0.0

        # --- Convert to list of arrays (so we can append layer moves) ---
        rates_list: list[np.ndarray] = [row.copy() for row in rates]

        # --- Handle layer access sites ---
        if layer_access_sites is not None and len(layer_access_sites) > 0:
            # build mapping from (r, c) → particle index
            site_to_idx = {
                (r, c): i for i, (r, c) in enumerate(zip(rows, cols, strict=True))
            }

            for site_x, site_y, _, _, _ in layer_access_sites:
                particle_idx = site_to_idx.get((site_x, site_y))
                if particle_idx is None:
                    continue  # no active particle at this site

                # compute energy difference using layer energy vs. self-energy
                delta_e = (
                    self.sulfur_nickel_data.layer_energy - energies[site_x, site_y]
                )
                exponent = np.clip(-beta * delta_e, a_min=None, a_max=max_exp_arg)
                rate = np.exp(exponent) * np.min(self._baserate.grid)

                # append to that particle's rate vector
                rates_list[particle_idx] = np.append(rates_list[particle_idx], rate)

        # --- Normalisation (ragged aware) ---
        final_rates: list[np.ndarray] = []
        for rate_vec in rates_list:
            row = rate_vec.copy()
            row_sum = row.sum()
            if row_sum > 1.0:
                row /= row_sum
            if row_sum > 0.5:  # noqa: PLR2004
                warnings.warn("Some probabilities exceed 0.5", stacklevel=2)

            # stationary probability (index 4 always)
            row[4] = 1 - row.sum()
            row = np.clip(row, 0.0, 1.0)
            final_rates.append(row)

        return final_rates

    def _get_blocked_and_available_sites(
        self,
        layers: np.ndarray,
        layer_data: np.ndarray,
        layer_indices: np.ndarray,
        layer_indices_mirrored: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_layers is None or not np.array_equal(
            layers, self._cached_layers
        ):
            self._cached_blocked_sites = _get_blocked_sites(
                layers=layers,
                layer_data=layer_data,
                layer_indices=layer_indices,
                layer_indices_mirrored=layer_indices_mirrored,
            )
            self._cached_layer_access_sites = _get_layer_access_sites(
                layers=layers,
                layer_data=layer_data,
                layer_indices=layer_indices,
                layer_indices_mirrored=layer_indices_mirrored,
            )

            self._cached_layers = np.copy(layers)

        return (self._cached_blocked_sites[:, :2], self._cached_layer_access_sites)

    @override
    def get_hopping_probabilities_and_destinations(
        self,
        positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
        layers: np.ndarray | None,
    ) -> tuple[
        list[np.ndarray[tuple[int, int], np.dtype[np.floating]]],
        list[np.ndarray[tuple[int, int], np.dtype[np.int_]]],
    ]:
        rows, cols = np.nonzero(positions)

        assert layers is not None

        blocked_sites, layer_access_sites = self._get_blocked_and_available_sites(
            layers=layers,
            layer_data=self._layer_data,
            layer_indices=self._layer_tiling_indices,
            layer_indices_mirrored=self._layer_tiling_indices_mirrored,
        )

        layer_edge_sites = _get_layer_edge_sites(
            layers=layers,
            layer_data=self._layer_data,
            layer_indices=self._layer_tiling_indices,
            layer_indices_mirrored=self._layer_tiling_indices_mirrored,
        )

        rates = self._get_rates_with_layers(
            positions=positions,
            layer_access_sites=layer_access_sites,
            blocked_sites=blocked_sites,
            layer_edge_sites=layer_edge_sites,
        )

        probabilities_list: list[np.ndarray] = []
        destinations_list: list[np.ndarray] = []

        shape = positions.shape
        rows, cols = np.nonzero(positions)
        for r, c, prob_row in zip(rows, cols, rates, strict=True):
            destinations = []
            # adding basic 9 directions
            for jump in JUMP_DIRECTIONS:  # includes stationary at index 4
                rr, cc = self._get_next_index((r, c), jump, shape)
                destinations.append((rr, cc, -1))  # ground level

            if len(layer_access_sites) > 0:
                # filter access sites for this particle
                mask = (layer_access_sites[:, 0] == r) & (layer_access_sites[:, 1] == c)
                for _, _, layer_num, layer_x, layer_y in layer_access_sites[mask]:
                    destinations.append((layer_x, layer_y, layer_num))

            probabilities_list.append(prob_row)
            destinations_list.append(np.array(destinations, dtype=int))

        return (probabilities_list, destinations_list)


def _get_blocked_sites(  # noqa: PLR0914
    layers: np.ndarray,
    layer_data: np.ndarray,
    layer_indices: np.ndarray,
    layer_indices_mirrored: np.ndarray,
) -> np.ndarray:
    """
    Generate a mapping of base sites that are blocked by filled layers.

    Return format is [blocked_site_x, blocked_site_y, layer_number, layer_x, layer_y].
    """
    layer_indices_lookup = {0: layer_indices, 1: layer_indices_mirrored}
    horizontal_vectors = {0: np.array([1, -3]), 1: np.array([1, 3])}
    vertical_vectors = {0: np.array([3, 1]), 1: np.array([3, -1])}

    num_tiles = (len(layer_indices) - 1) // 6 + 1
    half_layer_size = (layers.shape[2] - 1) // 2
    relative_positions = np.array(list(relative_tile_positions.values()))

    # Precompute lattice grids
    grid_cache: dict[int, np.ndarray] = {}
    for parity in (0, 1):
        x_grid, y_grid = np.meshgrid(
            np.arange(-num_tiles, num_tiles),
            np.arange(-num_tiles, num_tiles),
            indexing="ij",
        )
        tile_points = (
            x_grid[..., None] * horizontal_vectors[parity]
            + y_grid[..., None] * vertical_vectors[parity]
        )
        grid_cache[parity] = tile_points.reshape(-1, 2)

    records: list[list[int]] = []

    # Process each layer
    for layer_index, layer in enumerate(layers):
        orientation = int(layer_data[layer_index][1])
        parity = orientation % 2
        blocked_dictionary = blocked_hcp_sites[orientation]

        sulfur_positions = (
            grid_cache[parity][:, None, :] + relative_positions[None, :, :]
        ).reshape(-1, 2)

        # Keep only sulfur positions within bounds
        inside_bounds = (np.abs(sulfur_positions[:, 0]) <= half_layer_size) & (
            np.abs(sulfur_positions[:, 1]) <= half_layer_size
        )
        sulfur_positions = sulfur_positions[inside_bounds]

        x_indices = (sulfur_positions[:, 0] + half_layer_size).astype(int)
        y_indices = (sulfur_positions[:, 1] + half_layer_size).astype(int)

        # Check occupancy
        occupied = layer[x_indices, y_indices]
        if not np.any(occupied):
            continue

        sulfur_positions = sulfur_positions[occupied]
        x_indices, y_indices = x_indices[occupied], y_indices[occupied]

        index_grid = layer_indices_lookup[parity]
        tile_indices = index_grid[x_indices, y_indices]

        base_position = layer_data[layer_index][0]
        horizontal_vector = hcp_horizontal_vector[orientation]
        vertical_vector = hcp_vertical_vector[orientation]

        # Collect blocked sites + mapping to empty layer sites
        for pos, tile_idx, xi, yi in zip(
            sulfur_positions, tile_indices, x_indices, y_indices, strict=False
        ):
            site_offsets = blocked_dictionary[int(tile_idx)]
            if len(site_offsets) > 0:
                for offset in site_offsets:
                    base_site = (
                        base_position
                        + pos[0] * horizontal_vector
                        + pos[1] * vertical_vector
                        + np.array(offset, dtype=int)
                    )
                    records.append([base_site[0], base_site[1], layer_index, xi, yi])

    if records:
        return np.unique(np.array(records, dtype=int), axis=0)
    return np.empty((0, 5), dtype=int)


def _get_layer_access_sites(
    layers: np.ndarray,
    layer_data: np.ndarray,
    layer_indices: np.ndarray,
    layer_indices_mirrored: np.ndarray,
) -> np.ndarray:
    neighbour_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    new_layers = np.copy(layers)
    num_layers, num_rows, num_columns = layers.shape

    for dz in range(num_layers):
        true_positions = np.argwhere(layers[dz])

        # If the layer is completely empty → set center to True
        if true_positions.size == 0:
            center = (num_rows // 2, num_columns // 2)
            new_layers[dz, center[0], center[1]] = True
            continue

        for row, column in true_positions:
            for dr, dc in neighbour_offsets:
                rr = row + dr
                cc = column + dc
                if 0 <= rr < num_rows and 0 <= cc < num_columns:
                    new_layers[dz, rr, cc] = True

        # clear original sites
        new_layers[dz][layers[dz]] = False

    return _get_blocked_sites(
        layers=new_layers,
        layer_data=layer_data,
        layer_indices=layer_indices,
        layer_indices_mirrored=layer_indices_mirrored,
    )


def _get_layer_edge_sites(
    layers: np.ndarray,
    layer_data: np.ndarray,
    layer_indices: np.ndarray,
    layer_indices_mirrored: np.ndarray,
) -> np.ndarray:
    """Find particles which can leave layers, and their destinations."""
    num_layers, num_rows, num_columns = layers.shape
    edge_layers = np.zeros_like(layers, dtype=bool)

    neighbour_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    for dz in range(num_layers):
        true_positions = np.argwhere(layers[dz])

        for row, col in true_positions:
            for dr, dc in neighbour_offsets:
                rr = row + dr
                cc = col + dc
                if (
                    rr < 0
                    or rr >= num_rows
                    or cc < 0
                    or cc >= num_columns
                    or not layers[dz, rr, cc]
                ):
                    edge_layers[dz, row, col] = True
                    break

    return _get_blocked_sites(
        layers=edge_layers,
        layer_data=layer_data,
        layer_indices=layer_indices,
        layer_indices_mirrored=layer_indices_mirrored,
    )
