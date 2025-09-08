from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from sulfur_simulation.hopping_calculator import BaseRate, InteractingHoppingCalculator

if TYPE_CHECKING:
    from collections.abc import Callable
import warnings

import numpy as np
from scipy.constants import Boltzmann  # type: ignore[reportMissingTypeStubs]

from sulfur_simulation.scattering_calculation import JUMP_DIRECTIONS
from sulfur_simulation.sulfur_data import (
    DEFECT_LOCATIONS,
    blocked_hcp_sites,
    hcp_horizontal_vector,
    hcp_vertical_vector,
    relative_tile_positions,
)


@dataclass(kw_only=True, frozen=True)
class SulfurNickelData:
    """Parameters for the Sulfur on Nickel simulation."""

    max_layer_size: int
    "The maximum dimension a sulfur structure can grow to"

    def __post_init__(self) -> None:
        if self.max_layer_size % 2 == 0:
            msg = "max_layer_size must be an odd integer."
            raise ValueError(msg)

    @property
    def layer_indices(self) -> np.ndarray:
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
    def layer_indices_mirrored(self) -> np.ndarray:
        """The mirrored indices of a sulfur layer."""
        return self.layer_indices[:, ::-1]

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
        self._layer_data = sulfur_nickel_data.sulfur_layers_data
        self._layers = sulfur_nickel_data.initial_sulfur_layers
        self._layer_indices = sulfur_nickel_data.layer_indices
        self._layer_indices_mirrored = sulfur_nickel_data.layer_indices_mirrored

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

    @override
    def get_hopping_probabilities(  # noqa: PLR0914
        self,
        positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
        layers: np.ndarray | None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
        #  TODO: need checks with layers:  # noqa: FIX002
        #  layers must be filled in order, starting at the defect
        #  if a particle near possible layer spots, check to see if it moves into those layers first,
        #  then return probabilities of moving to other non-layer spots if not
        #  if a particle exists in a higher layer that blocks the spot, set probabilities to zero
        #  for speed, maybe keep an array of positions which are blocked by higher layers,
        #  and positions which can result in jumps to higher layers
        if layers is None:
            layers = self._layers
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

        rates = np.exp(exponent) * self._baserate.grid

        if self._cached_layers is None or not np.array_equal(
            layers, self._cached_layers
        ):
            self._cached_blocked_sites = _get_blocked_sites(
                layers=layers,
                layer_data=self._layer_data,
                layer_indices=self._layer_indices,
                layer_indices_mirrored=self._layer_indices_mirrored,
            )
            self._cached_layer_access_sites = _get_layer_access_sites(
                layers=layers,
                layer_data=self._layer_data,
                layer_indices=self._layer_indices,
                layer_indices_mirrored=self._layer_indices_mirrored,
            )

            self._cached_layers = np.copy(layers)

        blocked_sites = self._cached_blocked_sites

        if len(blocked_sites) > 0:
            blocked_mask = np.zeros_like(positions, dtype=bool)
            blocked_mask[blocked_sites[:, 0], blocked_sites[:, 1]] = True

            blocked_neighbors = blocked_mask[neighbor_rows, neighbor_cols]
            rates[blocked_neighbors] = 0.0

        row_sums = rates.sum(axis=1)
        over_rows = row_sums > 1.0
        rates[over_rows] /= row_sums[over_rows, None]

        warning_threshold = 0.5
        if np.any(row_sums > warning_threshold):
            warnings.warn("Some probabilities exceed 0.5", stacklevel=2)

        # Stationary probability
        rates[:, 4] = 1 - rates.sum(axis=1)

        return np.clip(rates, 0.0, 1.0)


def _get_blocked_sites(  # noqa: PLR0914
    layers: np.ndarray,
    layer_data: np.ndarray,
    layer_indices: np.ndarray,
    layer_indices_mirrored: np.ndarray,
) -> np.ndarray:
    """Generate a list of inaccessible base sites due to filled layers."""
    blocked_sites = []
    layer_indices_dict = {0: layer_indices, 1: layer_indices_mirrored}

    tile_horizontal_vector = {0: np.array([1, -3]), 1: np.array([1, 3])}
    tile_vertical_vector = {0: np.array([3, 1]), 1: np.array([3, -1])}

    n_tiles = int((len(layer_indices) - 1) // 6) + 1
    half_size = (layers.shape[1] - 1) // 2

    # Precompute relative positions
    rel_positions = np.array(list(relative_tile_positions.values()))

    for i in range(len(layers)):
        orientation = layer_data[i][1]
        blocked_site_dictionary = blocked_hcp_sites[orientation]

        # Generate tile grid
        xv, yv = np.meshgrid(
            np.arange(-n_tiles, n_tiles), np.arange(-n_tiles, n_tiles), indexing="ij"
        )
        tile_points = (
            xv[..., None] * tile_horizontal_vector[orientation % 2]
            + yv[..., None] * tile_vertical_vector[orientation % 2]
        )
        tile_points = tile_points.reshape(-1, 2)  # flatten grid

        # Add relative positions → candidate sulfur coordinates
        sulfur_positions = tile_points[:, None, :] + rel_positions[None, :, :]
        sulfur_positions = sulfur_positions.reshape(-1, 2)

        # Check bounds
        mask = (np.abs(sulfur_positions[:, 0]) <= half_size) & (
            np.abs(sulfur_positions[:, 1]) <= half_size
        )
        sulfur_positions = sulfur_positions[mask]

        # Convert to indices in layer grid
        xs = (sulfur_positions[:, 0] + half_size).astype(int)
        ys = (sulfur_positions[:, 1] + half_size).astype(int)

        # Keep only positions where there is a particle
        occupied_mask = layers[i, xs, ys]
        sulfur_positions = sulfur_positions[occupied_mask]
        xs, ys = xs[occupied_mask], ys[occupied_mask]

        # Map to layer index and then to blocked sites
        indices = layer_indices_dict[orientation % 2][xs, ys]
        for sulfur_pos, idx in zip(sulfur_positions, indices, strict=False):
            blocked_sites.extend(
                layer_data[i][0]
                + sulfur_pos[0] * hcp_horizontal_vector[orientation]
                + sulfur_pos[1] * hcp_vertical_vector[orientation]
                + tile_vector
                for tile_vector in blocked_site_dictionary[int(idx)]
            )

    return np.unique(np.array(blocked_sites, dtype=int), axis=0)


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
            new_layers[dz, center] = True
            continue

        for row, column in true_positions:
            for dr, dc in neighbour_offsets:
                rr = row + dr
                cc = column + dc
                if 0 <= rr < num_rows and 0 <= cc < num_columns:
                    new_layers[dz, rr, cc] = True

        # clear original sites
        new_layers[dz, layers[dz]] = False

    return _get_blocked_sites(
        layers=new_layers,
        layer_data=layer_data,
        layer_indices=layer_indices,
        layer_indices_mirrored=layer_indices_mirrored,
    )
