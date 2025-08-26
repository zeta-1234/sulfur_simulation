from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.fft  # type: ignore[reportMissingTypeStubs]
from scipy.optimize import curve_fit  # type: ignore library types
from tqdm import trange

from sulfur_simulation.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import NDArray

    from sulfur_simulation.scattering_calculation import (
        SimulationParameters,
        SimulationResult,
    )


def _get_delta_k(
    params: SimulationParameters,
    delta_k_max: float,
    direction: tuple[float, float] = (1, 0),
) -> np.ndarray:
    """Return a matrix of delta_k values in the [1,0] direction."""
    delta_k_interval = (2 * np.pi) / (params.lattice_dimension[0])
    abs_delta_k = np.arange(
        start=delta_k_interval, stop=delta_k_max, step=delta_k_interval
    )

    return np.asarray(direction)[np.newaxis, :] * abs_delta_k[:, np.newaxis]


@dataclass(kw_only=True, frozen=True)
class ISFParameters:
    """Parameters for plotting results of simulation."""

    delta_k_max: float = 2 * np.pi
    """The max value of delta_k"""
    params: SimulationParameters
    """Simulation parameters."""
    form_factor: float = 1
    """Prefactor for scattered amplitude"""

    @property
    def n_delta_k(self) -> int:
        """Number of delta_k values."""
        return self.delta_k_array.shape[0]

    @property
    def delta_k_array(self) -> np.ndarray:
        """All delta_k values."""
        return _get_delta_k(delta_k_max=self.delta_k_max, params=self.params)


def _get_autocorrelation(
    amplitudes: np.ndarray,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Compute the autocorrelation of a complex-valued 1D signal."""
    n = len(amplitudes)
    padded_length = 2 * n - 1
    padded = np.zeros(padded_length, dtype=amplitudes.dtype)
    padded[:n] = amplitudes
    transformed = np.asarray(scipy.fft.fft(padded))
    power_spectrum = np.abs(transformed) ** 2
    autocorrelation = np.asarray(scipy.fft.ifft(power_spectrum))
    return (autocorrelation / (autocorrelation[0])).real[:n]


def _gaussian_decay_function(
    x: np.ndarray[Any, np.dtype[np.float64]], a: float, b: float, c: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return a generic exponential function."""
    return a * np.exp(b * x) + c - 1000 * min(c, 0)


def plot_isf(
    x: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    isf_params: ISFParameters,
    delta_k_index: int,
    t: np.ndarray,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot autocorrelation data with an exponential curve fit on a given axis."""
    all_amplitudes = np.array(x)
    all_autocorrelations = np.array(
        [_get_autocorrelation(amplitude[delta_k_index]) for amplitude in all_amplitudes]
    )
    mean_autocorrelation = all_autocorrelations.mean(axis=0)
    denom = all_autocorrelations.shape[0] - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        std_autocorrelation = all_autocorrelations.std(axis=0) / np.sqrt(denom)

    fig, ax = get_figure(ax=ax)

    optimal_params = _fit_gaussian_decay(
        t=t,
        autocorrelation=mean_autocorrelation,
        std_autocorrelation=std_autocorrelation
        if all_autocorrelations.shape[0] > 1
        else None,
    )

    ax.plot(t, mean_autocorrelation, label="Mean ISF")
    if all_autocorrelations.shape[0] > 1:
        ax.fill_between(
            t,
            mean_autocorrelation - std_autocorrelation,
            mean_autocorrelation + std_autocorrelation,
            color="gray",
            alpha=0.3,
            label="±1 std",
        )

    ax.plot(t, _gaussian_decay_function(t, *optimal_params), "r-", label="Fitted Curve")
    ax.legend()
    ax.set_title(f"ISF of A for delta_k = {isf_params.delta_k_array[delta_k_index]}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(visible=True)

    return fig, ax


def _fit_gaussian_decay(
    t: np.ndarray,
    autocorrelation: np.ndarray,
    std_autocorrelation: np.ndarray | None = None,
    slope_threshold: float = 1e-3,
) -> NDArray[np.float64]:
    shortest_valid_length = 10
    derivative = np.gradient(autocorrelation, t)
    flat_indices = np.where(np.abs(derivative) < slope_threshold)[0]
    cutoff_index = (
        len(t)
        if len(flat_indices) == 0
        else valid_cutoffs[0]
        if len(valid_cutoffs := flat_indices[flat_indices >= shortest_valid_length]) > 0
        else flat_indices[0]
    )
    if std_autocorrelation is not None:
        std_autocorrelation[std_autocorrelation == 0] = 1e-8
    optimal_params, _ = curve_fit(  # type: ignore types defined by curve_fit
        _gaussian_decay_function,
        t[:cutoff_index],
        autocorrelation[:cutoff_index],
        p0=(1, -0.005, 1),
        bounds=([0, -np.inf, -np.inf], [np.inf, 0, np.inf]),
        maxfev=100000,
        sigma=None
        if std_autocorrelation is None
        else std_autocorrelation[:cutoff_index],
        absolute_sigma=True,
    )

    return cast("NDArray[np.float64]", optimal_params)


def get_dephasing_rates(
    amplitudes: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    t: np.ndarray,
) -> np.ndarray:
    """Calculate dephasing rates for all amplitudes."""
    all_amplitudes = np.array(amplitudes)
    mean_amplitudes = all_amplitudes.mean(axis=0)
    dephasing_rates = np.empty(len(mean_amplitudes))
    for i in trange(len(mean_amplitudes)):
        autocorrelation = _get_autocorrelation(mean_amplitudes[i])
        optimal_params = _fit_gaussian_decay(t=t, autocorrelation=autocorrelation)
        dephasing_rates[i] = optimal_params[1] * -1
    return dephasing_rates


def plot_dephasing_rates(
    dephasing_rates: np.ndarray, delta_k: np.ndarray, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    """Plot dephasing rates against delta_k values."""
    fig, ax = get_figure(ax=ax)
    ax.plot(delta_k, dephasing_rates)
    ax.set_title("Dephasing rate vs delta_k values")
    ax.set_xlabel("delta_k")
    ax.set_ylabel("dephasing rate")
    ax.grid(visible=True)

    return fig, ax


def get_amplitudes(
    isf_params: ISFParameters,
    positions: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Return summed complex amplitudes for each delta_k (rows) and timestep (columns)."""
    n_timesteps, dimension, _ = positions.shape
    n_delta_k = isf_params.n_delta_k

    amplitudes = np.zeros((n_delta_k, n_timesteps), dtype=np.complex128)

    # Precompute (row, col) coordinates for all grid points for speed
    all_coords = np.column_stack(
        np.unravel_index(np.arange(dimension * dimension), (dimension, dimension))
    )

    for t in trange(n_timesteps):
        coords = all_coords[positions[t].ravel()]

        phase = coords @ isf_params.delta_k_array.T

        amp = isf_params.form_factor * np.exp(-1j * phase)
        amplitudes[:, t] = np.sum(amp, axis=0)

    return amplitudes


def get_all_amplitudes(
    isf_params: ISFParameters,
    results: list[SimulationResult],
) -> list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
    """Get amplitudes for all runs and return as a list."""
    return [get_amplitudes(isf_params, positions=run.positions) for run in results]
