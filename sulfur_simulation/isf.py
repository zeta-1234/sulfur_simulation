from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import curve_fit  # type: ignore library types

from sulfur_simulation.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import NDArray


def _get_autocorrelation(
    x: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Compute the autocorrelation of the 1D signal x.

    Returns autocorrelation normalized to 1 at lag 0.
    """
    x_centered = x - np.mean(x)  # Remove mean
    x_centered_magnitude = np.abs(x_centered)
    result = np.correlate(x_centered_magnitude, x_centered_magnitude, mode="full")
    autocorr = result[result.size // 2 :]  # Take second half (non-negative lags)
    autocorr /= autocorr[0]  # Normalize
    return autocorr


def _gaussian_decay_function(
    x: np.ndarray[Any, np.dtype[np.float64]], a: float, b: float, c: float
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return a generic exponential function."""
    return a * np.exp(b * x) + c


def plot_isf(
    x: np.ndarray[Any, np.dtype[np.float64]], t: np.ndarray, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    """Plot autocorrelation data with an exponential curve fit on a given axis."""
    fig, ax = get_figure(ax=ax)
    autocorrelation = _get_autocorrelation(x)

    optimal_params = _fit_gaussian_decay(t=t, autocorrelation=autocorrelation)

    ax.plot(t, autocorrelation, label="data")
    ax.plot(t, _gaussian_decay_function(t, *optimal_params), "r-", label="Fitted Curve")
    ax.legend()
    ax.set_title("Autocorrelation of A")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(visible=True)

    return fig, ax


def _fit_gaussian_decay(
    t: np.ndarray, autocorrelation: np.ndarray, slope_threshold: float = 1e-3
) -> NDArray[np.float64]:
    derivative = np.gradient(autocorrelation, t)
    flat_indices = np.where(np.abs(derivative) < slope_threshold)[0]
    cutoff_index = len(t) if len(flat_indices) == 0 else flat_indices[0]

    optimal_params, _ = curve_fit(  # type: ignore types defined by curve_fit
        _gaussian_decay_function,
        t[:cutoff_index],
        autocorrelation[:cutoff_index],
        p0=(1, -0.005, 1),
        bounds=([0, -np.inf, -np.inf], [np.inf, 0, np.inf]),
        maxfev=10000,
    )

    return cast("NDArray[np.float64]", optimal_params)


def get_dephasing_rates(amplitudes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate dephasing rates for all amplitudes."""
    dephasing_rates = np.empty(len(amplitudes))
    for i in range(len(amplitudes)):
        autocorrelation = _get_autocorrelation(amplitudes[i])
        optimal_params = _fit_gaussian_decay(t=t, autocorrelation=autocorrelation)
        dephasing_rates[i] = optimal_params[0] * -1
    return dephasing_rates


def plot_dephasing_rates(
    dephasing_rates: np.ndarray, delta_k: np.ndarray, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    """Plot dephasing rates against delta_k values."""
    fig, ax = get_figure(ax=ax)
    ax.plot(delta_k, dephasing_rates)
    ax.set_title("Dephasing rate vs Delta_K values")
    ax.set_xlabel("delta_k")
    ax.set_ylabel("dephasing rate")
    ax.grid(visible=True)

    return fig, ax


def get_isf(
    params: ISFParameters,
    positions: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Return complex amplitudes per particle, timestep, and wavevector."""
    n_particles = np.count_nonzero(positions[0])  # Assuming fixed number
    amplitudes = np.zeros(
        (positions.shape[0], n_particles, params.delta_k_array.shape[0]),
        dtype=np.complex128,
    )

    for time in range(amplitudes.shape[0]):
        coords = np.nonzero(positions[time])

        # phase = r · Δk
        phase = np.transpose(coords) @ params.delta_k_array.T
        amplitudes[time] = params.form_factor * np.exp(-1j * phase)

    return amplitudes


def _get_delta_k(
    n_points: int,
    delta_k_range: tuple[float, float],
    direction: tuple[float, float] = (1, 0),
) -> np.ndarray:
    """Return a matrix of delta_k values in the [1,0] direction."""
    abs_delta_k = np.linspace(
        start=delta_k_range[0], stop=delta_k_range[1], num=n_points, endpoint=True
    )

    return np.asarray(direction)[np.newaxis, :] * abs_delta_k[:, np.newaxis]


@dataclass(kw_only=True, frozen=True)
class ISFParameters:
    """Parameters for plotting results of simulation."""

    n_delta_k_intervals: int
    """The number of delta_k values to use"""
    delta_k_max: float
    """The max value of delta_k"""
    delta_k_min: float = 0.1
    """The min value of delta_k"""
    form_factor: float = 1
    """Prefactor for scattered amplitude"""

    def __post_init__(self) -> None:
        if self.delta_k_min == 0:
            msg = "delta_k_min should not be zero"
            raise ValueError(msg)

    @property
    def delta_k_array(self) -> np.ndarray:
        """All delta_k values."""
        return _get_delta_k(
            n_points=self.n_delta_k_intervals,
            delta_k_range=(self.delta_k_min, self.delta_k_max),
        )
