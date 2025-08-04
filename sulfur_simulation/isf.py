from __future__ import annotations

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


def plot_autocorrelation(
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
    )

    return cast("NDArray[np.float64]", optimal_params)


def get_dephasing_rates(amplitudes: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate dephasing rates for all amplitudes."""
    dephasing_rates = np.empty(len(amplitudes))
    for i in range(len(amplitudes)):
        autocorrelation = _get_autocorrelation(amplitudes[i])
        optimal_params = _fit_gaussian_decay(t=t, autocorrelation=autocorrelation)
        dephasing_rates[i] = optimal_params[1] * -1
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
  