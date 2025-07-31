from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import curve_fit  # type: ignore library types

from sulfur_simulation.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


def _get_autocorrelation(
    x: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Compute the autocorrelation of the 1D signal x.

    Returns autocorrelation normalized to 1 at lag 0.
    """
    x -= np.mean(x)  # Remove mean
    result = np.correlate(x, x, mode="full")
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
    optimal_params, _ = curve_fit(  # type: ignore types defined by curve_fit
        _gaussian_decay_function, t, autocorrelation, p0=(1, -1, 1)
    )
    optimal_params = cast("tuple[float, float, float]", optimal_params)

    ax.plot(t, autocorrelation, label="data")
    ax.plot(t, _gaussian_decay_function(t, *optimal_params), "r-", label="Fitted Curve")
    ax.legend()
    ax.set_title("Autocorrelation of A")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(visible=True)

    return fig, ax
