from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import curve_fit  # type: ignore library types

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def autocorrelate(x: np.ndarray) -> np.ndarray:  # cspell:ignore ndarray
    """
    Compute the autocorrelation of the 1D signal x.

    Returns autocorrelation normalized to 1 at lag 0.
    """
    x -= np.mean(x)  # Remove mean
    result = np.correlate(x, x, mode="full")
    autocorr = result[result.size // 2 :]  # Take second half (non-negative lags)
    autocorr /= autocorr[0]  # Normalize
    return autocorr


"""Define a generic exponential for fitting to autocorrelation data"""


def exp_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Return a generic exponential function."""
    return a * np.exp(b * x) + c


"""Fit exponential and plot with autocorrelate data"""


def plot(autocorr: np.ndarray, t: np.ndarray, ax: Axes) -> tuple[Axes, float]:
    """Plot autocorrelation data with an exponential curve fit on a given axis.

    Args:
        autocorr (np.ndarray): Autocorrelated amplitude
        t (np.ndarray): Time
        axis (Axes): Passed axis

    Returns
    -------
        Axes: Returns plotted axes and fitted b value in exp(-bt)


    """
    popt, _pcov = curve_fit(exp_func, t, autocorr, p0=(1, -1, 1))  # type: ignore types defined by curve_fit
    popt = cast("np.ndarray", popt)
    b_fit: float = float(popt[1])

    ax.plot(t, autocorr, label="data")
    ax.plot(t, exp_func(t, *popt), "r-", label="Fitted Curve")
    ax.legend()
    ax.set_title("Autocorrelation of A")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(visible=True)

    return ax, b_fit
