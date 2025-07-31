from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


def get_figure(ax: Axes | None = None) -> tuple[Figure | SubFigure, Axes]:
    """Get a figure and axes for plotting."""
    if ax is None:
        return plt.subplots(figsize=(10, 6))
    return ax.figure, ax
