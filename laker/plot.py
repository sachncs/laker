"""Matplotlib-backed plotting for LAKER outputs.

The single public type is :class:`Plot`; helpers are static methods.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Plot:
    """Static-method helpers for radio-map and convergence plotting."""

    @staticmethod
    def image(
        predictions: torch.Tensor,
        grid_size: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> np.ndarray:
        """Reshape flat predictions on a regular grid to a 2-D image.

        Args:
            predictions: Flat tensor of shape ``(grid_size**2,)``.
            grid_size: Number of grid points per axis.
            x_min, x_max, y_min, y_max: Spatial extent (currently
                unused; reserved for future axis labelling).

        Returns:
            NumPy array of shape ``(grid_size, grid_size)`` with x
            varying horizontally and y varying vertically.
        """
        return predictions.detach().cpu().numpy().reshape(grid_size, grid_size)

    @staticmethod
    def field(
        predictions: torch.Tensor,
        grid_size: int,
        title: str = "Radio Map Reconstruction",
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: str = "RSS (dBm)",
        figsize: Tuple[int, int] = (6, 5),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Plot a 2-D radio-map reconstruction.

        Args:
            predictions: Flat predictions of shape ``(grid_size**2,)``.
            grid_size: Points per axis.
            title: Plot title.
            extent: ``(x_min, x_max, y_min, y_max)``.
            colorbar_label: Colorbar label.
            figsize: Figure size.
            vmin, vmax: Color limits.

        Returns:
            ``(figure, axes)`` tuple from matplotlib.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        img = Plot.image(predictions, grid_size)
        im = ax.imshow(
            img,
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=colorbar_label)
        fig.tight_layout()
        logger.info("Plotted radio map: %d x %d", grid_size, grid_size)
        return fig, ax

    @staticmethod
    def convergence(
        objective_gaps: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Convergence Behaviour",
        xlabel: str = "Iteration",
        ylabel: str = "Relative Objective Gap",
        figsize: Tuple[int, int] = (6, 4),
    ):
        """Plot convergence curves for one or more solvers.

        Args:
            objective_gaps: Per-solver list of objective gaps.
            labels: Optional labels.
            title, xlabel, ylabel, figsize: Plot styling.

        Returns:
            ``(figure, axes)`` tuple from matplotlib.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        for idx, gaps in enumerate(objective_gaps):
            label = labels[idx] if labels and idx < len(labels) else f"Solver {idx + 1}"
            ax.semilogy(gaps, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        logger.info("Plotted convergence curves: %d series", len(objective_gaps))
        return fig, ax


__all__ = ["Plot"]
