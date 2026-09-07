"""Matplotlib-backed plotting for LAKER outputs.

Public class :class:`Plot` with single-word static methods.
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
        size: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> np.ndarray:
        """Reshape flat predictions on a regular grid to a 2-D image."""
        return predictions.detach().cpu().numpy().reshape(size, size)

    @staticmethod
    def field(
        predictions: torch.Tensor,
        size: int,
        title: str = "Radio Map Reconstruction",
        extent: Optional[Tuple[float, float, float, float]] = None,
        label: str = "RSS (dBm)",
        figsize: Tuple[int, int] = (6, 5),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """Plot a 2-D radio-map reconstruction.

        Returns ``(figure, axes)`` from matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required. Install with: pip install matplotlib"
            ) from exc

        fig, ax = plt.subplots(figsize=figsize)
        img = Plot.image(predictions, size)
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
        fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        logger.info("Plotted radio map: %d x %d", size, size)
        return fig, ax

    @staticmethod
    def convergence(
        gaps: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Convergence Behaviour",
        xlabel: str = "Iteration",
        ylabel: str = "Relative Objective Gap",
        figsize: Tuple[int, int] = (6, 4),
    ):
        """Plot convergence curves for one or more solvers.

        Returns ``(figure, axes)`` from matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required. Install with: pip install matplotlib"
            ) from exc

        fig, ax = plt.subplots(figsize=figsize)
        for i, g in enumerate(gaps):
            label = labels[i] if labels and i < len(labels) else f"Solver {i + 1}"
            ax.semilogy(g, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        logger.info("Plotted convergence: %d series", len(gaps))
        return fig, ax


__all__ = ["Plot"]
