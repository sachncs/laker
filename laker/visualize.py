"""Backward-compat shim. New code should use ``laker.plot.Plot``.

Re-exports the legacy ``Visualizer`` class and ``plot_radio_map`` /
``plot_convergence`` / ``radio_map_to_image`` free functions so existing
tests and downstream callers continue to work after the move to
:class:`Plot`.
"""
from __future__ import annotations

from laker._visualize_impl import (
    Visualizer,
    _radio_map_to_image as radio_map_to_image,
    _plot_radio_map as plot_radio_map,
    _plot_convergence as plot_convergence,
)

__all__ = [
    "Visualizer",
    "plot_radio_map",
    "plot_convergence",
    "radio_map_to_image",
]
