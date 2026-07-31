"""LAKER: Learning-based Attention Kernel Regression.

LAKER is a PyTorch implementation of the algorithm from
Tao & Tan (2026), "Accelerating Regularized Attention Kernel Regression for
Spectrum Cartography". It solves regularised attention kernel regression
problems of the form

.. math::

    \\min_\\alpha \\|G \\alpha - y\\|_2^2 + \\lambda \\alpha^\\top G \\alpha

where :math:`G = \\exp(E E^\\top)` is an exponential attention kernel
induced by learned embeddings :math:`E`. The dominant computational cost is
solving the linear system :math:`(G + \\lambda I)\\alpha = y`, which LAKER
accelerates with a **learned data-dependent preconditioner** built by a
shrinkage-regularised Convex-Concave Procedure (CCCP). The preconditioner
reduces the system condition number by up to three orders of magnitude and
yields near size-independent Preconditioned Conjugate Gradient (PCG)
convergence.

The public surface of LAKER is a single class — :class:`Laker` —
importable as ``from laker import Laker``. Secondary classes live under
their module-qualified names:

- ``laker.helpers.Helpers``
- ``laker.backend.Backend``
- ``laker.base.Base``
- ``laker.embed.Position``, ``laker.embed.Visual``, ``laker.embed.Embed``
- ``laker.data.Data``
- ``laker.plot.Plot``
- ``laker.preconditioner.Preconditioner`` (and ``.cccp``, ``.adaptive``, ``.jacobi``)
- ``laker.solve.Solve`` (and ``.pcg``, ``.descent``, ``.jacobi``)
- ``laker.search.Search``
- ``laker.fit.Fit``
- ``laker.stream.Stream``
- ``laker.implicit.Implicit``
- ``laker.cli.CLI``

The default numerical configuration is single-precision
(``torch.float32``); switching to ``dtype=torch.float64`` is recommended
for the most ill-conditioned problems. Environment variables documented in
``.env.example`` and ``README.md`` configure the runtime defaults.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("laker")
except PackageNotFoundError:
    __version__ = "0.4.0+local"

from laker.model import Laker

__all__ = ["Laker", "__version__"]
