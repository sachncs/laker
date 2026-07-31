"""LAKER: Learning-based Attention Kernel Regression.

LAKER is a PyTorch implementation of the algorithm from
Tao & Tan (2026), "Accelerating Regularized Attention Kernel Regression
for Spectrum Cartography". It solves regularised attention kernel
regression problems of the form

.. math::

    \\min_\\alpha \\|G \\alpha - y\\|_2^2 + \\lambda \\alpha^\\top G \\alpha

where :math:`G = \\exp(E E^\\top)` is an exponential attention kernel
induced by learned embeddings :math:`E`. The dominant cost is solving
the linear system :math:`(G + \\lambda I)\\alpha = y`, which LAKER
accelerates with a learned data-dependent preconditioner built by a
shrinkage-regularised Convex-Concave Procedure (CCCP). The preconditioner
reduces the system condition number by up to three orders of magnitude
and yields near size-independent Preconditioned Conjugate Gradient (PCG)
convergence.

The single public type is :class:`Laker`. The rest of the package
exposes the building blocks under single-word names:

* :class:`~laker.backend.Backend` — device, dtype, compile, seed.
* :class:`~laker.check.Check` — input validation and tensor coercion.
* :class:`~laker.data.Data` — synthetic radio field generation.
* :class:`~laker.embed.Embed` — abstract encoder.
* :class:`~laker.embed.Position`, :class:`~laker.embed.Visual` — concrete encoders.
* :class:`~laker.math.Math` — numerical helpers; :class:`~laker.math.GP` — BO surrogate.
* :class:`~laker.core.Core` — embed → kernel → preconditioner → solve → predict.
* :class:`~laker.kernel.Exact`, :class:`~laker.kernel.Nystrom`,
  :class:`~laker.kernel.Fourier`, :class:`~laker.kernel.Neighbors`,
  :class:`~laker.kernel.Grid`, :class:`~laker.kernel.Hybrid`,
  :class:`~laker.kernel.Spectrum` — kernel operators.
* :class:`~laker.solve.PCG`, :class:`~laker.solve.Descent`,
  :class:`~laker.solve.Jacobi`, :class:`~laker.solve.Report` — solvers.
* :class:`~laker.prec.CCCP`, :class:`~laker.prec.Adaptive` — preconditioners.
* :class:`~laker.distributed.Distributed` — multi-GPU wrapper.
* :class:`~laker.search.Search` — grid / Bayesian hyperparameter search.
* :class:`~laker.bilevel.Bilevel` — bilevel optimisation.
* :class:`~laker.implicit.hypergradient` — adjoint method.
* :class:`~laker.corrector.Corrector` — residual corrector MLP.
* :class:`~laker.train.Trainer` — embedding / corrector / uncertainty training.
* :class:`~laker.stream.Stream` — incremental updates and path fitting.
* :class:`~laker.store.Store` — model save / load.
* :class:`~laker.plot.Plot` — radio-map and convergence plots.
* :class:`~laker.bench.Bench`, :class:`~laker.bench.SolveBench`,
  :class:`~laker.bench.BaseBench` — benchmarking harness.
* :class:`~laker.cli.CLI` — command-line interface.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("laker")
except PackageNotFoundError:
    __version__ = "0.5.0+local"

from laker.model import Laker

__all__ = ["Laker", "__version__"]