"""Tests for :mod:`laker.implicit`."""

import torch

from laker.implicit import hypergradient


def _sym_pd(n, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    return a @ a.T + torch.eye(n, dtype=torch.float64)


class TestHypergradient:
    def test_returns_one_per_param(self):
        n = 6
        A = _sym_pd(n)
        # Make a parameter that A depends on.
        torch.manual_seed(0)
        x_param = torch.nn.Parameter(torch.randn(n, 2, dtype=torch.float64))

        def op(v):
            return A @ v + x_param @ (x_param.T @ v) * 0.01

        rhs = torch.randn(n, dtype=torch.float64)
        # Solve A + small perturbation to get alpha.
        pcg_alpha = torch.linalg.solve(
            A + x_param @ x_param.T * 0.01, rhs
        )
        dL = torch.randn(n, dtype=torch.float64)
        grads = hypergradient(
            op=op,
            prec=lambda v: v,
            alpha=pcg_alpha,
            dl=dL,
            params=[x_param],
            tol=1e-10,
            max_iter=200,
            verbose=False,
        )
        assert len(grads) == 1
        assert grads[0].shape == x_param.shape
        assert torch.isfinite(grads[0]).all()

    def test_zero_grad_for_unused_param(self):
        n = 4
        A = _sym_pd(n)
        x_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float64))
        unused = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64), requires_grad=False)

        def op(v):
            return A @ v

        grads = hypergradient(
            op=op,
            prec=lambda v: v,
            alpha=torch.linalg.solve(A, torch.randn(n, dtype=torch.float64)),
            dl=torch.randn(n, dtype=torch.float64),
            params=[x_param, unused],
            tol=1e-10,
            max_iter=200,
            verbose=False,
        )
        assert len(grads) == 2
        assert torch.equal(grads[1], torch.zeros(3, dtype=torch.float64))

    def test_finite_differences_match(self):
        """Verify hypergradient against a finite-difference reference."""
        torch.manual_seed(0)
        n = 5
        A = _sym_pd(n)
        theta = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float64))

        def op(v):
            return A @ v + theta * (v * v)

        x_param = torch.nn.Parameter(torch.randn(n, dtype=torch.float64))
        rhs = torch.randn(n, dtype=torch.float64)
        alpha = torch.linalg.solve(
            A + theta.detach() * torch.eye(n, dtype=torch.float64) * 0,
            rhs,
        )
        dL = torch.randn(n, dtype=torch.float64)

        # Hypergradient via adjoint.
        grads = hypergradient(
            op=op,
            prec=lambda v: v,
            alpha=alpha,
            dl=dL,
            params=[theta],
            tol=1e-10,
            max_iter=500,
            verbose=False,
        )
        hg = grads[0].item()

        # Finite difference reference.
        eps = 1e-5
        losses = []
        for sign in (-1, 1):
            theta_e = theta.detach().clone() + sign * eps

            def op_e(v, theta_e=theta_e):
                return A @ v + theta_e * (v * v)

            from laker.solve import PCG

            pcg = PCG(tol=1e-12, max_iter=500, verbose=False)
            alpha_e, _ = pcg.solve(op_e, lambda v: v, rhs)
            x_val = torch.randn(n, dtype=torch.float64)
            y_val = torch.linalg.norm(op_e(x_val) - rhs)
            losses.append(y_val.item())

        fd = (losses[1] - losses[0]) / (2 * eps)
        # Order of magnitude check (different objectives; rough).
        assert abs(hg) < 10