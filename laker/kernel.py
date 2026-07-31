"""Attention kernel operators for matrix-free linear algebra.

All operators conform to the :class:`Kernel` protocol and implement
:meth:`matvec`, :meth:`diag`, :meth:`dense`, and :meth:`eval`.

Operators (exact → most compressed):

* :class:`Exact` — direct ``O(n^2)`` evaluation.
* :class:`Nystrom` — Nyström low-rank ``O(n m)``.
* :class:`Fourier` — random Fourier features ``O(n r)``.
* :class:`Neighbors` — sparse k-NN graph.
* :class:`Grid` — structured kernel interpolation (product grid).
* :class:`Hybrid` — Nyström + sparse k-NN blend.
* :class:`Spectrum` — learned spectral shaping via :class:`Shaper`.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Tuple

import torch
import torch.nn as nn

from laker.backend import Backend
from laker.math import Math

logger = logging.getLogger(__name__)


class Kernel(Protocol):
    """Protocol for matrix-free kernel operators."""

    n: int
    dim: int
    lam: float
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int]

    def matvec(self, x: torch.Tensor) -> torch.Tensor: ...
    def diag(self) -> torch.Tensor: ...
    def dense(self) -> torch.Tensor: ...
    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Safe element-wise exponential
# ---------------------------------------------------------------------------


def exp_safe(gram: torch.Tensor, out: Optional[torch.Tensor] = None, skip: bool = False) -> torch.Tensor:
    """Element-wise exp with dtype-aware overflow guard.

    Thin alias for :meth:`Math.exp` that also supports the legacy
    ``out=`` argument for in-place kernels.
    """
    if not skip:
        if gram.requires_grad:
            clamped = Math.exp(gram)
            return torch.exp(clamped) if out is None else torch.exp(clamped, out=out)
        result = Math.exp(gram)
        if out is not None and result is not gram:
            out.copy_(result)
            return out
        return result
    if gram.requires_grad or out is None:
        return torch.exp(gram)
    return torch.exp(gram, out=out)


def exact_matvec(
    embeddings: torch.Tensor,
    lam: float,
    x: torch.Tensor,
    skip: bool = False,
) -> torch.Tensor:
    """Apply the non-chunked exact attention matvec."""
    gram = embeddings @ embeddings.T
    gram = exp_safe(gram, out=gram, skip=skip)
    return lam * x + gram @ x


# ---------------------------------------------------------------------------
# Exact attention kernel
# ---------------------------------------------------------------------------


class Exact:
    """Matrix-free operator for ``G = exp(E E^T)``.

    For large ``n`` the dense matrix is never materialised; matvecs are
    computed in optionally chunked blocks to respect GPU memory limits.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        chunk: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.chunk = chunk

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype

        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)
        self.shape = (self.size, self.size)

        max_sq = torch.sum(self.embeddings ** 2, dim=1).max().item()
        if self.dtype == torch.float16:
            cap = 11.0
        elif self.dtype == torch.float32:
            cap = 80.0
        else:
            cap = 700.0
        self.skip = max_sq < cap

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.size}, dim={self.dim}, lam={self.lam})"

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``(lambda I + G)`` to vector(s) ``x``."""
        if x.dim() not in (1, 2):
            raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
        if x.shape[0] != self.size:
            raise ValueError(f"x must have {self.size} rows, got {x.shape[0]}")
        return self._impl(x)

    def _impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.chunk is None or self.size <= self.chunk:
            return exact_matvec(self.embeddings, self.lam, x, skip=self.skip)

        out = self.lam * x
        chunk_size = self.chunk
        size = self.size
        element_size = self.dtype.itemsize if hasattr(self.dtype, "itemsize") else 4
        mem_per_chunk = chunk_size * size * element_size
        if mem_per_chunk <= Backend.chunk:
            for start in range(0, size, chunk_size):
                end = min(start + chunk_size, size)
                gram_chunk = self.embeddings[start:end] @ self.embeddings.T
                gram_chunk = exp_safe(gram_chunk, out=gram_chunk, skip=self.skip)
                if x.dim() == 1:
                    out[start:end].addmv_(gram_chunk, x)
                else:
                    out[start:end].addmm_(gram_chunk, x)
            return out

        if x.dim() == 1:
            for i_start in range(0, size, chunk_size):
                i_end = min(i_start + chunk_size, size)
                accum = torch.zeros(i_end - i_start, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, size, chunk_size):
                    j_end = min(j_start + chunk_size, size)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    gram_block = exp_safe(gram_block, out=gram_block, skip=self.skip)
                    accum.addmv_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        else:
            k = x.shape[1]
            for i_start in range(0, size, chunk_size):
                i_end = min(i_start + chunk_size, size)
                accum = torch.zeros(i_end - i_start, k, device=self.device, dtype=self.dtype)
                e_i = self.embeddings[i_start:i_end]
                for j_start in range(0, size, chunk_size):
                    j_end = min(j_start + chunk_size, size)
                    gram_block = e_i @ self.embeddings[j_start:j_end].T
                    gram_block = exp_safe(gram_block, out=gram_block, skip=self.skip)
                    accum.addmm_(gram_block, x[j_start:j_end])
                out[i_start:i_end].add_(accum)
        return out

    def diag(self) -> torch.Tensor:
        """Return diagonal of ``lambda I + G``."""
        sq = torch.sum(self.embeddings ** 2, dim=1)
        return self.lam + exp_safe(sq)

    def dense(self) -> torch.Tensor:
        """Materialise the full dense ``(lambda I + G)`` matrix (debug only)."""
        gram = self.embeddings @ self.embeddings.T
        gram = exp_safe(gram, out=gram)
        gram.diagonal().add_(self.lam)
        return gram

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate the attention kernel between two sets of points."""
        if y is None:
            y = self.embeddings
        if chunk is None:
            chunk = self.chunk
        m = x.shape[0]
        p = y.shape[0]
        if chunk is None or m <= chunk:
            gram = x @ y.T
            return exp_safe(gram, out=gram, skip=self.skip)

        element_size = self.dtype.itemsize if hasattr(self.dtype, "itemsize") else 4
        mem_per_chunk = chunk * p * element_size
        if mem_per_chunk <= Backend.chunk:
            out = torch.empty(m, p, device=self.device, dtype=self.dtype)
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                gram_chunk = x[start:end] @ y.T
                gram_chunk = exp_safe(gram_chunk, out=gram_chunk, skip=self.skip)
                out[start:end] = gram_chunk
            return out

        out = torch.empty(m, p, device=self.device, dtype=self.dtype)
        for i_start in range(0, m, chunk):
            i_end = min(i_start + chunk, m)
            for j_start in range(0, p, chunk):
                j_end = min(j_start + chunk, p)
                gram_block = x[i_start:i_end] @ y[j_start:j_end].T
                gram_block = exp_safe(gram_block, out=gram_block, skip=self.skip)
                out[i_start:i_end, j_start:j_end] = gram_block
        return out


# ---------------------------------------------------------------------------
# Low-rank approximations
# ---------------------------------------------------------------------------


def _nystrom_matvec(cross_kernel: torch.Tensor, landmark_projection: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Core matmul for the Nyström approximation: K_approx @ x = cross_kernel @ landmark_projection @ cross_kernel^T @ x."""
    return cross_kernel @ (landmark_projection.T @ x)


_nystrom_matvec = Backend.compile(
    lambda cross_kernel, landmark_projection, x: cross_kernel @ (landmark_projection.T @ x)
)


class Nystrom:
    """Nyström low-rank approximation of the attention kernel.

    Approximates ``G = exp(E E^T)`` using ``m`` landmark points so
    ``matvec`` costs ``O(n*m)``.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        num: Optional[int] = None,
        method: str = "greedy",
        pilot: int = 1000,
        chunk: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.chunk = chunk
        self.method = method
        self.pilot = pilot

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        num_landmarks_v = num if num is not None else max(50, int(self.size ** 0.5))
        self.num_landmarks = min(num_landmarks_v, self.size)
        if self.num_landmarks != num_landmarks_v:
            logger.warning(
                "num_landmarks clamped to size=%d (was %d)", self.size, num_landmarks_v
            )

        self.skip = True
        self.landmark_index = self.landmarks()
        self.landmark_embed = self.embeddings[self.landmark_index]
        self.cross_kernel = self.kernel(self.embeddings, self.landmark_embed)
        self.landmark_kernel = self.kernel(self.landmark_embed, self.landmark_embed)
        reg_eps = max(1e-6, self.lam * 0.1)
        landmark_kernel_reg = self.landmark_kernel + reg_eps * torch.eye(self.num_landmarks, device=device, dtype=dtype)
        self.landmark_cholesky = torch.linalg.cholesky(landmark_kernel_reg)
        self.landmark_projection = torch.linalg.solve_triangular(
            self.landmark_cholesky.T,
            torch.linalg.solve_triangular(self.landmark_cholesky, self.cross_kernel.T, upper=False),
            upper=True,
        ).T
        self.shape = (self.size, self.size)

    def landmarks(self) -> torch.Tensor:
        if self.method == "greedy":
            return self._landmarks_greedy()
        if self.method == "leverage":
            return self._landmarks_leverage()
        raise ValueError(f"Unknown method={self.method}")

    def _landmarks_greedy(self) -> torch.Tensor:
        idx = torch.zeros(self.num_landmarks, dtype=torch.long, device=self.device)
        idx[0] = torch.randint(0, self.size, (1,), device=self.device)
        for i in range(1, self.num_landmarks):
            sel = self.embeddings[idx[:i]]
            dists = torch.cdist(self.embeddings, sel) ** 2
            idx[i] = dists.min(dim=1).values.argmax()
        return idx

    def _landmarks_leverage(self) -> torch.Tensor:
        pilot_size = min(self.pilot, self.size)
        if pilot_size == self.size:
            pilot_idx = torch.arange(self.size, device=self.device)
        else:
            pilot_idx = torch.randperm(self.size, device=self.device)[:pilot_size]
        pilot_emb = self.embeddings[pilot_idx]
        gram = pilot_emb @ pilot_emb.T
        k_pilot = exp_safe(gram, skip=self.skip)
        evals, evecs = torch.linalg.eigh(k_pilot)
        scaled = evals / (evals + self.lam)
        scores = (evecs**2) @ scaled
        scores = scores.clamp(min=0)
        probs = scores / scores.sum()
        sampled = torch.multinomial(probs, self.num_landmarks, replacement=False)
        return pilot_idx[sampled]

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gram = x @ y.T
        return exp_safe(gram, skip=self.skip)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.lam * x + _nystrom_matvec(self.cross_kernel, self.landmark_projection, x)

    def diag(self) -> torch.Tensor:
        diag_approx = torch.sum(self.landmark_projection * self.cross_kernel, dim=1)
        return self.lam + diag_approx

    def dense(self) -> torch.Tensor:
        approx = self.landmark_projection @ self.cross_kernel.T
        approx.diagonal().add_(self.lam)
        return approx

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if y is None:
            y = self.embeddings
        gram = x @ y.T
        return exp_safe(gram, skip=self.skip)


def _rff_matvec(phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    temp = phi.T @ x
    return phi @ temp


_rff_matvec = Backend.compile(_rff_matvec)


class Fourier:
    """Random Fourier Feature (RFF) approximation of the kernel."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        num: Optional[int] = None,
        sigma: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.sigma = float(sigma)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        r = num if num is not None else max(100, int(self.size ** 0.5 * 2))
        self.num = r

        gen = torch.Generator(device=device).manual_seed(42)
        self.freq = (
            torch.randn(self.dim, r, generator=gen, device=device, dtype=dtype) / sigma
        )
        self.phase = torch.rand(r, generator=gen, device=device, dtype=dtype) * 2.0 * torch.pi

        proj = self.embeddings @ self.freq
        phi = torch.cat(
            [torch.cos(proj + self.phase), torch.sin(proj + self.phase)], dim=1
        )
        self.phi = phi / (r**0.5)
        self.shape = (self.size, self.size)
        self.skip = True

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.lam * x + _rff_matvec(self.phi, x)

    def diag(self) -> torch.Tensor:
        sq = torch.sum(self.phi**2, dim=1)
        return self.lam + sq

    def dense(self) -> torch.Tensor:
        k = self.phi @ self.phi.T
        k.diagonal().add_(self.lam)
        return k

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if y is None:
            y = self.embeddings
        proj_x = x @ self.freq
        phi_x = torch.cat(
            [torch.cos(proj_x + self.phase), torch.sin(proj_x + self.phase)], dim=1
        )
        proj_y = y @ self.freq
        phi_y = torch.cat(
            [torch.cos(proj_y + self.phase), torch.sin(proj_y + self.phase)], dim=1
        )
        return (phi_x @ phi_y.T) / self.num


# ---------------------------------------------------------------------------
# Sparse k-NN approximation
# ---------------------------------------------------------------------------


class Neighbors:
    """Sparse k-NN approximation of the attention kernel."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        k: Optional[int] = None,
        chunk: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.chunk = chunk

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        num_neighbors_v = k if k is not None else min(50, self.size)
        self.num_neighbors = min(num_neighbors_v, self.size)

        self.build()
        self.mat = self._coo(self.size, self.size)
        self.shape = (self.size, self.size)
        self.skip = True

    def build(self) -> None:
        """Compute top-k Euclidean neighbours, symmetrise, store as COO."""
        size = self.size
        k = self.num_neighbors
        chunk_size = self.chunk or size

        rows, cols, vals = [], [], []
        for i_start in range(0, size, chunk_size):
            i_end = min(i_start + chunk_size, size)
            dists = torch.cdist(self.embeddings[i_start:i_end], self.embeddings)
            topk = torch.topk(dists, min(k, size), largest=False, dim=1)
            row_idx = (
                torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, min(k, size))
            )
            rows.append(row_idx.flatten())
            cols.append(topk.indices.flatten())
            gram_vals = torch.sum(
                self.embeddings[i_start:i_end].unsqueeze(1) * self.embeddings[topk.indices],
                dim=2,
            ).flatten()
            vals.append(exp_safe(gram_vals))

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)

        all_rows = torch.cat([rows, cols])
        all_cols = torch.cat([cols, rows])
        all_vals = torch.cat([vals, vals])

        sort_key = all_rows.to(torch.int64) * (size + 1) + all_cols.to(torch.int64)
        order = torch.argsort(sort_key)
        s_rows = all_rows[order]
        s_cols = all_cols[order]
        s_vals = all_vals[order]

        diff = torch.diff(s_rows.to(torch.int64) * (size + 1) + s_cols.to(torch.int64))
        is_new = torch.cat([torch.tensor([True], device=self.device), diff != 0])

        coo_indices = torch.stack([s_rows[is_new], s_cols[is_new]], dim=0)
        coo_values = s_vals[is_new]

        if k < size:
            diag_mask = coo_indices[0] == coo_indices[1]
            diag_rows = coo_indices[0, diag_mask]
            diag_vals = coo_values[diag_mask]

            off_mask = ~diag_mask
            off_rows = coo_indices[0, off_mask]
            off_vals = coo_values[off_mask]

            row_sums = torch.zeros(size, device=self.device, dtype=self.dtype)
            row_sums.index_add_(0, off_rows, off_vals.abs())

            diag_map = torch.full((size,), float("-inf"), device=self.device, dtype=self.dtype)
            diag_map[diag_rows] = diag_vals

            min_diag = row_sums * 1.01 + 1e-8
            new_diag = torch.maximum(diag_map, min_diag)

            coo_values = coo_values.clone()
            coo_values[diag_mask] = new_diag[diag_rows]

            missing = torch.where(~torch.isfinite(diag_map))[0]
            if missing.numel() > 0:
                missing_vals = min_diag[missing]
                coo_indices = torch.cat(
                    [coo_indices, torch.stack([missing, missing], dim=0)], dim=1
                )
                coo_values = torch.cat([coo_values, missing_vals])

        self.coo_indices = coo_indices
        self.coo_values = coo_values

    def _coo(self, m: int, n: int) -> torch.Tensor:
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                self.coo_indices,
                self.coo_values,
                (m, n),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lam * x
        if x.dim() == 1:
            out = out + torch.sparse.mm(self.mat, x.unsqueeze(1)).squeeze(1)
        else:
            out = out + torch.sparse.mm(self.mat, x)
        return out

    def diag(self) -> torch.Tensor:
        diag = torch.zeros(self.size, device=self.device, dtype=self.dtype)
        diag_mask = self.coo_indices[0] == self.coo_indices[1]
        diag_rows = self.coo_indices[0, diag_mask]
        diag_vals = self.coo_values[diag_mask]
        diag[diag_rows] = diag_vals
        sq_norms = torch.sum(self.embeddings**2, dim=1)
        exact_diag = torch.exp(sq_norms)
        diag = torch.maximum(diag, exact_diag)
        return self.lam + diag

    def dense(self) -> torch.Tensor:
        dense = self._coo(self.size, self.size).to_dense()
        dense.diagonal().add_(self.lam)
        return dense

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if y is None:
            y = self.embeddings
        m = x.shape[0]
        p = y.shape[0]
        k = min(self.num_neighbors, p)
        chunk_size = chunk or m

        rows, cols, vals = [], [], []
        for i_start in range(0, m, chunk_size):
            i_end = min(i_start + chunk_size, m)
            gram_chunk = x[i_start:i_end] @ y.T  # noqa
            topk = torch.topk(gram_chunk, k, largest=True, dim=1)
            row_idx = (
                torch.arange(i_start, i_end, device=self.device).unsqueeze(1).expand(-1, k)
            )
            rows.append(row_idx.flatten())
            cols.append(topk.indices.flatten())
            vals.append(exp_safe(topk.values).flatten())

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)

        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                torch.stack([rows, cols], dim=0),
                vals,
                (m, p),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()


# ---------------------------------------------------------------------------
# SKI approximation
# ---------------------------------------------------------------------------


def weights(
    x: torch.Tensor, grid: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multilinear interpolation weights for product-grid SKI.

    Args:
        x: ``(n, d)`` query points in ``[0, 1]``.
        grid: list of ``d`` sorted 1-D grid coordinate tensors.

    Returns:
        ``(indices, weights)`` of shapes ``(n, 2^d)`` each. Row sums of
        ``weights`` equal ``1``.
    """
    n, d = x.shape
    g_per_dim = [g.shape[0] for g in grid]
    vertices = 2**d

    low_idx = torch.zeros(n, d, dtype=torch.long, device=x.device)
    frac = torch.zeros(n, d, dtype=x.dtype, device=x.device)
    for dim, g in enumerate(grid):
        g = g.to(x.device, x.dtype)
        xc = x[:, dim].clamp(min=g[0], max=g[-1])
        diff = xc.unsqueeze(1) - g.unsqueeze(0)
        idx = (diff > 0).sum(dim=1) - 1
        idx = idx.clamp(min=0, max=g.shape[0] - 2)
        low = g[idx]
        high = g[idx + 1]
        denom = high - low
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        low_idx[:, dim] = idx
        frac[:, dim] = (xc - low) / denom

    vertex_offsets = torch.arange(vertices, device=x.device)
    bits = ((vertex_offsets.unsqueeze(1) >> torch.arange(d, device=x.device)) & 1).to(torch.bool)

    strides = [1]
    for g in reversed(g_per_dim[1:]):
        strides.append(strides[-1] * g)
    strides = list(reversed(strides))
    strides_t = torch.tensor(strides, dtype=torch.long, device=x.device)

    indices = torch.zeros(n, vertices, dtype=torch.long, device=x.device)
    w = torch.ones(n, vertices, dtype=x.dtype, device=x.device)
    for dim in range(d):
        dim_idx = low_idx[:, dim].unsqueeze(1) + bits[:, dim].unsqueeze(0).long()
        indices += dim_idx * strides_t[dim]
        dim_weight = torch.where(
            bits[:, dim].unsqueeze(0),
            frac[:, dim].unsqueeze(1),
            1.0 - frac[:, dim].unsqueeze(1),
        )
        w *= dim_weight

    return indices, w


class Grid:
    """SKI approximation via a product grid and multilinear interpolation."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        grid_size: Optional[int] = None,
        bounds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)

        d = self.dim
        if grid_size is None:
            grid_size = min(4096, max(64, 2**d))
        self.grid_size = grid_size
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")

        per_dim = max(2, int(grid_size ** (1.0 / d)))
        while per_dim ** d > grid_size and per_dim > 2:
            per_dim -= 1
        self.per_dim = per_dim
        actual = per_dim ** d
        if actual > grid_size:
            raise ValueError(
                f"Cannot build product grid: {per_dim}^{d}={actual} > {grid_size}. "
                "Use a larger grid_size or lower dim."
            )

        if self.dtype == torch.float32 and actual > 8192:
            logger.warning(
                "SKI grid has %d points; exact kernel evaluation on the grid may be slow.",
                actual,
                d,
            )

        self.build(bounds)
        self.shape = (self.size, self.size)
        self.skip = True

    def build(self, bounds: Optional[torch.Tensor]) -> None:
        d = self.dim
        per_dim = self.per_dim

        if bounds is None:
            mins = self.embeddings.min(dim=0).values
            maxs = self.embeddings.max(dim=0).values
            pad = (maxs - mins) * 0.05 + 1e-6
            mins = mins - pad
            maxs = maxs + pad
        else:
            mins = bounds[:, 0]
            maxs = bounds[:, 1]

        grid_1d = [
            torch.linspace(
                mins[i].item(),
                maxs[i].item(),
                per_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(d)
        ]
        self.grid_1d = grid_1d

        norm_embed = torch.zeros_like(self.embeddings)
        for i in range(d):
            denom = maxs[i] - mins[i]
            denom = denom if denom > 0 else 1.0
            norm_embed[:, i] = (self.embeddings[:, i] - mins[i]) / denom

        grid_1d_norm = [
            torch.linspace(0.0, 1.0, per_dim, device=self.device, dtype=self.dtype)
            for _ in range(d)
        ]
        indices, w = weights(norm_embed, grid_1d_norm)
        self.idx = indices
        self.w = w

        mesh = torch.meshgrid(*grid_1d, indexing="ij")
        grid_points = torch.stack([m.flatten() for m in mesh], dim=1).to(self.dtype)
        self.points = grid_points

        gram_grid = grid_points @ grid_points.T
        self.k_grid = exp_safe(gram_grid)

    def wx(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``W^T @ x`` via index_add."""
        g = self.points.shape[0]
        out = torch.zeros(g, *x.shape[1:], device=self.device, dtype=self.dtype)
        vertices = self.idx.shape[1]
        for v in range(vertices):
            idx = self.idx[:, v]
            w = self.w[:, v]
            if x.dim() == 1:
                out.index_add_(0, idx, w * x)
            else:
                out.index_add_(0, idx, w.unsqueeze(1) * x)
        return out

    def wu(self, u: torch.Tensor) -> torch.Tensor:
        """Compute ``W @ u`` via gathering."""
        gathered = u[self.idx]
        if u.dim() == 1:
            return (gathered * self.w).sum(dim=1)
        return (gathered * self.w.unsqueeze(-1)).sum(dim=1)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lam * x
        v = self.wx(x)
        u = self.k_grid @ v
        return out + self.wu(u)

    def diag(self) -> torch.Tensor:
        # (W K_grid W^T)_ii = sum_{j,k} W_ij W_ik K_grid_jk
        # Compute via W @ K_grid @ W^T column-wise.
        size = self.size
        g = self.points.shape[0]
        w_dense = torch.zeros(size, g, device=self.device, dtype=self.dtype)
        vertices = self.idx.shape[1]
        for v in range(vertices):
            w_dense[torch.arange(size), self.idx[:, v]] += self.w[:, v]
        diag_k = (w_dense @ self.k_grid * w_dense).sum(dim=1)
        return self.lam + diag_k

    def dense(self) -> torch.Tensor:
        size = self.size
        g = self.points.shape[0]
        w_dense = torch.zeros(size, g, device=self.device, dtype=self.dtype)
        vertices = self.idx.shape[1]
        for v in range(vertices):
            w_dense[torch.arange(size), self.idx[:, v]] += self.w[:, v]
        k = w_dense @ self.k_grid @ w_dense.T
        k.diagonal().add_(self.lam)
        return k

    def interp(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return interpolation indices and weights for arbitrary points."""
        d = self.dim
        per_dim = self.per_dim
        grid_1d_norm = [
            torch.linspace(0.0, 1.0, per_dim, device=self.device, dtype=self.dtype)
            for _ in range(d)
        ]
        mins = torch.stack([g[0] for g in self.grid_1d])
        maxs = torch.stack([g[-1] for g in self.grid_1d])
        denom = maxs - mins
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        norm_pts = (points - mins) / denom
        return weights(norm_pts, grid_1d_norm)

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if y is None:
            y = self.embeddings

        idx_x, w_x = self.interp(x)
        idx_y, w_y = self.interp(y)

        g = self.points.shape[0]
        v = torch.zeros(g, y.shape[0], device=self.device, dtype=self.dtype)
        vertices = idx_y.shape[1]
        for vi in range(vertices):
            idx = idx_y[:, vi]
            w = w_y[:, vi]
            v += self.k_grid[:, idx] * w.unsqueeze(0)

        gathered = v[idx_x]
        return (gathered * w_x.unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Two-scale kernel
# ---------------------------------------------------------------------------


class Hybrid:
    """Two-scale kernel: convex combination of Nyström (global) + Neighbors (local)."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        alpha: float = 0.5,
        num: Optional[int] = None,
        k: Optional[int] = None,
        chunk: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.chunk = chunk

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)
        self.shape = (self.size, self.size)

        self.global_op = Nystrom(
            embeddings=self.embeddings,
            lam=self.lam,
            num=num,
            chunk=chunk,
            device=device,
            dtype=dtype,
        )
        self.local_op = Neighbors(
            embeddings=self.embeddings,
            lam=self.lam,
            k=k,
            chunk=chunk,
            device=device,
            dtype=dtype,
        )

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.global_op.matvec(x) + (1.0 - self.alpha) * self.local_op.matvec(x)

    def diag(self) -> torch.Tensor:
        return (
            self.alpha * self.global_op.diag()
            + (1.0 - self.alpha) * self.local_op.diag()
        )

    def dense(self) -> torch.Tensor:
        return (
            self.alpha * self.global_op.dense()
            + (1.0 - self.alpha) * self.local_op.dense()
        )

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        return self.alpha * self.global_op.eval(x, y, chunk=chunk) + (
            1.0 - self.alpha
        ) * self.local_op.eval(x, y, chunk=chunk)


# ---------------------------------------------------------------------------
# Spectral-shaped kernel
# ---------------------------------------------------------------------------


class Shaper(nn.Module):
    """Learned monotone function applied to eigenvalues.

    Parameterised as a positive linear combination of shifted softplus
    functions plus a positive linear term. Monotonicity is enforced by
    construction (all coefficients are positive), so the shaped
    spectrum preserves the positive-semidefinite property of the kernel.
    """

    def __init__(self, knots: int = 5) -> None:
        super().__init__()
        self.knots = knots
        self.raw_weights = nn.Parameter(torch.full((knots,), -10.0))
        self.raw_slope = nn.Parameter(torch.tensor(-2.35))
        self.x: Optional[torch.Tensor] = None

    def set(
        self,
        lo: float,
        hi: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.x = torch.linspace(lo, hi, self.knots, device=device, dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.x is None:
            raise RuntimeError("Shaper knots not set; call set() first.")
        slope = torch.nn.functional.softplus(self.raw_slope)
        weights = torch.nn.functional.softplus(self.raw_weights)
        diffs = t.unsqueeze(-1) - self.x
        return slope * t + (weights * torch.nn.functional.softplus(diffs)).sum(dim=-1)


def _spectral_matvec(u: torch.Tensor, spectrum: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    coeffs = u.T @ x
    scaled = spectrum * coeffs if coeffs.dim() == 1 else spectrum.unsqueeze(-1) * coeffs
    return u @ scaled


_spectral_matvec = Backend.compile(_spectral_matvec)


class Spectrum:
    """Spectral-shaped kernel: ``K = U diag(exp(g(sigma_i^2))) U^T``.

    Built from the SVD of the embedding matrix; costs ``O(n d^2)`` to
    build and ``O(n d)`` per matvec.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        lam: float = 1e-2,
        knots: int = 5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        self.size = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.lam = float(lam)

        if device is None:
            device = embeddings.device
        if dtype is None:
            dtype = embeddings.dtype
        self.device = device
        self.dtype = dtype
        self.embeddings = embeddings.to(device=device, dtype=dtype)
        self.shape = (self.size, self.size)

        u, s, vh = torch.linalg.svd(self.embeddings, full_matrices=False)
        self.u = u
        self.sigma = s
        self.vh = vh
        sigma_sq = s**2

        self.shaper = Shaper(knots=knots)
        lo = float(sigma_sq.min().item())
        hi = float(sigma_sq.max().item())
        pad = max(1e-6, (hi - lo) * 0.1)
        self.shaper.set(lo - pad, hi + pad, device=device, dtype=dtype)
        self.shaper = self.shaper.to(device=device, dtype=dtype)

        with torch.no_grad():
            shaped = self.shaper(sigma_sq)
            cap = 80.0 if dtype == torch.float32 else 700.0
            self.spectrum = torch.exp(shaped.clamp(max=cap))

        self.sigma_inv = torch.where(s > 1e-12, s.reciprocal(), torch.zeros_like(s))

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.lam * x + _spectral_matvec(self.u, self.spectrum, x)

    def diag(self) -> torch.Tensor:
        diag_k = (self.u**2) @ self.spectrum
        return self.lam + diag_k

    def dense(self) -> torch.Tensor:
        k = self.u @ torch.diag(self.spectrum) @ self.u.T
        k.diagonal().add_(self.lam)
        return k

    def eval(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        chunk: Optional[int] = None,
    ) -> torch.Tensor:
        if y is None:
            y = self.embeddings
        cx = (x @ self.vh.T) * self.sigma_inv.unsqueeze(0)
        cy = (y @ self.vh.T) * self.sigma_inv.unsqueeze(0)
        return (cx * self.spectrum.unsqueeze(0)) @ cy.T


__all__ = [
    "Kernel",
    "Exact",
    "Nystrom",
    "Fourier",
    "Neighbors",
    "Grid",
    "Hybrid",
    "Spectrum",
    "Shaper",
    "exp_safe",
    "exact_matvec",
    "weights",
]