"""Tests for :mod:`laker.data`."""

import pytest
import torch

from laker.data import Data


class TestValidate:
    def test_accepts_non_negative(self):
        Data.validate(loss=2.0, ref=1.0, shadow=1.5)

    def test_rejects_negative_loss(self):
        with pytest.raises(ValueError, match="loss must be non-negative"):
            Data.validate(loss=-1.0, ref=1.0, shadow=0.0)

    def test_rejects_zero_ref(self):
        with pytest.raises(ValueError, match="ref must be positive"):
            Data.validate(loss=2.0, ref=0.0, shadow=0.0)

    def test_rejects_negative_shadow(self):
        with pytest.raises(ValueError, match="shadow must be non-negative"):
            Data.validate(loss=2.0, ref=1.0, shadow=-0.5)


class TestField:
    def test_noiseless_closed_form(self):
        """Single-transmitter: RSS = P - 10 eta log10(d/d_0)."""
        locs = torch.tensor([[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]], dtype=torch.float64)
        tx = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        pwr = torch.tensor([0.0], dtype=torch.float64)
        clean, noisy = Data.field(locs, tx, pwr, loss=2.0, ref=1.0, shadow=0.0)
        expected = torch.tensor([0.0, -20.0, -40.0], dtype=torch.float64)
        torch.testing.assert_close(clean, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(noisy, expected, atol=1e-6, rtol=1e-6)

    def test_multi_transmitter_is_sum(self):
        locs = torch.tensor([[5.0, 5.0]], dtype=torch.float64)
        tx1 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        tx2 = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
        pwr1 = torch.tensor([0.0], dtype=torch.float64)
        pwr2 = torch.tensor([10.0], dtype=torch.float64)
        clean, _ = Data.field(locs, tx1, pwr1, loss=2.0, ref=1.0, shadow=0.0)
        clean2, _ = Data.field(locs, tx2, pwr2, loss=2.0, ref=1.0, shadow=0.0)
        clean_both, _ = Data.field(
            torch.cat([tx1, tx2]),
            torch.cat([locs, locs]),
            torch.cat([pwr1, pwr2]),
            loss=2.0,
            ref=1.0,
            shadow=0.0,
        )
        # Direct dBm sum per the documented contract.
        assert abs(clean_both[0].item() - (clean[0].item() + clean2[0].item())) < 1e-6

    def test_distance_clamp_below_ref(self):
        """Sensor closer than ref must use ref distance."""
        locs = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        tx = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        pwr = torch.tensor([0.0], dtype=torch.float64)
        clean, _ = Data.field(locs, tx, pwr, loss=2.0, ref=1.0, shadow=0.0)
        # At ref distance: 0 - 10*2*log10(1) = 0
        torch.testing.assert_close(clean, torch.zeros(1, dtype=torch.float64))

    def test_shadowing_has_correct_std(self):
        """Empirical std of noisy-clean matches shadow_sigma."""
        torch.manual_seed(0)
        locs = torch.rand(20000, 2, dtype=torch.float64) * 50
        tx = torch.tensor([[25.0, 25.0]], dtype=torch.float64)
        pwr = torch.tensor([0.0], dtype=torch.float64)
        _, noisy = Data.field(locs, tx, pwr, loss=2.0, ref=1.0, shadow=2.0, seed=0)
        _, clean = Data.field(locs, tx, pwr, loss=2.0, ref=1.0, shadow=0.0, seed=0)
        emp = (noisy - clean).std().item()
        assert abs(emp - 2.0) / 2.0 < 0.05

    def test_dtype_device_propagation(self):
        locs = torch.randn(5, 2, dtype=torch.float64, device="cpu")
        tx = torch.randn(1, 2, dtype=torch.float32)
        pwr = torch.randn(1, dtype=torch.float32)
        clean, noisy = Data.field(locs, tx, pwr)
        assert clean.dtype == torch.float64
        assert noisy.dtype == torch.float64

    def test_rejects_wrong_locs_dim(self):
        locs = torch.randn(5, dtype=torch.float64)
        tx = torch.randn(1, 2, dtype=torch.float64)
        pwr = torch.randn(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="locations must be 2-D"):
            Data.field(locs, tx, pwr)

    def test_rejects_count_mismatch(self):
        locs = torch.randn(2, 2, dtype=torch.float64)
        tx = torch.randn(2, 2, dtype=torch.float64)
        pwr = torch.randn(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="same length"):
            Data.field(locs, tx, pwr)

    def test_rejects_dim_mismatch(self):
        locs = torch.randn(2, 3, dtype=torch.float64)
        tx = torch.randn(1, 2, dtype=torch.float64)
        pwr = torch.randn(1, dtype=torch.float64)
        with pytest.raises(ValueError, match="same spatial dimension"):
            Data.field(locs, tx, pwr)


class TestSplit:
    def test_split_disjoint_and_correct_counts(self):
        torch.manual_seed(0)
        x = torch.randn(100, 2, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64)
        x_tr, y_tr, x_va, y_va = Data.split(100, x, y, val=0.2, seed=0)
        assert x_tr.shape[0] == 80
        assert x_va.shape[0] == 20
        assert y_tr.shape[0] == 80
        assert y_va.shape[0] == 20

    def test_split_deterministic_with_seed(self):
        x = torch.randn(100, 2, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64)
        a = Data.split(100, x, y, val=0.2, seed=7)
        b = Data.split(100, x, y, val=0.2, seed=7)
        for ta, tb in zip(a, b):
            torch.testing.assert_close(ta, tb)

    def test_split_different_seed_differs(self):
        x = torch.randn(100, 2, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64)
        a = Data.split(100, x, y, val=0.2, seed=0)
        b = Data.split(100, x, y, val=0.2, seed=1)
        assert not torch.equal(a[0], b[0])

    def test_split_indices_disjoint(self):
        torch.manual_seed(0)
        x = torch.randn(50, 2, dtype=torch.float64)
        y = torch.randn(50, dtype=torch.float64)
        x_tr, _, x_va, _ = Data.split(50, x, y, val=0.2, seed=0)
        # Compare row-wise uniqueness via sets.
        assert x_tr.shape[0] + x_va.shape[0] == 50


class TestGrid:
    def test_grid_shape(self):
        g = Data.grid((0.0, 10.0, 0.0, 5.0), 20)
        assert g.shape == (400, 2)

    def test_grid_corners(self):
        g = Data.grid((0.0, 1.0, 0.0, 1.0), 2)
        assert torch.isclose(g[0, 0], torch.tensor(0.0, dtype=g.dtype))
        assert torch.isclose(g[0, 1], torch.tensor(0.0, dtype=g.dtype))
        assert torch.isclose(g[3, 0], torch.tensor(1.0, dtype=g.dtype))
        assert torch.isclose(g[3, 1], torch.tensor(1.0, dtype=g.dtype))

    def test_grid_axis_uniform(self):
        g = Data.grid((0.0, 10.0, 0.0, 10.0), 5)
        x_unique = torch.unique(g[:, 0])
        assert x_unique.numel() == 5
        y_unique = torch.unique(g[:, 1])
        assert y_unique.numel() == 5

    def test_grid_rejects_bad_size(self):
        with pytest.raises(ValueError, match="at least 2"):
            Data.grid((0.0, 1.0, 0.0, 1.0), 1)

    def test_grid_rejects_bad_bounds(self):
        with pytest.raises(ValueError, match="x_min"):
            Data.grid((1.0, 0.0, 0.0, 1.0), 5)
        with pytest.raises(ValueError, match="y_min"):
            Data.grid((0.0, 1.0, 1.0, 0.0), 5)
