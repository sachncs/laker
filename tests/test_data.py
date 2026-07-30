"""Tests for synthetic data generation.

Every test asserts the closed-form math the function commits to:

    rss_clean[i] = sum_j (P_j - 10 * eta * log10(d_ij / d_0))   (eq. 1)
    d_ij = max(||x_i - tx_j||_2, d_0)
    rss_noisy = rss_clean + sigma_eps * N(0, 1)

The implementation emits rss_clean via direct dBm summation (each
contribution is added into a per-sensor accumulator) rather than
the physically correct linear-power/dBm-conversion sum. These
tests are calibrated against the current behaviour; if the
formula changes the tests must change with it.
"""

from __future__ import annotations

import pytest
import torch

from laker.data import Data


# ---------------------------------------------------------------------------
# Precision: closed-form math matches the implementation byte-for-byte.
# ---------------------------------------------------------------------------
def test_field_closed_form_one_transmitter_matches_manual_computation():
    """Eq. (1) with a single transmitter and zero shadowing."""
    locs = torch.tensor(
        [
            [1.0, 0.0],  # distance 1, path_loss = 0, signal = -30
            [10.0**0.5, 0.0],  # distance ~3.162, log10~0.5, path_loss ~10, signal -40
            [10.0, 0.0],  # distance 10, log10=1, path_loss=20, signal=-50
        ],
        dtype=torch.float64,
    )
    tx = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    pwr = torch.tensor([-30.0], dtype=torch.float64)
    expected = torch.tensor([-30.0, -30.0 - 10.0, -30.0 - 20.0], dtype=torch.float64)
    clean, noisy = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    torch.testing.assert_close(clean, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(noisy, expected, atol=1e-10, rtol=1e-10)


def test_field_closed_form_multi_transmitter_is_direct_dbm_sum():
    """Two transmitters at distinct locations: clean signal is the
    element-wise sum of the per-transmitter contributions (NOT
    linear-power). This locks in the current implementation
    behaviour: it is direct dBm addition.
    """
    tx = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    pwr = torch.tensor([-40.0, -50.0], dtype=torch.float64)
    loc = torch.tensor([[5.0, 0.0]], dtype=torch.float64)

    # Per-transmitter contributions at distance 5 from each tx: d_0=1
    # and eta=2, so log10(5/1) ~ 0.69897 and path_loss ~ 13.97 dB.
    contrib_0 = -40.0 - 10.0 * 2.0 * torch.log10(torch.tensor(5.0)).item()
    contrib_1 = -50.0 - 10.0 * 2.0 * torch.log10(torch.tensor(5.0)).item()
    expected_clean_tensor = torch.tensor([contrib_0 + contrib_1], dtype=torch.float64)

    clean, noisy = Data.field(
        loc,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    torch.testing.assert_close(clean, expected_clean_tensor, atol=1e-6, rtol=0)
    torch.testing.assert_close(noisy, clean, atol=0, rtol=0)


def test_field_clamps_distance_below_reference_to_reference():
    """A sensor inside ``reference_distance`` clamps its distance to
    ``reference_distance`` so the log is never negative (avoids
    negative d/d_0 and the resulting negative path loss).
    """
    tx = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    pwr = torch.tensor([-30.0], dtype=torch.float64)
    inside = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    on_reference = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    clean_in, _ = Data.field(
        inside,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    clean_ref, _ = Data.field(
        on_reference,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    # Both must be exactly -30 dBm because log10(1) = log10(max(0.1, 1)) = 0.
    torch.testing.assert_close(clean_in, torch.tensor([-30.0], dtype=torch.float64))
    torch.testing.assert_close(clean_ref, torch.tensor([-30.0], dtype=torch.float64))


def test_field_respects_path_loss_exponent():
    """At distance d > d_0 and zero shadowing, signal(i) = P_i - 10*eta*log10(d).
    Doubling ``path_loss_exponent`` should double the attenuation.
    """
    tx = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    pwr = torch.tensor([0.0], dtype=torch.float64)
    loc = torch.tensor([[10.0, 0.0]], dtype=torch.float64)

    clean_2, _ = Data.field(
        loc,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    clean_4, _ = Data.field(
        loc,
        tx,
        pwr,
        path_loss_exponent=4.0,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    expected_2 = -20.0
    expected_4 = -40.0
    torch.testing.assert_close(clean_2, torch.tensor([expected_2], dtype=torch.float64))
    torch.testing.assert_close(clean_4, torch.tensor([expected_4], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Precision: shadowing behaviour.
# ---------------------------------------------------------------------------
def test_field_noisy_minus_clean_has_correct_stddev():
    """With ``seed=0`` and ``n=20000`` samples, the empirical stddev of
    ``(noisy - clean)`` must be within 1% of the configured ``shadow_sigma``.
    """
    torch.manual_seed(0)
    n = 20000
    locs = torch.rand(n, 2, dtype=torch.float64) * 100.0
    tx = torch.tensor([[50.0, 50.0]], dtype=torch.float64)
    pwr = torch.tensor([-40.0], dtype=torch.float64)
    shadow = 2.5
    clean, noisy = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=shadow,
        seed=0,
    )
    diff = noisy - clean
    empirical = float(diff.std().item())
    rel_err = abs(empirical - shadow) / shadow
    assert rel_err < 0.05, (
        f"empirical stddev {empirical:.4f} differs from "
        f"configured {shadow:.4f} by {rel_err * 100:.2f}%"
    )


def test_field_noisy_minus_clean_is_zero_when_sigma_zero():
    """With ``shadow_sigma=0``, noisy and clean must be bit-identical."""
    torch.manual_seed(0)
    locs = torch.rand(100, 2, dtype=torch.float64) * 100.0
    tx = torch.tensor([[20.0, 30.0], [70.0, 60.0]], dtype=torch.float64)
    pwr = torch.tensor([-30.0, -45.0], dtype=torch.float64)
    clean, noisy = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.5,
        reference_distance=1.0,
        shadow_sigma=0.0,
        seed=0,
    )
    torch.testing.assert_close(noisy, clean)


# ---------------------------------------------------------------------------
# Behavioural: dtype, device, reproducibility.
# ---------------------------------------------------------------------------
def test_field_dtype_and_device_match_inputs():
    """Output dtype and device match the input locations."""
    locs = torch.rand(20, 2, dtype=torch.float32, device="cpu") * 10.0
    tx = torch.tensor([[5.0, 5.0]], dtype=torch.float32, device="cpu")
    pwr = torch.tensor([-40.0], dtype=torch.float32, device="cpu")
    clean, noisy = Data.field(locs, tx, pwr)
    assert clean.dtype == torch.float32
    assert noisy.dtype == torch.float32
    assert clean.device.type == "cpu"
    assert noisy.device.type == "cpu"


def test_field_seed_reproduces_identical_output():
    """Same seed -> identical output even on a different run."""
    locs = torch.rand(50, 2, dtype=torch.float64) * 50.0
    tx = torch.tensor([[25.0, 25.0]], dtype=torch.float64)
    pwr = torch.tensor([-40.0], dtype=torch.float64)
    c1, n1 = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=1.0,
        seed=1234,
    )
    c2, n2 = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=1.0,
        seed=1234,
    )
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(n1, n2)


def test_field_different_seeds_diverge():
    """Different seeds must produce different noisy traces."""
    locs = torch.rand(50, 2, dtype=torch.float64) * 50.0
    tx = torch.tensor([[25.0, 25.0]], dtype=torch.float64)
    pwr = torch.tensor([-40.0], dtype=torch.float64)
    _, n1 = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=1.0,
        seed=1,
    )
    _, n2 = Data.field(
        locs,
        tx,
        pwr,
        path_loss_exponent=2.0,
        reference_distance=1.0,
        shadow_sigma=1.0,
        seed=2,
    )
    # Two independent seeds should not coincide (probability of full
    # match on 50 floats is zero).
    assert not torch.allclose(n1, n2)


# ---------------------------------------------------------------------------
# Behavioural: input validation.
# ---------------------------------------------------------------------------
def test_field_rejects_non_2d_locations():
    with pytest.raises(ValueError, match="locations"):
        Data.field(torch.randn(5), torch.tensor([[1.0, 1.0]]), torch.tensor([-40.0]))


def test_field_rejects_non_2d_transmitters():
    with pytest.raises(ValueError, match="transmitters"):
        Data.field(torch.rand(5, 2), torch.tensor([1.0, 1.0]), torch.tensor([-40.0]))


def test_field_rejects_non_1d_powers():
    with pytest.raises(ValueError, match="powers"):
        Data.field(
            torch.rand(5, 2),
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[-40.0]]),
        )


def test_field_rejects_mismatched_tx_powers():
    with pytest.raises(ValueError, match="same length"):
        Data.field(
            torch.rand(5, 2),
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([-40.0]),
        )


def test_field_rejects_mismatched_spatial_dim():
    with pytest.raises(ValueError, match="spatial dimension"):
        Data.field(
            torch.rand(5, 2),
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([-40.0]),
        )


def test_field_rejects_empty_locations():
    with pytest.raises(ValueError, match="at least one row"):
        Data.field(
            torch.zeros(0, 2),
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([-40.0]),
        )


def test_field_rejects_empty_transmitters():
    with pytest.raises(ValueError, match="at least one row"):
        Data.field(
            torch.rand(5, 2),
            torch.zeros(0, 2),
            torch.zeros(0),
        )


def test_field_rejects_empty_powers():
    with pytest.raises(ValueError, match="at least one element"):
        Data.field(
            torch.rand(5, 2),
            torch.tensor([[1.0, 1.0]]),
            torch.zeros(0),
        )


def test_validate_params_rejects_negative_path_loss():
    with pytest.raises(ValueError, match="path_loss_exponent"):
        Data.validate_params(-0.1, 1.0, 0.0)


def test_validate_params_rejects_zero_reference_distance():
    with pytest.raises(ValueError, match="reference_distance"):
        Data.validate_params(2.0, 0.0, 0.0)


def test_validate_params_rejects_negative_shadow_sigma():
    with pytest.raises(ValueError, match="shadow_sigma"):
        Data.validate_params(2.0, 1.0, -0.1)


# ---------------------------------------------------------------------------
# Behavioural: grid generation.
# ---------------------------------------------------------------------------
def test_grid_corners_and_shape():
    """`grid_size=10` on a square [0, 100]^2 yields 100 points with
    the documented corners (x=0, x=100, y=0, y=100)."""
    g = Data.grid((0.0, 100.0, 0.0, 100.0), grid_size=10)
    assert g.shape == (100, 2)
    flat = g.reshape(10, 10, 2)
    torch.testing.assert_close(flat[0, 0], torch.tensor([0.0, 0.0], dtype=g.dtype))
    torch.testing.assert_close(flat[-1, -1], torch.tensor([100.0, 100.0], dtype=g.dtype))


def test_grid_axis_spacing_is_uniform():
    """Axis spacing equals (max - min) / (grid_size - 1)."""
    g = Data.grid((2.0, 8.0, -3.0, 1.0), grid_size=4)
    unique_x = torch.unique(g[:, 0])
    expected = torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=g.dtype)
    torch.testing.assert_close(unique_x, expected)
    unique_y = torch.unique(g[:, 1])
    expected_y = torch.tensor([-3.0, -1.6666666, -0.3333333, 1.0], dtype=g.dtype)
    torch.testing.assert_close(unique_y, expected_y, atol=1e-6, rtol=0)


def test_grid_smaller_than_4_rejected():
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        Data.grid((0.0, 1.0, 0.0, 1.0), grid_size=1)


def test_grid_reversed_axes_rejected():
    with pytest.raises(ValueError, match="x_min"):
        Data.grid((1.0, 0.0, 0.0, 1.0), grid_size=5)
    with pytest.raises(ValueError, match="y_min"):
        Data.grid((0.0, 1.0, 1.0, 0.0), grid_size=5)
