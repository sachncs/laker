"""Tests for input validation and edge cases."""

import pytest
import torch

from laker.kernels import AttentionKernelOperator
from laker.models import LAKERRegressor


def test_invalid_embedding_dim():
    """Test that zero embedding_dim raises ValueError."""
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        LAKERRegressor(embedding_dim=0)


def test_invalid_lambda_reg():
    """Test that non-positive lambda_reg raises ValueError."""
    with pytest.raises(ValueError, match="lambda_reg must be positive"):
        LAKERRegressor(lambda_reg=0.0)
    with pytest.raises(ValueError, match="lambda_reg must be positive"):
        LAKERRegressor(lambda_reg=-1.0)


def test_invalid_gamma():
    """Test that negative gamma raises ValueError."""
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        LAKERRegressor(gamma=-0.1)


def test_invalid_epsilon():
    """Test that non-positive epsilon raises ValueError."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        LAKERRegressor(epsilon=0.0)


def test_invalid_base_rho():
    """Test that base_rho outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="base_rho must be in \\[0, 1\\]"):
        LAKERRegressor(base_rho=-0.1)
    with pytest.raises(ValueError, match="base_rho must be in \\[0, 1\\]"):
        LAKERRegressor(base_rho=1.5)


def test_invalid_max_iter():
    """Test that zero max_iter values raise ValueError."""
    with pytest.raises(ValueError, match="cccp_max_iter must be positive"):
        LAKERRegressor(cccp_max_iter=0)
    with pytest.raises(ValueError, match="pcg_max_iter must be positive"):
        LAKERRegressor(pcg_max_iter=0)


def test_invalid_tol():
    """Test that zero tolerance values raise ValueError."""
    with pytest.raises(ValueError, match="cccp_tol must be positive"):
        LAKERRegressor(cccp_tol=0.0)
    with pytest.raises(ValueError, match="pcg_tol must be positive"):
        LAKERRegressor(pcg_tol=0.0)


def test_predict_before_fit():
    """Test that predict before fit raises RuntimeError."""
    model = LAKERRegressor(verbose=False)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.predict(torch.rand(5, 2))


def test_save_before_fit():
    """Test that save before fit raises RuntimeError."""
    model = LAKERRegressor(verbose=False)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        model.save("/tmp/test.pt")


def test_fit_mismatched_shapes():
    """Test that 2-D y input raises ValueError."""
    model = LAKERRegressor(verbose=False)
    x = torch.rand(10, 2)
    y = torch.rand(10, 2)  # wrong shape
    with pytest.raises(ValueError, match="y must be 1-D"):
        model.fit(x, y)


def test_repr_before_and_after_fit():
    """Test __repr__ works before and after fit."""
    model = LAKERRegressor(verbose=False)
    r1 = repr(model)
    assert "not fitted" in r1
    assert "LAKERRegressor" in r1
    x = torch.rand(10, 2)
    y = torch.rand(10)
    model.fit(x, y)
    r2 = repr(model)
    assert "fitted" in r2


def test_invalid_k_neighbors():
    """Test that non-positive k_neighbors raises ValueError."""
    with pytest.raises(ValueError, match="k_neighbors must be positive"):
        LAKERRegressor(k_neighbors=0)


def test_invalid_grid_size():
    """Test that grid_size < 2 raises ValueError."""
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        LAKERRegressor(grid_size=1)


def test_invalid_landmark_method():
    """Test invalid landmark_method raises ValueError."""
    with pytest.raises(ValueError, match="landmark_method must be"):
        LAKERRegressor(landmark_method="invalid")


def test_invalid_preconditioner():
    """Test invalid preconditioner raises ValueError."""
    with pytest.raises(ValueError, match="preconditioner must be"):
        LAKERRegressor(preconditioner="invalid")


def test_chunk_memory_budget_default():
    """Test default chunk budget is 64 MB."""
    from laker.backend import get_chunk_memory_budget
    assert get_chunk_memory_budget() == 64 * 1024 * 1024


def test_chunk_disabled_default():
    """Test chunk is not disabled by default."""
    from laker.backend import get_chunk_disabled
    assert not get_chunk_disabled()


def test_fit_empty_tensor():
    """Test that fitting with empty x raises ValueError."""
    model = LAKERRegressor(verbose=False)
    with pytest.raises(ValueError, match="x must have at least one row"):
        model.fit(torch.empty(0, 2), torch.empty(0))


def test_get_set_params():
    """Test that get_params and set_params work correctly."""
    model = LAKERRegressor(lambda_reg=0.5, verbose=False)
    params = model.get_params()
    assert params["lambda_reg"] == 0.5
    model.set_params(lambda_reg=0.1)
    assert model.lambda_reg == 0.1
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(invalid_param=1)


def test_kernel_operator_invalid_embeddings():
    """Test that 1-D embeddings raises ValueError."""
    with pytest.raises(ValueError, match="embeddings must be 2-D"):
        AttentionKernelOperator(torch.randn(10))


def test_kernel_operator_repr():
    """Test AttentionKernelOperator repr shows key info."""
    op = AttentionKernelOperator(torch.randn(10, 5), lambda_reg=0.01)
    r = repr(op)
    assert "n=10" in r
    assert "embedding_dim=5" in r
    assert "lambda_reg=0.01" in r
    assert "AttentionKernelOperator" in r


def test_kernel_operator_matvec_wrong_shape():
    """Test matvec with wrong-shaped input raises ValueError."""
    op = AttentionKernelOperator(torch.randn(10, 4))
    with pytest.raises(ValueError, match="x must be 1-D or 2-D"):
        op.matvec(torch.randn(10, 4, 2))


def test_fit_with_search():
    """fit_with_search should find reasonable hyperparameters."""
    x = torch.rand(40, 2)
    y = torch.randn(40)
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit_with_search(
        x, y, lambda_reg_grid=[0.01], gamma_grid=[0.1], num_probes_grid=[20]
    )
    assert model.alpha is not None


def test_fit_with_bo():
    """fit_with_bo should run Bayesian optimisation without error."""
    x = torch.rand(30, 2)
    y = torch.randn(30)
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit_with_bo(x, y, n_calls=3, n_initial_points=2)
    assert model.alpha is not None


def test_fit_learned_embeddings():
    """fit_learned_embeddings should run without error."""
    x = torch.rand(30, 2)
    y = torch.randn(30)
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit(x, y)
    model.fit_learned_embeddings(x, y, epochs=3, lr=0.01)
    assert model.alpha is not None


def test_fit_residual_corrector():
    """fit_residual_corrector should run without error."""
    x = torch.rand(30, 2)
    y = torch.randn(30)
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit(x, y)
    model.fit_residual_corrector(x, y, epochs=10, patience=5)
    assert model.predict(torch.rand(5, 2)).shape == (5,)


def test_generate_radio_field_wrong_shapes():
    """Test generate_radio_field validates input shapes."""
    from laker.data import generate_radio_field
    locs = torch.randn(10, 2)
    tx = torch.randn(3, 2)
    pwr = torch.randn(3)
    with pytest.raises(ValueError, match="locations must be 2-D"):
        generate_radio_field(torch.randn(10), tx, pwr)
    with pytest.raises(ValueError, match="transmitters must be 2-D"):
        generate_radio_field(locs, torch.randn(3), pwr)
    with pytest.raises(ValueError, match="powers must be 1-D"):
        generate_radio_field(locs, tx, torch.randn(3, 1))
    with pytest.raises(ValueError, match="transmitters and powers must have same length"):
        generate_radio_field(locs, torch.randn(4, 2), pwr)


def test_radio_field_generator_repr():
    """Test RadioFieldGenerator repr shows parameters."""
    from laker.data import RadioFieldGenerator
    gen = RadioFieldGenerator(path_loss_exponent=3.0, shadow_sigma=2.0)
    r = repr(gen)
    assert "path_loss_exponent=3.0" in r
    assert "shadow_sigma=2.0" in r
    assert "RadioFieldGenerator" in r


def test_matvec_wrong_size():
    """AttentionKernelOperator matvec should reject mismatched n."""
    op = AttentionKernelOperator(torch.randn(10, 5))
    with pytest.raises(ValueError, match="must have"):
        op.matvec(torch.randn(5))


def test_matvec_wrong_size_2d():
    """AttentionKernelOperator matvec 2-D should reject mismatched n."""
    op = AttentionKernelOperator(torch.randn(10, 5))
    with pytest.raises(ValueError, match="must have"):
        op.matvec(torch.randn(5, 3))


def test_generate_grid_small():
    """generate_grid should reject grid_size < 2."""
    from laker.data import generate_grid
    with pytest.raises(ValueError, match="grid_size must be at least 2"):
        generate_grid((0.0, 1.0, 0.0, 1.0), 1)


def test_generate_grid_reversed_x():
    """generate_grid should reject reversed x bounds."""
    from laker.data import generate_grid
    with pytest.raises(ValueError, match="x_min"):
        generate_grid((1.0, 0.0, 0.0, 1.0), 5)


def test_generate_grid_reversed_y():
    """generate_grid should reject reversed y bounds."""
    from laker.data import generate_grid
    with pytest.raises(ValueError, match="y_min"):
        generate_grid((0.0, 1.0, 1.0, 0.0), 5)


def test_generate_radio_field_empty_locations():
    """generate_radio_field should reject empty locations."""
    from laker.data import generate_radio_field
    with pytest.raises(ValueError, match="at least one"):
        generate_radio_field(torch.empty(0, 2), torch.randn(1, 2), torch.randn(1))


def test_generate_radio_field_empty_transmitters():
    """generate_radio_field should reject empty transmitters."""
    from laker.data import generate_radio_field
    with pytest.raises(ValueError, match="at least one"):
        generate_radio_field(torch.randn(5, 2), torch.empty(0, 2), torch.empty(0))


def test_predict_wrong_features():
    """predict should reject wrong feature dimension."""
    model = LAKERRegressor(embedding_dim=4, verbose=False)
    model.fit(torch.rand(20, 2), torch.randn(20))
    with pytest.raises(ValueError, match="features"):
        model.predict(torch.rand(5, 3))
