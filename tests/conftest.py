"""Shared pytest fixtures."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype():
    return torch.float64


@pytest.fixture
def small_problem(device, dtype):
    torch.manual_seed(0)
    n, d = 30, 2
    x = torch.rand(n, d, device=device, dtype=dtype) * 100
    y = torch.sin(x[:, 0] / 50) + torch.cos(x[:, 1] / 50) + 0.01 * torch.randn(n, dtype=dtype)
    return x, y


@pytest.fixture
def poly_problem(device, dtype):
    torch.manual_seed(1)
    n, d = 40, 1
    x = torch.linspace(-1, 1, n, device=device, dtype=dtype).unsqueeze(-1)
    y = (x[:, 0] ** 3) + 0.05 * torch.randn(n, dtype=dtype)
    return x, y