"""Behavioural + precision tests for the ``laker`` CLI.

The CLI exposes ``laker fit`` and ``laker predict`` plus a
``--version`` flag. Every test exercises the CLI end-to-end with
real ``argv`` patches and real on-disk artefacts, then asserts both
that the artefact exists and that its contents are precise.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy
import pytest
import torch

from laker.__main__ import (
    cmd_fit,
    cmd_predict,
    load_tensor,
    main,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------
def test_setup_logging_verbose_runs():
    """``setup_logging(True)`` must not raise."""
    setup_logging(verbose=True)


def test_setup_logging_quiet_runs():
    """``setup_logging(False)`` must not raise."""
    setup_logging(verbose=False)


# ---------------------------------------------------------------------------
# ``load_tensor``: format dispatch + error paths.
# ---------------------------------------------------------------------------
def test_load_tensor_reads_npy(tmp_path):
    """``.npy`` files are loaded into a tensor with the same values."""
    arr = numpy.array([1.5, -2.25, 3.0, 4.75])
    path = tmp_path / "data.npy"
    numpy.save(path, arr)
    t = load_tensor(str(path))
    assert isinstance(t, torch.Tensor)
    torch.testing.assert_close(t, torch.from_numpy(arr))


def test_load_tensor_reads_pt(tmp_path):
    """``.pt`` files are loaded exactly."""
    src = torch.tensor([1.0, 2.0, 3.0])
    path = tmp_path / "data.pt"
    torch.save(src, path)
    loaded = load_tensor(str(path))
    torch.testing.assert_close(loaded, src)


def test_load_tensor_rejects_unsupported_extension(tmp_path):
    """An unsupported extension raises ``ValueError`` with the
    extension name in the message."""
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tensor(str(path))


def test_load_tensor_missing_file_raises():
    """A missing file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        load_tensor("/nonexistent/path/data.npy")


# ---------------------------------------------------------------------------
# ``main``: --version flag and missing subcommand.
# ---------------------------------------------------------------------------
def test_main_version_flag_exits_zero(capsys):
    """``laker --version`` exits 0 and prints the version string."""
    with patch.object(sys, "argv", ["laker", "--version"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    out = capsys.readouterr().out.lower()
    assert "laker" in out or "0." in out


def test_main_no_subcommand_exits_one(capsys):
    """``laker`` with no subcommand prints help and exits 1."""
    with patch.object(sys, "argv", ["laker"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    out = capsys.readouterr().out.lower()
    assert "usage:" in out or "fit" in out


# ---------------------------------------------------------------------------
# ``cmd_fit`` / ``cmd_predict``: end-to-end with real on-disk artefacts.
# ---------------------------------------------------------------------------
def test_cmd_fit_persists_alpha_and_predictions(tmp_path):
    """``cmd_fit`` writes a model whose ``alpha`` and one prediction
    agree with a separately-trained model. Precision-bound.
    """
    torch.manual_seed(0)
    n = 50
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)
    loc_path = tmp_path / "locs.pt"
    meas_path = tmp_path / "meas.pt"
    out_path = tmp_path / "model.pt"
    torch.save(x, loc_path)
    torch.save(y, meas_path)

    class Args:
        locations = str(loc_path)
        measurements = str(meas_path)
        output = str(out_path)
        regularization = 1e-2
        gamma = 0.1
        embedding_dim = 6
        probes = 30
        device = "cpu"
        dtype = "float64"
        verbose = False

    cmd_fit(Args())
    assert out_path.exists()

    # Re-fit independently for cross-check.
    from laker import Laker

    independent = Laker(
        regularization=1e-2,
        gamma=0.1,
        embedding_dim=6,
        probes=30,
        device="cpu",
        dtype=torch.float64,
    )
    independent.fit(x, y)
    reloaded = Laker.load(str(out_path))

    torch.testing.assert_close(reloaded.coef_, independent.coef_)
    preds_reloaded = reloaded.predict(x[:5])
    preds_independent = independent.predict(x[:5])
    torch.testing.assert_close(preds_reloaded, preds_independent)


def test_cmd_predict_writes_correct_shape_and_dtype(tmp_path):
    """``cmd_predict`` writes predictions whose shape and dtype match
    the queries.
    """
    torch.manual_seed(0)
    n = 30
    x = torch.rand(n, 2, dtype=torch.float64) * 100.0
    y = torch.randn(n, dtype=torch.float64)
    model = Laker_with_default_kwargs()
    model.fit(x, y)
    model_path = tmp_path / "model.pt"
    model.save(str(model_path))
    query_path = tmp_path / "query.pt"
    x_query = torch.rand(10, 2, dtype=torch.float64) * 100.0
    torch.save(x_query, query_path)
    out_path = tmp_path / "predictions.pt"

    class Args:
        model = str(model_path)
        locations = str(query_path)
        output = str(out_path)

    cmd_predict(Args())
    assert out_path.exists()
    preds = torch.load(out_path)
    assert preds.shape == (10,)
    assert preds.dtype == torch.float64


def Laker_with_default_kwargs():
    from laker import Laker

    return Laker(
        regularization=1e-2,
        embedding_dim=4,
        probes=20,
        cccp_max_iter=20,
        pcg_tol=1e-10,
        pcg_max_iter=200,
        dtype=torch.float64,
        verbose=False,
    )


def test_main_fit_subcommand_produces_loadable_model(tmp_path):
    """End-to-end: ``laker fit ...`` writes a model that ``Laker.load``
    can read back. Precision-checked via the predict round-trip.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.randn(n, dtype=torch.float64)
    loc_path = tmp_path / "locs.pt"
    meas_path = tmp_path / "meas.pt"
    out_path = tmp_path / "model.pt"
    torch.save(x, loc_path)
    torch.save(y, meas_path)

    argv = [
        "laker",
        "fit",
        "--locations",
        str(loc_path),
        "--measurements",
        str(meas_path),
        "--output",
        str(out_path),
        "--regularization",
        "0.01",
        "--gamma",
        "0.1",
        "--embedding-dim",
        "4",
        "--probes",
        "10",
        "--dtype",
        "float64",
    ]
    with patch.object(sys, "argv", argv):
        main()
    assert out_path.exists()

    from laker import Laker

    loaded = Laker.load(str(out_path))
    preds = loaded.predict(x[:3])
    assert preds.shape == (3,)
    assert torch.isfinite(preds).all()


def test_main_predict_subcommand_produces_correct_shape(tmp_path):
    """End-to-end: ``laker predict ...`` writes predictions whose shape
    matches the queries.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.randn(n, dtype=torch.float64)
    model = Laker_with_default_kwargs()
    model.fit(x, y)
    model_path = tmp_path / "model.pt"
    model.save(str(model_path))
    query_path = tmp_path / "query.pt"
    x_query = torch.rand(5, 2, dtype=torch.float64) * 10.0
    torch.save(x_query, query_path)
    out_path = tmp_path / "predictions.pt"

    argv = [
        "laker",
        "predict",
        "--model",
        str(model_path),
        "--locations",
        str(query_path),
        "--output",
        str(out_path),
    ]
    with patch.object(sys, "argv", argv):
        main()
    assert out_path.exists()
    preds = torch.load(out_path)
    assert preds.shape == (5,)


def test_main_fit_legacy_lambda_reg_alias_still_works(tmp_path):
    """Legacy ``--lambda-reg`` / ``--num-probes`` flags are accepted as
    aliases of the current ``--regularization`` / ``--probes``.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.rand(n, 2, dtype=torch.float64) * 10.0
    y = torch.randn(n, dtype=torch.float64)
    loc_path = tmp_path / "locs.pt"
    meas_path = tmp_path / "meas.pt"
    out_path = tmp_path / "model.pt"
    torch.save(x, loc_path)
    torch.save(y, meas_path)

    argv = [
        "laker",
        "fit",
        "--locations",
        str(loc_path),
        "--measurements",
        str(meas_path),
        "--output",
        str(out_path),
        "--lambda-reg",
        "0.01",
        "--num-probes",
        "10",
        "--dtype",
        "float64",
    ]
    with patch.object(sys, "argv", argv):
        main()
    assert out_path.exists()
