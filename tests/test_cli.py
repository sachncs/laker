"""Tests for :mod:`laker.cli`."""

import os

import pytest
import torch

from laker import Laker
from laker.cli import CLI


class TestLoad:
    def test_load_npy(self, tmp_path):
        import numpy as np

        path = str(tmp_path / "x.npy")
        np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))
        t = CLI.load(path)
        assert t.shape == (2, 3)
        assert t.dtype == torch.float32

    def test_load_pt(self, tmp_path):
        path = str(tmp_path / "x.pt")
        torch.save(torch.randn(4), path)
        t = CLI.load(path)
        assert t.shape == (4,)

    def test_load_pth(self, tmp_path):
        path = str(tmp_path / "x.pth")
        torch.save(torch.randn(4), path)
        t = CLI.load(path)
        assert t.shape == (4,)

    def test_load_unsupported_raises(self, tmp_path):
        path = str(tmp_path / "x.txt")
        with open(path, "w") as f:
            f.write("hi")
        with pytest.raises(ValueError, match="Unsupported"):
            CLI.load(path)


class TestParser:
    def test_parser_has_subcommands(self):
        p = CLI.parser()
        choices = p._subparsers._group_actions[0].choices
        assert "fit" in choices
        assert "predict" in choices

    def test_version_flag(self):
        p = CLI.parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--version"])


class TestFitRun:
    def test_fit_end_to_end(self, tmp_path):
        import numpy as np

        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        m_path = str(tmp_path / "m.pt")
        np.save(x_path, np.random.RandomState(0).rand(20, 2).astype(np.float64) * 100)
        np.save(y_path, np.random.RandomState(0).randn(20).astype(np.float64))

        with pytest.raises(SystemExit) as exc:
            CLI.run(
                [
                    "fit",
                    "--locations",
                    x_path,
                    "--measurements",
                    y_path,
                    "--output",
                    m_path,
                    "--dtype",
                    "float64",
                    "--embed-dim",
                    "4",
                ]
            )
        assert exc.value.code == 0
        assert os.path.exists(m_path)
        m = Laker.load(m_path)
        assert m.dtype == torch.float64

    def test_predict_end_to_end(self, tmp_path):
        import numpy as np

        np.random.seed(0)
        x = np.random.rand(20, 2).astype(np.float64) * 100
        y = np.random.randn(20).astype(np.float64)
        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        m_path = str(tmp_path / "m.pt")
        out_path = str(tmp_path / "pred.pt")
        np.save(x_path, x)
        np.save(y_path, y)
        with pytest.raises(SystemExit) as exc:
            CLI.run(
                [
                    "fit",
                    "--locations",
                    x_path,
                    "--measurements",
                    y_path,
                    "--output",
                    m_path,
                    "--dtype",
                    "float64",
                ]
            )
        assert exc.value.code == 0
        with pytest.raises(SystemExit) as exc:
            CLI.run(["predict", "--model", m_path, "--locations", x_path, "--output", out_path])
        assert exc.value.code == 0
        assert os.path.exists(out_path)
        preds = torch.load(out_path, weights_only=True)
        assert preds.shape == (20,)


class TestExitCodes:
    def test_no_subcommand_exits_one(self):
        with pytest.raises(SystemExit) as exc:
            CLI.run([])
        assert exc.value.code == 1

    def test_unknown_kernelexits(self, tmp_path):
        import numpy as np

        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        m_path = str(tmp_path / "m.pt")
        np.save(x_path, np.random.rand(20, 2).astype(np.float64))
        np.save(y_path, np.random.randn(20).astype(np.float64))
        with pytest.raises(SystemExit):
            CLI.run(
                [
                    "fit",
                    "--locations",
                    x_path,
                    "--measurements",
                    y_path,
                    "--output",
                    m_path,
                    "--kernel",
                    "bogus",
                ]
            )


class TestLogging:
    def test_logging_sets_root(self):
        import logging

        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        try:
            root.handlers.clear()
            CLI.logging(True)
            assert logging.getLogger().level <= logging.DEBUG
        finally:
            root.handlers = old_handlers
            root.level = old_level
