"""Tests for :mod:`laker.executor`."""

import pytest

from laker.executor import Executor


class Concrete(Executor):
    def section(self, title):
        self.sections.append(title)

    def log_result(self, key, value):
        self.results[key] = value

    def log_metric(self, name, value, fmt=".4f"):
        self.metrics[name] = value

    def time_operation(self, name, operation):
        return operation()

    def __init__(self):
        self.sections = []
        self.results = {}
        self.metrics = {}


class TestExecutorContract:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Executor()

    def test_subclass_with_all_methods(self):
        e = Concrete()
        e.section("Title")
        assert "Title" in e.sections

    def test_log_result(self):
        e = Concrete()
        e.log_result("answer", 42)
        assert e.results["answer"] == 42

    def test_log_metric(self):
        e = Concrete()
        e.log_metric("acc", 0.95)
        assert e.metrics["acc"] == 0.95

    def test_time_operation_returns_value(self):
        e = Concrete()
        out = e.time_operation("op", lambda: 42)
        assert out == 42

    def test_time_operation_propagates_exceptions(self):
        e = Concrete()

        def bad():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            e.time_operation("x", bad)


class TestExecutorSubclassing:
    def test_partial_subclass_raises(self):
        class Partial(Executor):
            def section(self, title):
                pass

        with pytest.raises(TypeError):
            Partial()

    def test_missing_log_result_raises(self):
        class Missing(Executor):
            def section(self, title):
                pass

            def log_metric(self, name, value, fmt=".4f"):
                pass

            def time_operation(self, name, operation):
                return operation()

        with pytest.raises(TypeError):
            Missing()