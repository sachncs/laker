"""Tests for the abstract Executor base class."""

from laker.executor import Executor


class RecordingExecutor(Executor):
    """Minimal Executor that records all calls for verification."""

    def __init__(self):
        self.sections = []
        self.results = {}
        self.metrics = {}
        self.times = {}

    def section(self, title: str) -> None:
        self.sections.append(title)

    def log_result(self, key: str, value) -> None:
        self.results[key] = value

    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        self.metrics[name] = value

    def time_operation(self, name: str, operation) -> any:
        import time
        start = time.time()
        result = operation()
        elapsed = time.time() - start
        self.times[name] = elapsed
        return result


def test_executor_section():
    """Executor.section should record section titles."""
    ex = RecordingExecutor()
    ex.section("Test Section")
    assert "Test Section" in ex.sections


def test_executor_log_result():
    """Executor.log_result should record key-value pairs."""
    ex = RecordingExecutor()
    ex.log_result("answer", 42)
    assert ex.results["answer"] == 42


def test_executor_log_metric():
    """Executor.log_metric should record metrics."""
    ex = RecordingExecutor()
    ex.log_metric("accuracy", 0.95)
    assert ex.metrics["accuracy"] == 0.95


def test_executor_time_operation():
    """Executor.time_operation should time an operation."""
    ex = RecordingExecutor()

    def op():
        return 42

    result = ex.time_operation("test_op", op)
    assert result == 42
    assert "test_op" in ex.times
    assert ex.times["test_op"] >= 0
