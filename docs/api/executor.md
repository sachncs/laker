# `laker.executor` — abstract workflow logging

`Executor` is an abstract base class defining the interface that
workflow scripts use to log results. The `laker.bench.BenchmarkExecutor`
in `benchmarks/executor.py` is the concrete implementation used by
the benchmark suite.

## Abstract methods

Subclasses must implement:

```python
class Executor(ABC):
    @abstractmethod
    def section(self, title: str) -> None:
        """Emit a section banner."""

    @abstractmethod
    def log_result(self, key: str, value: Any) -> None:
        """Log a named result."""

    @abstractmethod
    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None:
        """Log a named numeric metric."""

    @abstractmethod
    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run operation, time it, return its return value."""
```

The contract is intentionally minimal so concrete implementations
can target arbitrary backends (rich console output, JSON file logs,
markdown reports, etc.).

## Thread safety

The base class imposes no thread-safety requirements. Implementations
that wish to be called from multiple threads must provide their own
synchronisation.

## Concrete implementation

`benchmarks/executor.py::BenchmarkExecutor` implements the four
methods with:
- `section`: logs `===` banners
- `log_result`: logs `key: value` lines
- `log_metric`: logs `name: formatted_value` lines
- `time_operation`: measures wall-clock time with
  `time.perf_counter`, logs the elapsed time, and returns the
  operation's return value
- Plus `run`, `run_repeated`, `run_once` for statistical timing

## Subclassing

To instantiate, you must implement all four abstract methods:

```python
class JsonExecutor(Executor):
    def section(self, title):
        self._log["sections"].append(title)

    def log_result(self, key, value):
        self._log["results"][key] = value

    def log_metric(self, name, value, fmt=".4f"):
        self._log["metrics"][name] = format(value, fmt)

    def time_operation(self, name, operation):
        import time
        start = time.perf_counter()
        result = operation()
        self._log["times"][name] = time.perf_counter() - start
        return result
```

Attempting to instantiate a subclass that doesn't implement all four
abstract methods raises `TypeError`.
