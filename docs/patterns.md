# LAKER Design Patterns

This document describes the architectural patterns used throughout the LAKER
project. All new code should follow these conventions.

---

## 1. Executor Pattern

### What it is

The **Executor Pattern** decouples workflow logic from output formatting.
Workflow classes (e.g. examples, benchmarks) accept an optional ``Executor``
instance in their constructor and delegate all logging, timing, and result
collection to it.

### Why we use it

- **Consistency**: Every workflow emits the same structured output format.
- **Testability**: Workflows can be run with a silent or mock executor in tests.
- **Reusability**: The same workflow can be run in a notebook, a benchmark
  suite, or a CI job without code changes.

### Base interface

The abstract base class lives in ``laker.executor.Executor``:

```python
from abc import ABC, abstractmethod
from typing import Any, Callable

class Executor(ABC):
    @abstractmethod
    def section(self, title: str) -> None: ...
    @abstractmethod
    def log_result(self, key: str, value: Any) -> None: ...
    @abstractmethod
    def log_metric(self, name: str, value: float, fmt: str = ".4f") -> None: ...
    @abstractmethod
    def time_operation(self, name: str, operation: Callable[[], Any]) -> Any: ...
```

### Concrete executors

- ``examples.executor.ExampleExecutor`` — structured logging for demonstrations.
- ``benchmarks.executor.BenchmarkExecutor`` — timing, warmup, and statistical
  aggregation for performance measurement.

### Writing a workflow that uses an executor

```python
from typing import Optional
from laker.executor import Executor

class MyWorkflow:
    def __init__(self, executor: Optional[Executor] = None):
        self.executor = executor if executor is not None else ExampleExecutor()

    def run(self) -> None:
        self.executor.section("My Workflow")
        result = self.executor.time_operation("heavy step", self.heavy_step)
        self.executor.log_metric("accuracy", result.accuracy)
```

Constructor rule: always accept ``executor: Optional[Executor] = None`` and
fall back to a sensible default.

---

## 2. Class + Convenience Wrapper

### Rule: every public workflow must have a class implementation

Standalone free functions are **not** acceptable as the primary API for
workflows, benchmarks, or examples. Helpers exist only as
``@staticmethod`` on the module's primary class; they are *not*
public module-level functions.

### Pattern

```python
class MyBenchmark:
    def __init__(self, executor: Optional[Executor] = None):
        self.executor = executor if executor is not None else BenchmarkExecutor()

    def run(self) -> dict:
        ...

    @classmethod
    def run_default(cls) -> dict:
        """Convenience entry point."""
        return cls().run()
```

### When to keep a free-function wrapper

Public free functions are forbidden. Inlined `@staticmethod` on the
primary class is the single canonical way to expose a helper. If you
find yourself reaching for a wrapper, inline the call site or add a
`@staticmethod` to the responsible class.

---

## 3. Naming Conventions

- **No semi-private naming**: leading underscores are not used for
  public functions, methods, classes, or variables. Implementation
  helpers are `private` (single underscore) inside a class, never as
  public module-level names. The `_*.py` underscore-module convention
  for entire files is also removed; modules are unprefixed.
- **Descriptive names**: avoid single-letter or cryptic abbreviations.
  ``chunk_size`` is preferred over ``cs``; ``preconditioner`` over ``pre``.
- **PEP 8**: all identifiers are ``snake_case``; classes are
  ``PascalCase``. Top-level classes follow ``<Module>`` naming
  (``Laker``, ``Data``, ``Plot``, ``Embed`` etc.); secondary classes
  follow the same PascalCase (``Nystrom``, ``PCG``, ``Adaptive``).
- **Single source of truth**: every renames lives in `NAMING.md`.
  Names change in lock-step with code; no deprecation layer.

---

## 4. Logging

All diagnostic output must use the ``logging`` module. ``print()`` is forbidden
in library code, examples, and benchmarks. Documentation code blocks should
likewise demonstrate ``logger.info(...)`` rather than ``print(...)``.
