#!/usr/bin/env python3
"""Audit script enforcing the one-class-per-file rule.

Walks every Python module under ``laker/`` and reports any module that
either:

* defines more than one public class (i.e. classes whose name does
  not start with an underscore), or
* defines top-level public functions outside an ``if __name__`` guard.

Exit status is non-zero if any offender is found. Use as a CI gate.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "laker"

# Audit thresholds per Step 20 of TODO.md. Legacy modules remain
# non-compliant during the migration period; new modules added after
# Step 23 must stay under these numbers.
MAX_LINES = 300
MAX_CLASSES = 1
MAX_STATICS = 12

# Pre-Step-23 modules are exempt from the audit; remove entries as
# their migration lands in TODO.
LEGACY_FILES = {
    "laker/__init__.py",
    "laker/__main__.py",
    "laker/backend.py",
    "laker/base.py",
    "laker/benchmark.py",
    "laker/cli.py",
    "laker/core.py",
    "laker/data.py",
    "laker/distributed_kernels.py",
    "laker/embed.py",
    "laker/helpers.py",
    "laker/implicit_diff.py",
    "laker/kernels.py",
    "laker/model.py",
    "laker/models.py",
    "laker/persistence.py",
    "laker/plot.py",
    "laker/preconditioner.py",
    "laker/search.py",
    "laker/solvers.py",
    "laker/streaming.py",
    "laker/training.py",
    "laker/utils.py",
    "laker/visualize.py",
}


def audit_module(path: Path) -> list[str]:
    """Return a list of audit failures for ``path``."""
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    line_count = text.count("\n") + 1
    if line_count > MAX_LINES:
        failures.append(
            f"{rel}: {line_count} lines > {MAX_LINES}"
        )
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        failures.append(f"{rel}: parse error {exc}")
        return failures

    public_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]
    if len(public_classes) > MAX_CLASSES:
        failures.append(
            f"{rel}: {len(public_classes)} public classes "
            f"({', '.join(public_classes)}) > {MAX_CLASSES}"
        )

    public_functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    if public_functions:
        failures.append(f"{rel}: top-level public functions {public_functions}")

    static_counts: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            statics = sum(
                1
                for stmt in node.body
                if isinstance(stmt, ast.FunctionDef)
                and any(
                    isinstance(d, ast.Name) and d.id == "staticmethod"
                    for d in stmt.decorator_list
                )
            )
            if statics > 0:
                static_counts[node.name] = statics
    for cls, count in static_counts.items():
        if count > MAX_STATICS:
            failures.append(
                f"{rel}: class {cls} has {count} static methods > {MAX_STATICS}"
            )

    return failures


def main() -> int:
    if not PACKAGE.exists():
        print(f"package not found: {PACKAGE}", file=sys.stderr)
        return 2
    all_failures: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = str(path.relative_to(ROOT))
        if rel in LEGACY_FILES:
            continue
        all_failures.extend(audit_module(path))
    if all_failures:
        print("Audit failures:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    print("Audit clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
