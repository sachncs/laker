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

# A small budget per module: large modules are allowed, but a single
# class with many static methods is the path, not many tiny classes.
MAX_LINES = 600
MAX_CLASSES = 1
MAX_STATICS = 20


def audit_module(path: Path) -> list[str]:
    """Return a list of audit failures for ``path``."""
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    line_count = text.count("\n") + 1
    if line_count > MAX_LINES:
        failures.append(
            f"{path.relative_to(ROOT)}: {line_count} lines > {MAX_LINES}"
        )
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        failures.append(f"{path.relative_to(ROOT)}: parse error {exc}")
        return failures

    public_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]
    if len(public_classes) > MAX_CLASSES:
        failures.append(
            f"{path.relative_to(ROOT)}: {len(public_classes)} public classes "
            f"({', '.join(public_classes)}) > {MAX_CLASSES}"
        )

    public_functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    if public_functions:
        failures.append(
            f"{path.relative_to(ROOT)}: top-level public functions "
            f"{public_functions}"
        )

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
                f"{path.relative_to(ROOT)}: class {cls} has {count} static "
                f"methods > {MAX_STATICS}"
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
