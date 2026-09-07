<!--
Thanks for contributing to LAKER! Please fill in the sections
below. Items marked with *optional* can be deleted when empty.
-->

## Summary

<!-- One-paragraph summary of what this PR changes. -->

## Changes

<!-- Bulleted list of changes; reference the file(s) touched. -->

- `laker/...` — description
- `tests/...` — description
- `docs/...` — description

## Motivation

<!-- Why is this change needed? What problem does it solve?
Link any related issues with `Closes #N`. -->

## Public API impact

<!-- Check all that apply; delete the rest. -->

- [ ] Adds a new public symbol.
- [ ] Renames an existing public symbol.
- [ ] Removes a public symbol.
- [ ] Changes the signature of a public method or parameter.
- [ ] Changes default behaviour of `Laker`.

If any of the above are checked, this change requires a `[Migration]`
section in `docs/migration.md`.

## Migration

<!-- *Optional.* Required when public API impact above is checked.
Old usage → new usage, side-by-side. -->

## Testing

<!-- How have you tested these changes? Tick all that apply. -->

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New tests added for new functionality
- [ ] Precision tests added against a closed-form reference
- [ ] Behavioural tests added for non-trivial state changes
- [ ] Code style checks pass (`black`, `isort`, `flake8`, `mypy`)

## Test output

<!-- Paste the relevant lines of `pytest tests/` output here.
For large PRs this is optional, but please include the summary line. -->

```

tests/test_X.py::test_Y PASSED    [...]

```

## Documentation

- [ ] Updated `docs/` to reflect the change
- [ ] Updated `NAMING.md` if a public symbol was renamed
- [ ] Updated `README.md` quick-start if the entry point changed
- [ ] Added a new example under `examples/`

## Checklist

- [ ] I have read the [Contributing Guidelines](../blob/master/CONTRIBUTING.md)
- [ ] My code follows the [module-class convention](../blob/master/CONTRIBUTING.md#module-class-convention)
- [ ] I have performed a self-review of my code
- [ ] I have commented non-obvious decisions
- [ ] My changes generate no new warnings (`pytest -W error`)
- [ ] I have added tests that prove my fix is effective or that my feature works
