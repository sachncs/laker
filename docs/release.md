# Release process

Version is bumped in `pyproject.toml`, the changelog is updated, a
`vX.Y.Z` tag is cut, and the PyPI publishing workflow publishes the
source and wheel distributions.

## Pre-release checklist

1. All tests pass: `pytest tests/ -x`.
2. Coverage above gates: `pytest tests/ --cov=laker --cov-branch
   --cov-fail-under=90 --cov-branch-fail-under=80`.
3. `mypy laker/ --strict` clean.
4. `python -m build && twine check dist/*` reports `PASSED`.
5. Wheel install smoke: `pip install --force-reinstall dist/*.whl`
   in a fresh venv followed by `python -c "from laker import Laker"`.

## Steps

```bash
# 1. Bump version and edit CHANGELOG
$EDITOR pyproject.toml CHANGELOG.md

# 2. Confirm tests + types
pytest tests/ -x
mypy laker/ --strict

# 3. Build artefacts
rm -rf dist/ build/
python -m build
twine check dist/*

# 4. Tag and push
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z

# 5. Publish (only after explicit approval)
twine upload dist/*
```
