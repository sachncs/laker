"""Sphinx configuration for LAKER documentation."""

import os
import sys

# Robust insertion: locate the repo root regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

project = "LAKER"
copyright = "2026, LAKER Contributors"
author = "LAKER Contributors"

# Single source of truth: read from installed package metadata when
# available; fall back to a documented local fallback.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("laker")
except Exception:
    release = "0.4.0+local"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
