"""Sphinx configuration for LiouScope documentation.

Build locally with::

    pip install -e .[docs]
    sphinx-build -b html docs docs/_build/html
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# We rely on the package being pip-installed (editable or wheel). The
# ``sys.path`` shim is kept only as a safety net for ad-hoc builds where
# the user invokes ``sphinx-build docs/`` without ``pip install -e .``;
# in CI and on ReadTheDocs the install always happens first.
ROOT = Path(__file__).resolve().parent.parent
_src = ROOT / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# -- Project information -----------------------------------------------------
project = "LiouScope"
author = "Coworker Research / Coworkerz"
copyright = "2026, Coworker Research"

try:
    release = importlib.metadata.version("liouscope")
except importlib.metadata.PackageNotFoundError:
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = os.environ.get("LIOUSCOPE_DOC_THEME", "alabaster")
html_title = f"LiouScope {release}"
html_static_path = ["_static"]
html_show_sourcelink = False

# -- Cross-references --------------------------------------------------------
default_role = "any"
nitpicky = False
