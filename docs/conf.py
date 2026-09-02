"""Sphinx configuration for the LiouScope documentation site.

Skeleton introduced by issue #72 (Sphinx + furo + MyST on Read the Docs).
Only the curated pages under this directory participate in the build; the
pre-existing standalone Markdown docs (release audits, ADRs, CANON status)
are intentionally excluded via ``include_patterns`` so the Sphinx source tree
stays scoped and ``-W`` (warnings-as-errors) builds clean.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "LiouScope"
author = "Coworker Research"
copyright = "2026, Coworker Research"  # Sphinx-mandated name (shadows the builtin)

# Single source of truth: read the installed package version. Falls back to a
# sentinel so an uninstalled checkout still builds (RTD installs the package).
try:
    release = importlib.metadata.version("liouscope")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - build-time guard
    release = "0.0.0+unknown"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Scope the Sphinx source scan to the curated Diataxis pages only. Without this
# every pre-existing docs/*.md (RELEASE_AUDIT_*, CANON_STATUS, adr/**) would be
# picked up and warn "document isn't included in any toctree", breaking -W.
include_patterns = [
    "index.md",
    "tutorials/**",
    "how-to/**",
    "reference/**",
    "explanation/**",
    "_static/**",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]

# -- Autodoc / napoleon ------------------------------------------------------

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = True

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"LiouScope {release}"
html_static_path = ["_static"]
