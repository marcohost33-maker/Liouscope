"""Single source of truth for the LiouScope version string.

This module is the *only* place the version literal lives. ``pyproject.toml``
declares ``dynamic = ["version"]`` and reads ``__version__`` from here via
``[tool.setuptools.dynamic]``, so the build metadata, the importable
``liouscope.__version__`` and every run manifest's ``framework_version`` are
guaranteed to be one and the same string. Keep this a bare literal (no imports)
so setuptools can read it statically at build time.
"""

__version__ = "0.3.0"
