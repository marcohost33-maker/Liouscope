"""Machine enforcement of the ``docs/CANON_STATUS.md`` section 4 invariants.

Why this file exists
--------------------
``CANON_STATUS.md`` section 4 declares six identifiers that "must remain
synchronized or be explicitly version-bumped". As of 2026-09-02 **nothing read
that list**: no test and no CI step opened ``CITATION.cff``, ``codemeta.json``
or ``CHANGELOG.md``, and the release workflow verified exactly one of the six
(``_version.py`` against the tag and the built distribution).

A rule that is complete, correct and never executed is an intention, not a
control. This file turns the declaration into a gate.

The version fields deliberately LAG the development line: ``CITATION.cff`` and
``codemeta.json`` name the last cut release while ``_version.py`` carries the
next ``.devN`` line, and ``CITATION.cff`` says so in its own comments. The rule
enforced here is therefore conditional, which is what makes it useful:

* on a development version (``X.Y.Z.devN``) the citation metadata may lag, but
  the two citation files must still agree WITH EACH OTHER, and the lag must be
  backwards (a citation ahead of the source is drift in the dangerous
  direction);
* on a final release version the lag must be closed -- every file names the
  same string, and ``CHANGELOG.md`` carries a section for it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from liouscope._consts import (
    DIAGNOSTIC_SCHEMA_VERSION,
    TAXONOMY_VERSION,
)
from liouscope._version import __version__
from liouscope.io.manifest import MANIFEST_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
CANON = REPO_ROOT / "docs" / "CANON_STATUS.md"


def _release_parts(version: str) -> tuple[int, ...]:
    """Return the numeric ``X.Y.Z`` prefix of a PEP 440 version string."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    assert match is not None, f"unparsable version {version!r}"
    return tuple(int(g) for g in match.groups())


def _is_development(version: str) -> bool:
    return ".dev" in version or "rc" in version or version.endswith("a0")


def _citation_version() -> str:
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    # Deliberately a line-anchored scan rather than a YAML parse: CITATION.cff
    # carries a `version:` key inside `references:` too, and a naive top-level
    # parse would silently pick whichever one the loader saw last.
    for line in text.splitlines():
        match = re.match(r'^version:\s*"?([^"\s]+)"?\s*$', line)
        if match:
            return match.group(1)
    pytest.fail("CITATION.cff has no top-level `version:` key")


def _codemeta() -> dict[str, object]:
    return json.loads((REPO_ROOT / "codemeta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CANON_STATUS section 4: the identifiers must be the ones the code actually has
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier",
    [TAXONOMY_VERSION, DIAGNOSTIC_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION],
)
def test_canon_status_names_the_live_schema_identifiers(identifier: str) -> None:
    """CANON_STATUS must quote the values the package actually exports.

    Catches the drift direction that matters: a schema bump in ``_consts.py``
    that never reaches the canon document, so an auditor reading the document
    certifies a version the code does not produce.
    """
    text = CANON.read_text(encoding="utf-8")
    assert identifier in text, (
        f"{identifier!r} is live in the package but absent from "
        f"docs/CANON_STATUS.md -- section 4 declares it a no-drift invariant."
    )


# ---------------------------------------------------------------------------
# Citation metadata coherence
# ---------------------------------------------------------------------------


def test_citation_and_codemeta_agree_with_each_other() -> None:
    """The two citation surfaces must never disagree, release or dev.

    They describe the same released artifact for the same consumers; a
    divergence between them cannot be explained by the documented dev-line lag.
    """
    codemeta = _codemeta()
    assert _citation_version() == codemeta["version"] == codemeta["softwareVersion"], (
        f"citation metadata disagrees: CITATION.cff={_citation_version()!r}, "
        f"codemeta.version={codemeta['version']!r}, "
        f"codemeta.softwareVersion={codemeta['softwareVersion']!r}"
    )


def test_citation_never_runs_ahead_of_the_source_version() -> None:
    """The documented lag is backwards only.

    Citation metadata naming a version the source has not reached would make
    every downstream citation unverifiable against the tree it points at.
    """
    assert _release_parts(_citation_version()) <= _release_parts(__version__), (
        f"CITATION.cff names {_citation_version()!r}, which is AHEAD of "
        f"src/liouscope/_version.py ({__version__!r})."
    )


def test_release_versions_close_the_citation_lag() -> None:
    """On a final release every declared version surface must name it.

    This is the assertion the release workflow does not make: it binds
    ``_version.py`` to the tag and to the built wheel, but never to
    ``CITATION.cff`` or ``codemeta.json``. Cutting v0.6.0 while the citation
    files still say 0.5.0 would pass every existing gate.
    """
    if _is_development(__version__):
        pytest.skip(
            f"{__version__} is a development version; the documented citation "
            "lag is permitted here and is checked for direction elsewhere."
        )
    assert _citation_version() == __version__
    assert _codemeta()["version"] == __version__


def test_release_versions_have_a_changelog_section() -> None:
    """A release without a changelog entry is an unannotated release."""
    if _is_development(__version__):
        pytest.skip(f"{__version__} is a development version")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{__version__}]" in changelog, (
        f"CHANGELOG.md has no section for released version {__version__}"
    )


def test_unreleased_work_is_recorded_on_a_development_version() -> None:
    """A dev line must keep an ``[Unreleased]`` section to accumulate into."""
    if not _is_development(__version__):
        pytest.skip(f"{__version__} is a release version")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
