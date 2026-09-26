#!/usr/bin/env python3
"""Compare two independent builds of the same commit (release reproducibility).

``pypi.yml`` builds the distribution twice from the same checkout, with the
same hash-locked toolchain and the same ``SOURCE_DATE_EPOCH`` (the commit
time), and runs this script on the two output directories. Publishing is
blocked unless the two builds agree:

* **Wheels must be byte-identical** (equal SHA-256). setuptools' wheel builder
  honours ``SOURCE_DATE_EPOCH`` for the zip timestamps, so with a fixed
  toolchain nothing else may differ -- measured 2026-09-25: two builds were
  bit-identical with the variable set and differed without it.
* **Sdists must be content-identical**: the same members, in the same order,
  of the same type, mode, size and bytes. Byte identity is not required
  because setuptools' sdist does not honour ``SOURCE_DATE_EPOCH``: the gzip
  header and member mtimes carry the build time, and the owner fields carry the
  build user (pypa/setuptools#2133, open). Those fields are ignored, and ONLY
  those.

Anything else -- a file present in only one build, an unknown artifact type, a
link or device member -- fails. The script also prints each artifact's SHA-256,
which is the digest the publish step uploads.

Standard library only, like the rest of the quality contract.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import zlib
from pathlib import Path

#: Member types an sdist may contain. Links and devices are refused outright:
#: a reproducibility check that followed or skipped them would not be checking.
_ALLOWED_TYPES = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sdist_members(path: Path) -> tuple[list[tuple[str, int, int, int, str]], list[str]]:
    """``(name, type, mode, size, content-sha256)`` per member, plus defects.

    A truncated or corrupt archive can fail midway through the stream as well
    as at open, so the whole read sits inside one handler.
    """
    members: list[tuple[str, int, int, int, str]] = []
    defects: list[str] = []
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.type not in _ALLOWED_TYPES:
                    defects.append(
                        f"{path.name}: member {member.name!r} is not a regular file or directory"
                    )
                    continue
                content = ""
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        defects.append(f"{path.name}: member {member.name!r} cannot be read")
                        continue
                    content = hashlib.sha256(extracted.read()).hexdigest()
                members.append(
                    (member.name, int(member.isdir()), member.mode, member.size, content)
                )
    except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
        return [], [f"{path.name}: not a readable .tar.gz ({exc})"]
    return members, defects


def compare(first: Path, second: Path) -> tuple[list[str], list[str]]:
    """``(errors, report)`` for two build output directories."""
    errors: list[str] = []
    report: list[str] = []
    names_a = sorted(p.name for p in first.iterdir() if p.is_file())
    names_b = sorted(p.name for p in second.iterdir() if p.is_file())
    if not names_a:
        errors.append(f"{first}: no artifacts")
    if names_a != names_b:
        errors.append(f"artifact sets differ: {names_a} vs {names_b}")
        return errors, report
    for name in names_a:
        a, b = first / name, second / name
        digest_a, digest_b = sha256(a), sha256(b)
        report.append(f"{digest_a}  {name}")
        if name.endswith(".whl"):
            if digest_a != digest_b:
                errors.append(f"{name}: wheel is not byte-reproducible ({digest_a} vs {digest_b})")
        elif name.endswith(".tar.gz"):
            members_a, defects_a = _sdist_members(a)
            members_b, defects_b = _sdist_members(b)
            errors.extend(defects_a + defects_b)
            if members_a != members_b:
                differing = sorted(
                    {m[0] for m in members_a} ^ {m[0] for m in members_b}
                    | {x[0] for x, y in zip(members_a, members_b) if x != y}
                )
                errors.append(
                    f"{name}: sdist content differs between builds "
                    f"(members: {differing[:10] or 'order'})"
                )
        else:
            errors.append(f"{name}: unknown artifact type; refusing to vouch for it")
    return errors, report


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: compare_dists.py BUILD_DIR_A BUILD_DIR_B", file=sys.stderr)
        return 2
    first, second = Path(argv[1]), Path(argv[2])
    for directory in (first, second):
        if not directory.is_dir():
            print(f"not a directory: {directory}", file=sys.stderr)
            return 2
    errors, report = compare(first, second)
    for line in report:
        print(line)
    if errors:
        print("Release reproducibility check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release reproducibility check passed: wheel byte-identical, sdist content-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
