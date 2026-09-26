#!/usr/bin/env python3
"""Adversarial tests for the release reproducibility comparator.

Each negative case changes ONE property of an otherwise identical pair of
builds, next to a positive control that differs only in the fields the
comparator is documented to ignore (sdist mtimes, owner, gzip header).
"""

from __future__ import annotations

import gzip
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import compare_dists as comparator

WHEEL = "pkg-1.0-py3-none-any.whl"
SDIST = "pkg-1.0.tar.gz"


def _wheel(path: Path, payload: bytes = b"print('x')\n") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("pkg/__init__.py", date_time=(2026, 1, 1, 0, 0, 0))
        archive.writestr(info, payload)


def _sdist(
    path: Path,
    files: list[tuple[str, bytes, int]] | None = None,
    *,
    mtime: int = 1_700_000_000,
    owner: str = "root",
    gzip_mtime: int = 1_700_000_000,
    symlink: bool = False,
) -> None:
    if files is None:
        files = [
            ("pkg-1.0/PKG-INFO", b"Name: pkg\n", 0o644),
            ("pkg-1.0/pkg/__init__.py", b"x = 1\n", 0o644),
        ]
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        directory = tarfile.TarInfo("pkg-1.0")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.mtime = mtime
        directory.uname = owner
        archive.addfile(directory)
        for name, data, mode in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            info.mtime = mtime
            info.uname = owner
            archive.addfile(info, io.BytesIO(data))
        if symlink:
            link = tarfile.TarInfo("pkg-1.0/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)
    with (
        path.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=gzip_mtime) as gz,
    ):
        gz.write(raw.getvalue())


class CompareDistsTests(unittest.TestCase):
    def dirs(self) -> tuple[Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        a, b = Path(tmp.name) / "a", Path(tmp.name) / "b"
        a.mkdir()
        b.mkdir()
        return a, b

    def pair(self) -> tuple[Path, Path]:
        a, b = self.dirs()
        for directory in (a, b):
            _wheel(directory / WHEEL)
            _sdist(directory / SDIST)
        return a, b

    def assert_fails(self, a: Path, b: Path, fragment: str) -> None:
        errors, _ = comparator.compare(a, b)
        self.assertTrue(any(fragment in e for e in errors), f"expected {fragment!r} in {errors}")

    # --- positive controls -------------------------------------------------

    def test_identical_builds_pass(self) -> None:
        a, b = self.pair()
        errors, report = comparator.compare(a, b)
        self.assertEqual(errors, [])
        self.assertEqual(len(report), 2)

    def test_sdist_build_time_and_owner_are_ignored(self) -> None:
        """Exactly the fields setuptools' sdist cannot make deterministic."""
        a, b = self.pair()
        _sdist(b / SDIST, mtime=1_800_000_000, owner="runner", gzip_mtime=1_800_000_000)
        self.assertNotEqual(comparator.sha256(a / SDIST), comparator.sha256(b / SDIST))
        self.assertEqual(comparator.compare(a, b)[0], [])

    # --- wheels ------------------------------------------------------------

    def test_wheel_must_be_byte_identical(self) -> None:
        a, b = self.pair()
        _wheel(b / WHEEL, payload=b"print('y')\n")
        self.assert_fails(a, b, "wheel is not byte-reproducible")

    # --- sdists ------------------------------------------------------------

    def test_sdist_content_mode_order_and_members_are_compared(self) -> None:
        base = [
            ("pkg-1.0/PKG-INFO", b"Name: pkg\n", 0o644),
            ("pkg-1.0/pkg/__init__.py", b"x = 1\n", 0o644),
        ]
        variants = {
            "content": [base[0], ("pkg-1.0/pkg/__init__.py", b"x = 2\n", 0o644)],
            "mode": [base[0], ("pkg-1.0/pkg/__init__.py", b"x = 1\n", 0o755)],
            "order": [base[1], base[0]],
            "extra member": [*base, ("pkg-1.0/extra.txt", b"", 0o644)],
        }
        for label, files in variants.items():
            with self.subTest(label=label):
                a, b = self.pair()
                _sdist(b / SDIST, files=files)
                self.assert_fails(a, b, "sdist content differs")

    def test_links_in_an_sdist_are_refused(self) -> None:
        a, b = self.pair()
        _sdist(a / SDIST, symlink=True)
        _sdist(b / SDIST, symlink=True)
        self.assert_fails(a, b, "is not a regular file or directory")

    def test_unreadable_sdist_is_refused(self) -> None:
        a, b = self.pair()
        (a / SDIST).write_bytes(b"not gzip")
        (b / SDIST).write_bytes(b"not gzip")
        self.assert_fails(a, b, "not a readable .tar.gz")

    # --- artifact sets -----------------------------------------------------

    def test_artifact_sets_must_match(self) -> None:
        a, b = self.pair()
        (b / WHEEL).unlink()
        self.assert_fails(a, b, "artifact sets differ")

    def test_unknown_artifact_types_are_refused(self) -> None:
        a, b = self.pair()
        for directory in (a, b):
            (directory / "pkg-1.0.egg").write_bytes(b"same")
        self.assert_fails(a, b, "unknown artifact type")

    def test_empty_build_is_refused(self) -> None:
        a, b = self.dirs()
        self.assert_fails(a, b, "no artifacts")

    # --- command line ------------------------------------------------------

    def test_exit_codes(self) -> None:
        a, b = self.pair()
        self.assertEqual(comparator.main(["x", str(a), str(b)]), 0)
        _wheel(b / WHEEL, payload=b"changed\n")
        self.assertEqual(comparator.main(["x", str(a), str(b)]), 1)
        self.assertEqual(comparator.main(["x", str(a)]), 2)
        self.assertEqual(comparator.main(["x", str(a), str(a / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
