#!/usr/bin/env python3
"""Hash-pin contract for the release workflow (code-scanning alerts #14/#15).

The job that builds the distribution PyPI receives used to run
``pip install build twine check-wheel-contents`` and
``python -m pip install --upgrade pip``: version-free, hash-free installs, so the
toolchain that produced and inspected the published artifact was whatever PyPI
served at that minute. OpenSSF Scorecard's Pinned-Dependencies check flags
exactly that. This gate keeps the release path at the stronger state:

1. In every release workflow each ``pip install`` is either

   * hash-locked -- ``--require-hashes`` plus ``-r <lock>``, nothing else
     positional; or
   * a local wheel -- ``--no-deps`` and only literal ``dist/*.whl`` paths
     (no shell expansion), which installs the artifact the job itself just
     built and verified.

   Anything else, including ``--upgrade pip`` and ``uv pip install``, fails.
2. Every lock such an install names is a real hash lock: each entry is an exact
   ``name==version`` pin carrying at least one well-formed digest, and there
   are no nested ``-r``/``-c``/``-e`` lines that would pull in unhashed input.
3. The lock covers every top-level requirement of its ``.in`` source, each at
   a version that satisfies the source's specifier, and it was compiled for
   the interpreter the workflow runs (``python-version``), because
   environment markers are resolved for exactly one interpreter (which is also
   why the ``.in`` may not carry markers of its own).
4. The locked build backend satisfies ``[build-system] requires`` in
   ``pyproject.toml`` -- the workflow builds with ``--no-isolation``, so the
   locked setuptools IS the backend, and a lock that drifted below the
   project's floor would build with an unsupported one.
5. Its header regeneration command still generates hashes and does not carry
   the spurious ``--no-index`` that pip-tools 7.6.1 writes under click >= 8.3
   (see ``.github/requirements/release.in``).
6. The release workflow is parseable YAML as far as inline values go (see
   :func:`plain_scalar_defects`): it runs only on release, so nothing else
   would notice before then.

Workflows are read STRUCTURALLY (PR #170 review): a strict YAML subset reader
(:func:`load_workflow_yaml`) decodes every scalar the way the runner receives
it -- a folded ``>-`` block is one command, not two lines -- and the shell
text is tokenised with :mod:`shlex`. Whatever the reader or the tokeniser
cannot decode fails the gate: an unreadable workflow is an unverified one.

Standard library only, like the rest of the quality contract.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only on Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]

#: Workflows that build or publish the distribution. Listed rather than inferred,
#: so that renaming one cannot silently drop it from the gate (it must exist).
RELEASE_WORKFLOWS: tuple[str, ...] = ("pypi.yml",)
#: A workflow containing one of these publishes to PyPI and must be listed above.
PUBLISH_MARKERS: tuple[str, ...] = (
    "pypa/gh-action-pypi-publish",
    "twine upload",
    "uv publish",
    "poetry publish",
    "flit publish",
    "hatch publish",
    "pdm publish",
)

#: ``pip install`` options that consume the following token as their value, so
#: it is not mistaken for a package argument. An option missing from this set
#: makes its value look positional, which fails the contract -- the safe side.
_VALUE_OPTIONS = frozenset(
    {
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-e",
        "--editable",
        "-t",
        "--target",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
        "--root",
        "--prefix",
        "--src",
        "--upgrade-strategy",
        "--only-binary",
        "--no-binary",
        "--progress-bar",
        "--root-user-action",
        "--report",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-C",
        "--config-settings",
        "--global-option",
        "--trusted-host",
        "--log",
        "--cache-dir",
        "--proxy",
        "--retries",
        "--timeout",
        "--exists-action",
        "--cert",
        "--client-cert",
        "--keyring-provider",
        "--python",
        "--group",
    }
)
_REQUIREMENT_OPTIONS = frozenset({"-r", "--requirement"})
_EDITABLE_OPTIONS = frozenset({"-e", "--editable"})
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "|", "&", ";;", "(", ")"})
_DIGEST_LENGTHS = {"sha256": 64, "sha384": 96, "sha512": 128}

_PYTHON_RE = re.compile(r"^(?:.*/)?python(?:\d+(?:\.\d+)?)?$")
_PIP_RE = re.compile(r"^(?:.*/)?pip(?:\d+(?:\.\d+)?)?$")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: The local-wheel exception: a literal, unexpanded path to a wheel in the
#: directory the release job itself builds into (``python -m build --outdir
#: dist``). No ``$``/backtick expansion, no scheme, no other directory -- pip
#: also installs "local or remote source archives", so ``"$BASE/pkg.whl"`` with
#: ``BASE=https://host`` would be an unhashed remote install (PR #170 review).
_LOCAL_WHEEL_RE = re.compile(r"dist/[A-Za-z0-9_.+*-]+\.whl")
#: Publisher command -> the sub-command that uploads. Matched on shell TOKENS,
#: so spacing and line continuations cannot hide it (PR #170 review).
_PUBLISH_TOOLS = {
    "twine": "upload",
    "uv": "publish",
    "poetry": "publish",
    "flit": "publish",
    "hatch": "publish",
    "pdm": "publish",
}
_HEADER_PYTHON_RE = re.compile(
    r"^#\s*This file is autogenerated by pip-compile with Python (\d+\.\d+)\s*$"
)
_HEADER_COMMAND_RE = re.compile(r"^#\s+(pip-compile\s.*)$")
_NAME_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_RELEASE_RE = re.compile(r"^\d+(?:\.\d+)*$")
_SPEC_RE = re.compile(r"^\s*(~=|==|!=|<=|>=|<|>)\s*([^\s,]+)\s*$")


def normalise_name(name: str) -> str:
    """PEP 503 normalisation, so ``Check_Wheel.Contents`` names one project."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class PipInstall:
    """One ``pip install`` invocation found in a workflow's shell text."""

    line: int
    requirement_files: tuple[str, ...]
    positionals: tuple[str, ...]
    flags: frozenset[str]


@dataclass
class Lock:
    """What a parsed hash lock pins, and what it was compiled for."""

    pins: dict[str, str] = field(default_factory=dict)
    python: str | None = None
    command: str | None = None


# ---------------------------------------------------------------------------
# Workflow YAML: a strict subset reader (PR #170 review)
# ---------------------------------------------------------------------------
#
# The gate used to scan workflow TEXT line by line. That is not what the runner
# executes: YAML decides what a ``run:`` value is. A folded block (``>-``)
# joins its lines with spaces, so ``python -m pip`` / ``install build`` on two
# physical lines is ONE command that no single line shows. The quality
# contract stays dependency-free (a required check that installs nothing), so
# instead of PyYAML this reader implements the block subset GitHub workflows
# use -- block mappings and sequences, one-line plain / single- /
# double-quoted scalars, one-line flow collections of scalars, and ``|``,
# ``|-``, ``>``, ``>-`` block scalars with YAML's folding rules (mirroring
# PyYAML's ``scan_block_scalar``) -- and REFUSES everything else with its line:
# anchors, aliases, tags, multi-line plain or quoted scalars, explicit
# indentation and keep indicators, duplicate keys, tabs in indentation,
# directives and document markers. Refusal is the safe side: the caller turns
# it into a gate failure. The decoded values are checked against PyYAML's
# ``BaseLoader`` in the test suite wherever PyYAML is installed.


class WorkflowYamlError(ValueError):
    """A construct outside the strict workflow YAML subset, with its line."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class YamlScalar:
    """A decoded scalar and the physical line it starts on."""

    value: str
    line: int
    #: ``"|"`` or ``">"`` for a block scalar (content starts on ``line + 1``).
    block: str = ""


_YAML_KEY_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.-]*)[ ]*:(?=[ ]|$)")
_PLAIN_FORBIDDEN_FIRST = frozenset(",[]{}#&*!|>'\"%@`")
_FLOW_PLAIN_RE = re.compile(r"[A-Za-z0-9_.$/+~=^(][A-Za-z0-9_.$/+~=^()* -]*")
_DQ_ESCAPES = {
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "t": "\t",
    "\t": "\t",
    "n": "\n",
    "v": "\v",
    "f": "\f",
    "r": "\r",
    "e": "\x1b",
    " ": " ",
    '"': '"',
    "/": "/",
    "\\": "\\",
    "N": "\x85",
    "_": "\xa0",
    "L": " ",
    "P": " ",
}
_DQ_HEX = {"x": 2, "u": 4, "U": 8}


def _quoted(text: str, start: int, index: int) -> tuple[str, int]:
    """Decode a one-line quoted scalar at ``text[start]``; ``(value, end)``."""
    quote = text[start]
    out: list[str] = []
    k = start + 1
    while k < len(text):
        char = text[k]
        if quote == "'":
            if char == "'":
                if text[k + 1 : k + 2] == "'":
                    out.append("'")
                    k += 2
                    continue
                return "".join(out), k + 1
            out.append(char)
            k += 1
            continue
        if char == '"':
            return "".join(out), k + 1
        if char == "\\":
            code = text[k + 1 : k + 2]
            if code in _DQ_ESCAPES:
                out.append(_DQ_ESCAPES[code])
                k += 2
                continue
            if code in _DQ_HEX:
                width = _DQ_HEX[code]
                digits = text[k + 2 : k + 2 + width]
                if len(digits) != width or not re.fullmatch(r"[0-9A-Fa-f]+", digits):
                    raise WorkflowYamlError(index + 1, "malformed escape in a double-quoted scalar")
                point = int(digits, 16)
                if point > 0x10FFFF or 0xD800 <= point <= 0xDFFF:
                    raise WorkflowYamlError(index + 1, "escape outside the Unicode scalar range")
                out.append(chr(point))
                k += 2 + width
                continue
            raise WorkflowYamlError(index + 1, f"escape {code!r} is outside the subset")
        out.append(char)
        k += 1
    raise WorkflowYamlError(
        index + 1,
        "quoted scalar does not close on its line (multi-line quoted scalars are outside the subset)",
    )


def _expect_end(text: str, end: int, index: int) -> None:
    """Only blanks or a comment may follow a complete scalar on its line."""
    rest = text[end:]
    if rest.strip(" \t") and not re.match(r"[ \t]+#", rest):
        raise WorkflowYamlError(index + 1, f"unexpected text {rest.strip()!r} after a scalar")


def _strip_plain_comment(text: str) -> str:
    """A plain scalar ends at `` #``: YAML plain scalars know no quotes."""
    cut = re.search(r"[ \t]#", text)
    return (text[: cut.start()] if cut else text).rstrip(" \t")


def _plain(text: str, index: int) -> str:
    first = text[0]
    if first in _PLAIN_FORBIDDEN_FIRST or (
        first in "-?:" and (len(text) == 1 or text[1] in " \t")
    ):
        raise WorkflowYamlError(index + 1, f"plain scalar {text!r} starts with an indicator")
    if ": " in text or ":\t" in text or text.endswith(":"):
        raise WorkflowYamlError(index + 1, f"plain scalar {text!r} contains a mapping indicator")
    return text


def _flow(text: str, index: int) -> Any:
    """A one-line flow sequence or mapping of scalars (no nesting)."""
    closer = "]" if text[0] == "[" else "}"
    is_mapping = closer == "}"
    items: list[YamlScalar] = []
    pairs: dict[str, YamlScalar] = {}

    def skip(k: int) -> int:
        while k < len(text) and text[k] in " \t":
            k += 1
        return k

    def scalar(k: int) -> tuple[str, int]:
        if k >= len(text):
            raise WorkflowYamlError(index + 1, "unterminated flow collection")
        char = text[k]
        if char in "'\"":
            return _quoted(text, k, index)
        # Unquoted flow entries are restricted to plain words (the subset's own
        # rule, stricter than YAML): no indicator characters, no quotes, no ':'
        # inside -- `[:"main"]` decoded differently from PyYAML before this.
        match = _FLOW_PLAIN_RE.match(text, k)
        if match is None:
            raise WorkflowYamlError(
                index + 1, f"unquoted flow entry starting with {char!r} is outside the subset"
            )
        return match.group(0).rstrip(" "), match.end()

    k = skip(1)
    if text[k : k + 1] == closer:
        k += 1
    else:
        while True:
            first, k = scalar(k)
            k = skip(k)
            if is_mapping:
                if text[k : k + 1] != ":":
                    raise WorkflowYamlError(index + 1, "flow mapping entry without ':'")
                k = skip(k + 1)
                second, k = scalar(k)
                if first in pairs:
                    raise WorkflowYamlError(index + 1, f"duplicate key {first!r}")
                pairs[first] = YamlScalar(second, index + 1)
                k = skip(k)
            else:
                if text[k : k + 1] == ":":
                    raise WorkflowYamlError(
                        index + 1, "implicit mapping inside a flow sequence is outside the subset"
                    )
                items.append(YamlScalar(first, index + 1))
            if text[k : k + 1] == ",":
                k = skip(k + 1)
                if text[k : k + 1] == closer:
                    k += 1
                    break
                continue
            if text[k : k + 1] == closer:
                k += 1
                break
            raise WorkflowYamlError(
                index + 1, "malformed or multi-line flow collection (outside the subset)"
            )
    _expect_end(text, k, index)
    return pairs if is_mapping else items


class _WorkflowYamlReader:
    def __init__(self, text: str) -> None:
        if text.startswith("﻿"):
            raise WorkflowYamlError(1, "byte-order mark")
        text = text.replace("\r\n", "\n")
        for bad, name in (
            ("\r", "carriage return"),
            ("\x85", "NEL"),
            (" ", "line separator"),
            (" ", "paragraph separator"),
        ):
            if bad in text:
                line = text[: text.index(bad)].count("\n") + 1
                raise WorkflowYamlError(line, f"{name} character (YAML 1.1 reads it as a line break)")
        self.lines = text.split("\n")
        self.pos = 0

    def _indent(self, index: int) -> int:
        raw = self.lines[index]
        width = len(raw) - len(raw.lstrip(" "))
        if raw[width : width + 1] == "\t":
            raise WorkflowYamlError(index + 1, "tab in indentation")
        return width

    def _next(self) -> int | None:
        """Advance past blank and comment-only lines; the next significant one."""
        while self.pos < len(self.lines):
            stripped = self.lines[self.pos].strip(" ")
            if stripped and not stripped.startswith("#"):
                # PyYAML rejects a tab in several structural positions (inside
                # a plain scalar, after a flow entry); outside block-scalar
                # content the subset allows none (measured by the differential
                # fuzz against PyYAML 6.0.3).
                if "\t" in self.lines[self.pos]:
                    raise WorkflowYamlError(self.pos + 1, "tab outside a block scalar")
                return self.pos
            self.pos += 1
        return None

    @staticmethod
    def _is_entry(content: str) -> bool:
        return content == "-" or content.startswith("- ")

    def _key(self, content: str, index: int) -> tuple[str, str] | None:
        if content[:1] in ("'", '"'):
            try:
                key, end = _quoted(content, 0, index)
            except WorkflowYamlError:
                return None
            rest = content[end:].lstrip(" ")
            if rest.startswith(":") and (len(rest) == 1 or rest[1] == " "):
                return key, rest[1:]
            return None
        match = _YAML_KEY_RE.match(content)
        if match is None:
            return None
        return match.group(1), content[match.end() :]

    def load(self) -> dict[str, Any]:
        index = self._next()
        if index is None:
            raise WorkflowYamlError(1, "empty document")
        if self._indent(index) != 0:
            raise WorkflowYamlError(index + 1, "the document does not start at column 0")
        node = self._node(0)
        index = self._next()
        if index is not None:
            raise WorkflowYamlError(index + 1, "content after the top-level mapping")
        if not isinstance(node, dict):
            raise WorkflowYamlError(1, "the top level is not a mapping")
        return node

    def _node(self, indent: int) -> Any:
        index = self.pos
        content = self.lines[index][indent:]
        if self._is_entry(content):
            return self._sequence(indent)
        if self._key(content, index) is not None:
            return self._mapping(indent)
        raise WorkflowYamlError(
            index + 1,
            "expected 'key: value' or '- item' (bare and multi-line scalars are outside the subset)",
        )

    def _mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while True:
            index = self._next()
            if index is None:
                return result
            width = self._indent(index)
            if width < indent:
                return result
            if width > indent:
                raise WorkflowYamlError(index + 1, "unexpected indentation")
            content = self.lines[index][indent:]
            if self._is_entry(content):
                raise WorkflowYamlError(index + 1, "a sequence entry where a mapping key was expected")
            entry = self._key(content, index)
            if entry is None:
                raise WorkflowYamlError(index + 1, "expected 'key: value'")
            key, rest = entry
            if key in result:
                raise WorkflowYamlError(index + 1, f"duplicate key {key!r}")
            self.pos = index + 1
            result[key] = self._value(rest, index, indent, indentless=True)

    def _sequence(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while True:
            index = self._next()
            if index is None:
                return items
            width = self._indent(index)
            if width < indent:
                return items
            if width > indent:
                raise WorkflowYamlError(index + 1, "unexpected indentation")
            content = self.lines[index][indent:]
            if not self._is_entry(content):
                return items
            after = content[1:]
            text = after.lstrip(" ")
            column = indent + 1 + len(after) - len(text)
            if text and not text.startswith("#") and (
                self._is_entry(text) or self._key(text, index) is not None
            ):
                # A compact nested node ("- key: value", "- - x"): blank out
                # the dash, and the item is an ordinary block node at its column.
                self.lines[index] = " " * column + text
                self.pos = index
                items.append(self._node(column))
                continue
            self.pos = index + 1
            items.append(self._value(after, index, indent, indentless=False))

    def _value(self, rest: str, index: int, indent: int, *, indentless: bool) -> Any:
        text = rest.strip(" ")
        if text == "" or text.startswith("#"):
            following = self._next()
            if following is not None:
                width = self._indent(following)
                if width > indent:
                    return self._node(width)
                if (
                    indentless
                    and width == indent
                    and self._is_entry(self.lines[following][indent:])
                ):
                    return self._sequence(indent)
            return YamlScalar("", index + 1)
        if text[0] in "|>":
            return self._block_scalar(_strip_plain_comment(text), index, indent)
        node: Any
        if text[0] in "'\"":
            value, end = _quoted(text, 0, index)
            _expect_end(text, end, index)
            node = YamlScalar(value, index + 1)
        elif text[0] in "[{":
            node = _flow(text, index)
        else:
            node = YamlScalar(_plain(_strip_plain_comment(text), index), index + 1)
        following = self._next()
        if following is not None and self._indent(following) > indent:
            raise WorkflowYamlError(
                following + 1, "continuation of a multi-line scalar (outside the subset)"
            )
        return node

    def _block_scalar(self, header: str, index: int, indent: int) -> YamlScalar:
        match = re.fullmatch(r"([|>])(-?)", header)
        if match is None:
            raise WorkflowYamlError(
                index + 1,
                f"block scalar header {header!r} is outside the subset (only |, |-, > and >-)",
            )
        style, strip = match.group(1), match.group(2) == "-"
        body: list[str] = []
        content: int | None = None
        cursor = index + 1
        while cursor < len(self.lines):
            raw = self.lines[cursor]
            if raw.strip(" ") == "":
                body.append(raw)
                cursor += 1
                continue
            width = len(raw) - len(raw.lstrip(" "))
            if content is None:
                if width <= indent:
                    break
                content = width
            if width < content:
                break
            body.append(raw)
            cursor += 1
        self.pos = cursor
        if content is None:
            return YamlScalar("", index + 1, style)
        for offset, raw in enumerate(body):
            if raw.strip(" ") == "" and len(raw) > content:
                raise WorkflowYamlError(
                    index + 2 + offset, "whitespace-only line deeper than the block indentation"
                )
        lines = [raw[content:] if raw.strip(" ") else "" for raw in body]
        while lines and lines[-1] == "":
            lines.pop()
        # Clip keeps the line break AFTER the last content line -- which does
        # not exist when that line ends the file (measured against PyYAML).
        last_physical = index + len(lines)
        final_break = "\n" if last_physical < len(self.lines) - 1 else ""
        # PyYAML's scan_block_scalar, line by line.
        chunks: list[str] = []
        breaks: list[str] = []
        k = 0
        while k < len(lines) and lines[k] == "":
            breaks.append("\n")
            k += 1
        while k < len(lines):
            chunks.extend(breaks)
            line = lines[k]
            leading_non_space = not line.startswith((" ", "\t"))
            chunks.append(line)
            k += 1
            breaks = []
            while k < len(lines) and lines[k] == "":
                breaks.append("\n")
                k += 1
            if k < len(lines):
                if style == ">" and leading_non_space and not lines[k].startswith((" ", "\t")):
                    if not breaks:
                        chunks.append(" ")
                else:
                    chunks.append("\n")
        if not strip:
            chunks.append(final_break)
        return YamlScalar("".join(chunks), index + 1, style)


def load_workflow_yaml(text: str) -> dict[str, Any]:
    """Parse a workflow in the strict subset; :class:`WorkflowYamlError` outside it."""
    return _WorkflowYamlReader(text).load()


def yaml_plain(node: Any) -> Any:
    """The decoded document as plain ``dict``/``list``/``str`` (PyYAML BaseLoader shape)."""
    if isinstance(node, YamlScalar):
        return node.value
    if isinstance(node, dict):
        return {key: yaml_plain(value) for key, value in node.items()}
    return [yaml_plain(item) for item in node]


def _scalars(node: Any, key: str | None = None) -> list[tuple[str | None, YamlScalar]]:
    """Every scalar with the nearest enclosing mapping key (sequences inherit it)."""
    if isinstance(node, YamlScalar):
        return [(key, node)]
    out: list[tuple[str | None, YamlScalar]] = []
    if isinstance(node, dict):
        for child_key, child in node.items():
            out.extend(_scalars(child, child_key))
    else:
        for item in node:
            out.extend(_scalars(item, key))
    return out


def _script_lines(scalar: YamlScalar) -> list[tuple[int, str]]:
    """Logical shell lines of a scalar, each with its physical line where known.

    A literal block maps decoded lines one-to-one onto physical ones; a folded
    block or an inline value is reported at its first line.
    """
    first = scalar.line + 1 if scalar.block else scalar.line
    return [
        (first + offset - 1 if scalar.block == "|" else first, line)
        for offset, line in _logical_lines(scalar.value)
    ]


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash continuations, keeping the first physical line number."""
    out: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        if not pending:
            start = number
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            pending.append(stripped[:-1])
            continue
        pending.append(stripped)
        out.append((start, " ".join(pending)))
        pending = []
    if pending:
        out.append((start, " ".join(pending)))
    return out


def _segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SHELL_OPERATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _install_arguments(segment: list[str]) -> list[str] | None:
    """Arguments after ``install`` for a recognised pip command, else ``None``.

    Recognised: ``pip``/``pip3[.N]`` and
    ``python[3[.N]] [interpreter options] -m pip``, after optional leading
    ``VAR=value`` assignments, then pip's own global options, then
    ``install``. Global options are skipped rather than required absent:
    ``pip --disable-pip-version-check install x`` is an install too, and a
    parser that only looked for ``pip install`` side by side would let it
    through unchecked.
    """
    words = list(segment)
    while words and _ENV_ASSIGNMENT_RE.match(words[0]):
        words.pop(0)
    start: int | None = None
    if words and _PIP_RE.match(words[0]):
        start = 1
    elif words and _PYTHON_RE.match(words[0]):
        for index in range(1, len(words) - 1):
            if words[index] == "-m":
                if words[index + 1] == "pip":
                    start = index + 2
                break
    if start is None:
        return None
    index = start
    while index < len(words) and words[index].startswith("-"):
        name, has_inline, _value = words[index].partition("=")
        index += 1
        if name in _VALUE_OPTIONS and not has_inline:
            index += 1
    if index < len(words) and words[index] == "install":
        return words[index + 1 :]
    return None


def _mentions_pip_install(segment: list[str]) -> bool:
    """A ``pip`` word (any spelling) with ``install`` anywhere after it.

    The net for everything :func:`_install_arguments` does not recognise --
    ``uv pip install``, ``sudo pip install``, ``"$PY" -m pip install`` -- so
    that an unfamiliar spelling is reported instead of silently skipped.
    """
    for index, word in enumerate(segment):
        if _PIP_RE.match(word) and "install" in segment[index + 1 :]:
            return True
    return False


def find_pip_installs(text: str) -> tuple[list[PipInstall], list[str]]:
    """Every ``pip install`` in a workflow, plus the ones it cannot read.

    The second list holds ``"<line>: <reason>"`` entries for text that
    mentions a pip install the parser does not understand. Those fail the
    contract rather than being skipped: an unreadable install is an unverified
    install. That includes the whole workflow when it is outside the YAML
    subset, and a shell line that names both ``pip`` and ``install`` without a
    recognised invocation (``bash -c "pip install x"``).

    Every ``run`` value is read as shell text; so is every OTHER scalar that
    mentions both words -- over-inclusive on purpose: a false match can only
    produce an error, never hide an install.
    """
    try:
        document = load_workflow_yaml(text)
    except WorkflowYamlError as exc:
        return [], [
            f"{exc.line}: cannot read the workflow as YAML ({exc}); "
            "an unreadable workflow is an unverified one"
        ]
    installs: list[PipInstall] = []
    unreadable: list[str] = []
    for key, scalar in _scalars(document):
        if key != "run" and not ("pip" in scalar.value and "install" in scalar.value):
            continue
        for number, line in _script_lines(scalar):
            if "pip" not in line or "install" not in line:
                continue
            try:
                tokens = _tokens(line)
            except ValueError as exc:
                unreadable.append(f"{number}: cannot tokenise shell text ({exc})")
                continue
            recognised = False
            for segment in _segments(tokens):
                arguments = _install_arguments(segment)
                if arguments is None:
                    if _mentions_pip_install(segment):
                        unreadable.append(
                            f"{number}: unrecognised pip invocation {' '.join(segment)!r}"
                        )
                        recognised = True
                    continue
                installs.append(_parse_install(number, arguments))
                recognised = True
            if (
                not recognised
                and any("pip" in token for token in tokens)
                and any("install" in token for token in tokens)
            ):
                unreadable.append(
                    f"{number}: names pip and install but no recognised pip invocation "
                    f"{line.strip()!r}"
                )
    return installs, unreadable


def _parse_install(number: int, arguments: list[str]) -> PipInstall:
    requirement_files: list[str] = []
    positionals: list[str] = []
    flags: set[str] = set()
    index = 0
    while index < len(arguments):
        word = arguments[index]
        index += 1
        if word.startswith("-") and word != "-":
            name, has_inline, value = word.partition("=")
            flags.add(name)
            if name in _VALUE_OPTIONS and not has_inline and index < len(arguments):
                value = arguments[index]
                index += 1
            if name in _REQUIREMENT_OPTIONS and value:
                requirement_files.append(value)
            continue
        positionals.append(word)
    return PipInstall(
        line=number,
        requirement_files=tuple(requirement_files),
        positionals=tuple(positionals),
        flags=frozenset(flags),
    )


def classify(install: PipInstall) -> str | None:
    """``None`` when the install satisfies the contract, else the reason."""
    if install.flags & _EDITABLE_OPTIONS:
        return "editable install in a release workflow"
    if "--require-hashes" in install.flags:
        if install.positionals:
            return (
                f"--require-hashes install also names unhashed packages {list(install.positionals)}"
            )
        if not install.requirement_files:
            return "--require-hashes without a -r lock file"
        return None
    if (
        install.positionals
        and not install.requirement_files
        and "--no-deps" in install.flags
        and all(_LOCAL_WHEEL_RE.fullmatch(item) for item in install.positionals)
    ):
        return None
    return (
        "not hash-pinned: use --require-hashes -r <lock>, or --no-deps with literal "
        "dist/*.whl paths only"
    )


def parse_lock(text: str) -> tuple[Lock, list[str]]:
    """Parse a pip-compile hash lock; the second value lists its defects."""
    lock = Lock()
    errors: list[str] = []
    for raw in text.splitlines():
        header = _HEADER_PYTHON_RE.match(raw)
        if header:
            lock.python = header.group(1)
        command = _HEADER_COMMAND_RE.match(raw)
        if command:
            lock.command = command.group(1).strip()
    for number, line in _logical_lines(text):
        content = line.split(" #", 1)[0].strip()
        if not content or content.startswith("#"):
            continue
        if content.startswith("-"):
            errors.append(
                f"line {number}: option line {content.split()[0]!r} -- a lock "
                "must not pull in other files or indexes"
            )
            continue
        spec, _sep, rest = content.partition(" --hash=")
        hashes = re.findall(r"--hash=(\w+):(\S+)", "--hash=" + rest) if _sep else []
        leftover = re.sub(r"--hash=\S+", "", "--hash=" + rest).strip() if _sep else ""
        name_match = _NAME_RE.match(spec)
        if name_match is None:
            errors.append(f"line {number}: unreadable requirement {spec!r}")
            continue
        requirement, _semicolon, marker = (part.strip() for part in spec.partition(";"))
        pinned = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^=*\s]+)", requirement
        )
        if pinned is None:
            errors.append(f"line {number}: {requirement!r} is not an exact == pin")
            continue
        if not hashes:
            errors.append(f"line {number}: {requirement!r} carries no --hash")
            continue
        if leftover:
            errors.append(f"line {number}: unexpected text after the hashes {leftover!r}")
            continue
        bad = [
            f"{algorithm}:{digest}"
            for algorithm, digest in hashes
            if _DIGEST_LENGTHS.get(algorithm) != len(digest)
            or not re.fullmatch(r"[0-9a-f]+", digest)
        ]
        if bad:
            errors.append(f"line {number}: malformed digest(s) {bad}")
            continue
        # A marker decides whether pip installs the entry AT ALL on the lock's
        # interpreter (PR #170 review): `setuptools==84.0.0 ; python_version <
        # "3"` is skipped by pip on 3.12 ("markers ... don't match your
        # environment"), so it must not count as a pin -- neither for source
        # coverage nor for the build-backend floor. A marker the gate cannot
        # evaluate fails closed.
        if marker:
            verdict = evaluate_marker(marker, lock.python)
            if verdict is None:
                errors.append(
                    f"line {number}: cannot evaluate environment marker {marker!r} for "
                    f"the lock's Python {lock.python}; only python_version comparisons "
                    "joined by and/or are evaluated"
                )
                continue
            if not verdict:
                continue
        lock.pins[normalise_name(pinned.group(1))] = pinned.group(2)
    if not lock.pins:
        errors.append("the lock pins nothing")
    return lock, errors


def source_requirements(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """``(name, specifier)`` per top-level line of a ``.in`` file, plus defects.

    Refused, because the gate could not vouch for the lock against them:

    * option lines (``-r``, ``-c``, ``--index-url`` ...) -- they pull in input
      this check never reads;
    * environment markers -- the lock is compiled for exactly one interpreter,
      so a marker that excludes it would make the requirement legitimately
      absent from the lock and the coverage check below would misreport it,
      while a marker that includes it is redundant;
    * direct references (``name @ url``) -- not an exact, index-hashed pin.
    """
    requirements: list[tuple[str, str]] = []
    defects: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if content.startswith("-"):
            defects.append(f"line {number}: option line {content.split()[0]!r} is not supported")
            continue
        match = _NAME_RE.match(content)
        if match is None:
            defects.append(f"line {number}: unreadable requirement {content!r}")
            continue
        rest = content[match.end() :].strip()
        if rest.startswith("["):
            rest = rest.split("]", 1)[1].strip() if "]" in rest else rest
        if ";" in rest:
            defects.append(
                f"line {number}: environment marker in {content!r}; the lock holds one interpreter"
            )
            continue
        if rest.startswith("@"):
            defects.append(f"line {number}: direct reference {content!r} is not an index pin")
            continue
        requirements.append((normalise_name(match.group(1)), rest))
    return requirements, defects


def _release_tuple(version: str) -> tuple[int, ...] | None:
    return tuple(int(part) for part in version.split(".")) if _RELEASE_RE.match(version) else None


def _compare(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    a = left + (0,) * (width - len(left))
    b = right + (0,) * (width - len(right))
    return (a > b) - (a < b)


def satisfies(version: str, specifier: str) -> bool | None:
    """Whether ``version`` meets a comma-separated specifier.

    Only final releases (``N(.N)*``) and the operators ``~= == != <= >= < >``
    are evaluated; anything else returns ``None`` so the caller fails closed
    instead of guessing at pre-release or wildcard semantics.
    """
    have = _release_tuple(version)
    if have is None:
        return None
    for clause in filter(None, (part.strip() for part in specifier.split(","))):
        match = _SPEC_RE.match(clause)
        if match is None:
            return None
        operator, target = match.groups()
        want = _release_tuple(target)
        if want is None:
            return None
        order = _compare(have, want)
        if operator == "~=":
            if len(want) < 2:
                return None
            ok = order >= 0 and have[: len(want) - 1] == want[:-1]
        else:
            ok = {
                "==": order == 0,
                "!=": order != 0,
                "<=": order <= 0,
                ">=": order >= 0,
                "<": order < 0,
                ">": order > 0,
            }[operator]
        if not ok:
            return False
    return True


_MARKER_TOKEN_RE = re.compile(
    r"\s*(?:(?P<op><=|>=|==|!=|<|>)|(?P<string>'[^']*'|\"[^\"]*\")|(?P<paren>[()])"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_.]*))"
)
_MARKER_OPS = {
    "<": lambda order: order < 0,
    "<=": lambda order: order <= 0,
    ">": lambda order: order > 0,
    ">=": lambda order: order >= 0,
    "==": lambda order: order == 0,
    "!=": lambda order: order != 0,
}
_MARKER_MIRROR = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


def evaluate_marker(marker: str, python: str | None) -> bool | None:
    """A PEP 508 marker on the lock interpreter ``python`` (``"X.Y"``).

    Only ``python_version`` compared with a quoted final release, combined by
    ``and``/``or`` and parentheses, is evaluated. Anything else -- another
    variable, ``in``/``~=``/``===``, a wildcard or pre-release, an unknown
    interpreter -- returns ``None`` and the caller fails closed.
    """
    have = _release_tuple(python) if python else None
    if have is None:
        return None
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(marker):
        if marker[position:].strip() == "":
            break
        match = _MARKER_TOKEN_RE.match(marker, position)
        if match is None or match.end() == position:
            return None
        kind = match.lastgroup
        if kind is None:
            return None
        tokens.append((kind, match.group(kind)))
        position = match.end()
    cursor = 0

    def peek() -> tuple[str, str] | None:
        return tokens[cursor] if cursor < len(tokens) else None

    def take() -> tuple[str, str]:
        nonlocal cursor
        token = tokens[cursor]
        cursor += 1
        return token

    def comparison() -> bool | None:
        if len(tokens) - cursor < 3:
            return None
        left, op, right = take(), take(), take()
        if op[0] != "op":
            return None
        operator = op[1]
        if left == ("word", "python_version") and right[0] == "string":
            literal = right[1][1:-1]
        elif right == ("word", "python_version") and left[0] == "string":
            literal = left[1][1:-1]
            operator = _MARKER_MIRROR[operator]
        else:
            return None
        want = _release_tuple(literal)
        if want is None:
            return None
        return _MARKER_OPS[operator](_compare(have, want))

    def atom() -> bool | None:
        token = peek()
        if token == ("paren", "("):
            take()
            value = expression()
            if value is None or peek() != ("paren", ")"):
                return None
            take()
            return value
        return comparison()

    def conjunction() -> bool | None:
        value = atom()
        while value is not None and peek() == ("word", "and"):
            take()
            right = atom()
            value = None if right is None else (value and right)
        return value

    def expression() -> bool | None:
        value = conjunction()
        while value is not None and peek() == ("word", "or"):
            take()
            right = conjunction()
            value = None if right is None else (value or right)
        return value

    result = expression()
    if result is None or cursor != len(tokens):
        return None
    return result


def build_system_requires(pyproject_text: str) -> list[str] | None:
    """``[build-system] requires``, or ``None`` when it cannot be read."""
    if tomllib is None:  # pragma: no cover
        return None
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return None
    requires = data.get("build-system", {}).get("requires")
    if not isinstance(requires, list) or not all(isinstance(r, str) for r in requires):
        return None
    return list(requires)


def check_lock(lock_path: Path, root: Path) -> tuple[Lock | None, list[str]]:
    """Validate one lock file and its relation to its ``.in`` and pyproject."""
    shown = lock_path.relative_to(root) if lock_path.is_relative_to(root) else lock_path
    if not lock_path.is_file():
        return None, [f"{shown}: lock file does not exist"]
    lock, defects = parse_lock(lock_path.read_text(encoding="utf-8"))
    errors = [f"{shown}: {defect}" for defect in defects]
    if lock.python is None:
        errors.append(f"{shown}: no 'autogenerated by pip-compile with Python X.Y' header")
    if lock.command is None:
        errors.append(f"{shown}: no pip-compile command in the header")
    else:
        options = lock.command.split()
        if "--generate-hashes" not in options:
            errors.append(f"{shown}: header command does not --generate-hashes")
        if "--no-index" in options:
            errors.append(
                f"{shown}: header command carries --no-index (pip-tools/click "
                "header bug); regenerate with click<8.3, see release.in"
            )
    source = lock_path.with_suffix(".in")
    if not source.is_file():
        errors.append(f"{shown}: missing requirements source {source.name}")
    else:
        requirements, source_defects = source_requirements(source.read_text(encoding="utf-8"))
        errors.extend(f"{shown.parent / source.name}: {defect}" for defect in source_defects)
        missing = [name for name, _spec in requirements if name not in lock.pins]
        if missing:
            errors.append(f"{shown}: does not lock {missing} from {source.name}; recompile")
        # Coverage by NAME is not enough: `twine<7` added to the .in while the
        # lock still pins twine 7.0.0 passed the check above. Every specifier in
        # the source must hold for the version the lock actually installs.
        for name, specifier in requirements:
            if name not in lock.pins or not specifier:
                continue
            verdict = satisfies(lock.pins[name], specifier)
            if verdict is None:
                errors.append(
                    f"{shown}: cannot evaluate locked {name}=={lock.pins[name]} "
                    f"against {source.name} specifier {specifier!r}"
                )
            elif not verdict:
                errors.append(
                    f"{shown}: locked {name}=={lock.pins[name]} violates {source.name} "
                    f"specifier {specifier!r}; recompile"
                )
    requires = build_system_requires((root / "pyproject.toml").read_text(encoding="utf-8"))
    if requires is None:
        why = " (tomllib needs Python 3.11+)" if tomllib is None else ""
        errors.append(f"pyproject.toml: cannot read [build-system] requires{why}")
    else:
        for requirement in requires:
            name_match = _NAME_RE.match(requirement)
            if name_match is None:
                errors.append(f"pyproject.toml: unreadable build requirement {requirement!r}")
                continue
            name = normalise_name(name_match.group(1))
            specifier = requirement[name_match.end() :].split(";", 1)[0].strip()
            locked = lock.pins.get(name)
            if locked is None:
                errors.append(
                    f"{shown}: build backend requirement {requirement!r} is not "
                    "locked, but the release builds with --no-isolation"
                )
                continue
            verdict = satisfies(locked, specifier)
            if verdict is None:
                errors.append(
                    f"{shown}: cannot evaluate locked {name}=={locked} against {specifier!r}"
                )
            elif not verdict:
                errors.append(
                    f"{shown}: locked {name}=={locked} violates pyproject.toml "
                    f"build-system requirement {requirement!r}"
                )
    return lock, errors


_KEY_VALUE_RE = re.compile(r"^(\s*(?:-\s+)?)([A-Za-z0-9_.-]+):(?:\s+(.*))?$")


def plain_scalar_defects(text: str) -> list[str]:
    """Inline plain-scalar values that YAML cannot parse.

    Inside an unquoted (plain) scalar, ``": "`` and a trailing ``":"`` are
    mapping indicators, so ``run: pip install --only-binary :all: -r x`` makes
    the WHOLE file unparsable. A release workflow only runs when a release is
    published, so that error would first surface on release day -- and the
    zizmor gate does not catch it: measured with zizmor 1.30.1, an unparsable
    workflow is skipped with a warning and the run still exits 0 with "No
    findings". This is a narrow stdlib stand-in for a YAML parser, which the
    quality contract deliberately does not install: it reads only inline
    ``key: value`` lines and skips block-scalar bodies, which may contain
    anything.
    """
    defects: list[str] = []
    block_parent: int | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        indent = len(line) - len(line.lstrip())
        if block_parent is not None:
            if not line.strip() or indent > block_parent:
                continue
            block_parent = None
        match = _KEY_VALUE_RE.match(line)
        if match is None or line.lstrip().startswith("#"):
            continue
        key_column = len(match.group(1))
        value = (match.group(3) or "").split(" #", 1)[0].rstrip()
        if not value:
            continue
        if value[0] in "|>":
            block_parent = key_column
            continue
        if value[0] in "'\"[{&*!":
            continue
        if ": " in value or value.endswith(":"):
            defects.append(
                f"{number}: inline plain scalar {value!r} contains a YAML mapping "
                "indicator (': ' or trailing ':'); use a block scalar (|) or quotes"
            )
    return defects


def _python_versions(text: str) -> set[str]:
    """Every ``python-version`` value in the workflow, read structurally.

    A workflow outside the YAML subset yields none; :func:`find_pip_installs`
    has already failed it.
    """
    try:
        document = load_workflow_yaml(text)
    except WorkflowYamlError:
        return set()
    return {scalar.value for key, scalar in _scalars(document) if key == "python-version"}


def publish_evidence(text: str) -> tuple[list[str], str | None]:
    """Why a workflow counts as publishing to PyPI, and a read error if any.

    Structural (PR #170 review): every ``uses`` value naming the PyPA publish
    action, and every shell segment in which a publishing tool is followed by
    its upload sub-command -- on shlex TOKENS, so ``twine  upload`` and a
    backslash-continued ``twine``/``upload`` are the same command. A line that
    names a publishing tool but cannot be tokenised counts as publishing. The
    old substring net over the comment-stripped text is kept as an addition.
    A workflow outside the YAML subset returns the read error: the caller
    cannot rule out that it publishes.
    """
    evidence: list[str] = []
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    flattened = re.sub(r"\s+", " ", uncommented.replace("\\\n", " "))
    evidence.extend(f"mentions {marker!r}" for marker in PUBLISH_MARKERS if marker in flattened)
    try:
        document = load_workflow_yaml(text)
    except WorkflowYamlError as exc:
        return evidence, str(exc)
    for key, scalar in _scalars(document):
        if key == "uses":
            action = scalar.value.split("@", 1)[0].strip().lower()
            if action.startswith("pypa/gh-action-pypi-publish"):
                evidence.append(f"line {scalar.line}: uses {scalar.value!r}")
        for number, line in _script_lines(scalar):
            if not any(tool in line for tool in _PUBLISH_TOOLS):
                continue
            try:
                tokens = _tokens(line)
            except ValueError:
                evidence.append(f"line {number}: cannot tokenise {line.strip()!r}")
                continue
            for segment in _segments(tokens):
                words = [word.rsplit("/", 1)[-1] for word in segment]
                for position, word in enumerate(words):
                    verb = _PUBLISH_TOOLS.get(word)
                    if verb is not None and verb in words[position + 1 :]:
                        evidence.append(f"line {number}: {' '.join(segment)!r}")
                        break
    return evidence, None


def check_workflow(path: Path, root: Path) -> list[str]:
    """All contract violations of one release workflow."""
    text = path.read_text(encoding="utf-8")
    shown = path.relative_to(root) if path.is_relative_to(root) else path
    installs, unreadable = find_pip_installs(text)
    errors = [f"{shown}:{entry}" for entry in unreadable]
    errors.extend(f"{shown}:{entry}" for entry in plain_scalar_defects(text))
    locked_installs = 0
    python_versions = _python_versions(text)
    for install in installs:
        reason = classify(install)
        if reason is not None:
            errors.append(f"{shown}:{install.line}: {reason}")
            continue
        for requirement_file in install.requirement_files:
            locked_installs += 1
            lock, lock_errors = check_lock(root / requirement_file, root)
            errors.extend(lock_errors)
            if lock is None or lock.python is None:
                continue
            if not python_versions:
                errors.append(
                    f"{shown}: installs {requirement_file} but pins no python-version; "
                    "the lock only holds for the interpreter it was compiled with"
                )
            mismatched = sorted(v for v in python_versions if v != lock.python)
            if mismatched:
                errors.append(
                    f"{shown}: python-version {mismatched} differs from the Python "
                    f"{lock.python} that {requirement_file} was compiled for"
                )
    if locked_installs == 0:
        errors.append(f"{shown}: release workflow has no hash-locked tool install")
    return errors


def check(root: Path = ROOT) -> list[str]:
    """Every violation of the release hash-pin contract under ``root``."""
    workflows = root / ".github" / "workflows"
    errors: list[str] = []
    for name in RELEASE_WORKFLOWS:
        path = workflows / name
        if not path.is_file():
            errors.append(f".github/workflows/{name}: listed release workflow is missing")
            continue
        errors.extend(check_workflow(path, root))
    if workflows.is_dir():
        for path in sorted(workflows.iterdir()):
            if path.suffix not in {".yml", ".yaml"} or path.name in RELEASE_WORKFLOWS:
                continue
            evidence, unreadable = publish_evidence(path.read_text(encoding="utf-8"))
            if unreadable is not None:
                errors.append(
                    f".github/workflows/{path.name}: cannot be read as YAML ({unreadable}), "
                    "so the gate cannot rule out that it publishes to PyPI"
                )
            if evidence:
                errors.append(
                    f".github/workflows/{path.name}: publishes to PyPI but is not in "
                    f"RELEASE_WORKFLOWS, so its installs are not hash-checked ({evidence[0]})"
                )
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Release hash-pin check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Release hash-pin check passed for {', '.join(RELEASE_WORKFLOWS)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
