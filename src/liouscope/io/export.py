"""JSON serialisation of :class:`DiagnosticReport`."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._types import DiagnosticReport, FitResult


def _nonfinite_tag(value: float) -> dict[str, str]:
    if math.isnan(value):
        return {"__nonfinite__": "nan"}
    return {"__nonfinite__": "inf" if value > 0 else "-inf"}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "__complex_array__": True,
                "real": _to_jsonable(value.real.tolist()),
                "imag": _to_jsonable(value.imag.tolist()),
                "shape": list(value.shape),
            }
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _to_jsonable(value.item())
    if isinstance(value, FitResult):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, complex):
        return {
            "__complex__": True,
            "real": _to_jsonable(value.real),
            "imag": _to_jsonable(value.imag),
        }
    # Non-finite floats (evidence ratios are legitimately inf when a gap
    # floors to 0) must not reach json.dumps' default allow_nan=True path:
    # that emits bare ``Infinity``/``NaN`` tokens, which are NOT valid JSON
    # (RFC 8259) and are rejected by strict consumers (JavaScript JSON.parse,
    # Postgres jsonb, serde, ...). They are tagged like complex values.
    if isinstance(value, float) and not math.isfinite(value):
        return _nonfinite_tag(value)
    return value


def dump_report(report: DiagnosticReport, path: str | Path) -> None:
    """Serialise a :class:`DiagnosticReport` to JSON at ``path``.

    Parent directories are created if missing so a caller-supplied artefact
    path (e.g. ``out/run/report.json``) does not fail with a bare
    ``FileNotFoundError`` on the first write.
    """
    obj = _to_jsonable(report)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False enforces the RFC 8259 guarantee: if a non-finite float
    # ever escapes _to_jsonable untagged, dumping fails loudly instead of
    # silently writing an unparseable artefact.
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a dumped report as a nested dictionary.

    The result is not converted back into the original dataclass tree; it is
    intended for downstream consumption (CI artefacts, plotting). Non-finite
    floats appear as ``{"__nonfinite__": "inf" | "-inf" | "nan"}`` tags
    (mirroring the ``__complex__`` tagging) so the file stays RFC 8259 valid.

    Fails closed with structured, actionable errors rather than reading the
    file raw: a missing path or malformed/non-object JSON is reported with the
    offending path, instead of a bare ``FileNotFoundError`` or a
    ``json.JSONDecodeError`` with no context (the "exists()-then-read-raw"
    bug class).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a regular file.
    ValueError
        If the file is not valid JSON, or its top level is not a JSON object.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"report file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"report file {p} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"report file {p} must contain a JSON object at top level, "
            f"got {type(loaded).__name__}"
        )
    result: dict[str, Any] = loaded
    return result
