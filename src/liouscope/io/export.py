"""JSON serialisation of :class:`DiagnosticReport`."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._types import DiagnosticReport, FitResult


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "__complex_array__": True,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
                "shape": list(value.shape),
            }
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, FitResult):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, complex):
        return {"__complex__": True, "real": value.real, "imag": value.imag}
    return value


def dump_report(report: DiagnosticReport, path: str | Path) -> None:
    """Serialise a :class:`DiagnosticReport` to JSON at ``path``."""
    obj = _to_jsonable(report)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a dumped report as a nested dictionary.

    The result is not converted back into the original dataclass tree; it is
    intended for downstream consumption (CI artefacts, plotting).
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
