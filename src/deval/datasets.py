from __future__ import annotations

from pathlib import Path

from .jsonio import loads
from .types import EvalCase


def read_jsonl(path: str | Path) -> tuple[EvalCase, ...]:
    cases = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = loads(line)
            if not isinstance(row, dict) or set(row) - {"id", "state", "labels"}:
                raise ValueError("Invalid case fields.")
            cases.append(EvalCase(**row))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid evaluation case on line {line_number}.") from None
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must be non-empty with unique case IDs.")
    return tuple(cases)
