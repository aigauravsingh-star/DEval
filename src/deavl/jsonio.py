from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not supported.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite JSON numbers are not supported.")


def loads(value: str | bytes) -> Any:
    return json.loads(value, object_pairs_hook=_object, parse_constant=_constant)


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def snapshot(value: Any) -> Any:
    return loads(canonical(value))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write("\n")
