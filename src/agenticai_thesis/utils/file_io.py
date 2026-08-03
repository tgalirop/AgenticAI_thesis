"""Safe file input and output helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def write_json_atomic(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Serialize a mapping as UTF-8 JSON without exposing a partial final file.

    A profiling process may be interrupted while scanning millions of rows.  The
    report is therefore written to a temporary file in the destination directory
    and atomically moved into place only after JSON serialization succeeds.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
            )
            temporary.write("\n")

        os.replace(temporary_path, destination)
    finally:
        # If serialization fails, do not leave a misleading temporary report.
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination
