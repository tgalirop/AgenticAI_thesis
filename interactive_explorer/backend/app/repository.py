"""Thread-safe repository for the immutable thesis snapshot."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.models import ExplorerPayload


class SnapshotRepository:
    """Load and validate the generated JSON once per backend process.

    The web service never traverses the main repository and never reads raw data.
    This explicit boundary prevents the explanatory app from accidentally
    mutating or exposing experiment files.
    """

    def __init__(self, snapshot_path: Path) -> None:
        self._snapshot_path = snapshot_path

    @lru_cache(maxsize=1)
    def load(self) -> ExplorerPayload:
        """Return the validated snapshot, caching it because it is immutable."""

        raw = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        return ExplorerPayload.model_validate(raw)


def default_repository() -> SnapshotRepository:
    """Build a repository using the snapshot copied into the backend image."""

    snapshot_path = Path(__file__).resolve().parents[1] / "data" / "thesis_snapshot.json"
    return SnapshotRepository(snapshot_path)
