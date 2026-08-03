"""Checkpoint storage interfaces and atomic JSON implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from agenticai_thesis.agentic.state import AgentState
from agenticai_thesis.utils.file_io import write_json_atomic


@runtime_checkable
class CheckpointStoreProtocol(Protocol):
    """Persistence contract replaceable by a future LangGraph checkpointer."""

    def save(self, state: AgentState) -> Path: ...

    def load(self, run_id: str) -> AgentState: ...

    def exists(self, run_id: str) -> bool: ...


class JsonCheckpointStore:
    """Persist complete Agent states atomically as human-readable JSON."""

    _SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> Path:
        """Atomically replace the checkpoint for the state's stable run ID."""

        path = self._path_for(state.run_id)
        payload = state.model_dump(mode="json")
        return write_json_atomic(payload, path)

    def load(self, run_id: str) -> AgentState:
        """Load and fully validate a checkpoint before returning it."""

        path = self._path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Agent checkpoint not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        state = AgentState.model_validate(payload)
        if state.run_id != run_id:  # Defensive check against manual file replacement.
            raise ValueError("Checkpoint run_id does not match the requested identifier")
        return state

    def exists(self, run_id: str) -> bool:
        return self._path_for(run_id).is_file()

    def _path_for(self, run_id: str) -> Path:
        """Map a validated identifier to a path without permitting traversal."""

        if not self._SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id contains unsafe path characters")
        path = (self._directory / f"{run_id}.json").resolve()
        if path.parent != self._directory:
            raise ValueError("Checkpoint path escapes the configured directory")
        return path
