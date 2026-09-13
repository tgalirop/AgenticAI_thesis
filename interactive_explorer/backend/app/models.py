"""Pydantic response contracts for the explorer API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Small health contract used by Docker Compose and the browser."""

    status: str
    snapshot_version: str


class ExplorerPayload(BaseModel):
    """Validated top-level contract for the generated scientific snapshot.

    Nested chart payloads intentionally remain dictionaries/lists. This keeps the
    API stable when a new visualisation is added, while the mandatory top-level
    sections still fail fast if the generated snapshot is incomplete.
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_version: str
    generated_at_utc: str
    source_run_id: str
    methodology: dict[str, Any]
    learning: dict[str, Any]
    langgraph: dict[str, Any]
    workflow: list[dict[str, Any]]
    datasets: dict[str, Any]
    quality: dict[str, Any]
    cross_validation: dict[str, Any]
    test_evaluation: dict[str, Any]
    resources: dict[str, Any]
    selected_plan: dict[str, Any]
    caveats: list[str] = Field(min_length=1)
