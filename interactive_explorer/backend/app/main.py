"""FastAPI entry point for the Agentic AI Thesis Interactive Explorer."""

from __future__ import annotations

from fastapi import FastAPI

from app.models import ExplorerPayload, HealthResponse
from app.repository import default_repository


app = FastAPI(
    title="Agentic AI Thesis Explorer API",
    description=(
        "Read-only API exposing verified, aggregate evidence from the PaySim "
        "Conventional-versus-Agentic experiment."
    ),
    version="1.0.0",
)
repository = default_repository()


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Confirm that both the API and validated snapshot are available."""

    snapshot = repository.load()
    return HealthResponse(status="ok", snapshot_version=snapshot.snapshot_version)


@app.get("/api/explorer", response_model=ExplorerPayload, tags=["evidence"])
def explorer() -> ExplorerPayload:
    """Return all methodology and chart evidence in one cache-friendly response."""

    return repository.load()
