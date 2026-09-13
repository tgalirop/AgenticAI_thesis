"""Contract checks for the isolated explorer backend."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_exposes_snapshot_version() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_explorer_contains_every_scientific_section() -> None:
    response = client.get("/api/explorer")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["workflow"]) >= 10
    assert payload["datasets"]["raw"]["rows"] == 6_362_620
    assert payload["test_evaluation"]["metrics"]
    assert payload["cross_validation"]["paired_tests"]
