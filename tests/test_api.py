"""Unit and integration tests for NeuroCare-AI FastAPI application."""

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_root_endpoint_returns_hello_neurocare():
    """Verify root endpoint returns HTTP 200 and 'Hello NeuroCare'."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Hello NeuroCare"
    assert data["status"] == "online"


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "NeuroCare-AI"
    assert "services" in data
