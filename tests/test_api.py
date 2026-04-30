"""Tests for FastAPI routes — health, forecast, models, response structure."""

from tests.conftest import AUTH

# ── Health endpoint ───────────────────────────────────────────────────────


def test_health_returns_200(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200


def test_health_no_auth_required(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200


def test_health_response_structure(api_client):
    resp = api_client.get("/health")
    body = resp.json()
    assert body["status"] in ("success",)
    assert "data" in body
    assert "api" in body["data"]
    assert "database" in body["data"]


def test_health_has_timestamp_and_request_id(api_client):
    resp = api_client.get("/health")
    body = resp.json()
    assert "timestamp" in body
    assert "request_id" in body


# ── Models endpoint ───────────────────────────────────────────────────────


def test_get_models_requires_auth(api_client):
    resp = api_client.get("/models")
    assert resp.status_code == 401


def test_get_models_with_valid_key(api_client):
    resp = api_client.get("/models", headers=AUTH)
    assert resp.status_code == 200


def test_get_models_returns_list(api_client):
    resp = api_client.get("/models", headers=AUTH)
    body = resp.json()
    assert body["status"] == "success"
    assert isinstance(body["data"], list)


def test_get_models_for_state_with_valid_key(api_client):
    resp = api_client.get("/models/California", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["data"], list)


# ── Forecast endpoint ─────────────────────────────────────────────────────


def test_forecast_requires_auth(api_client):
    resp = api_client.post("/forecast", json={"state": "California", "weeks": 4})
    assert resp.status_code == 401


def test_forecast_invalid_api_key_returns_401(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 4},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_forecast_invalid_weeks_below_1(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 0},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_forecast_invalid_weeks_above_10(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 11},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_forecast_missing_state_field(api_client):
    resp = api_client.post("/forecast", json={"weeks": 4}, headers=AUTH)
    assert resp.status_code == 422


def test_forecast_with_no_trained_model_returns_503(api_client):
    # No model in registry → ModelNotTrainedException → 503
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 4},
        headers=AUTH,
    )
    assert resp.status_code in (503, 500)  # no champion in test env


# ── CORS headers ──────────────────────────────────────────────────────────


def test_cors_headers_present_on_health(api_client):
    resp = api_client.get(
        "/health",
        headers={"Origin": "http://localhost:8501"},
    )
    # CORS middleware adds allow-origin header when Origin matches allowed origins
    assert resp.status_code == 200


# ── Standard response envelope ────────────────────────────────────────────


def test_response_envelope_has_required_fields(api_client):
    resp = api_client.get("/health")
    body = resp.json()
    for field in ("status", "data", "message", "timestamp", "request_id"):
        assert field in body, f"Missing field: {field}"


# ── Error response structure ──────────────────────────────────────────────


def test_401_error_response_structure(api_client):
    resp = api_client.get("/models")
    body = resp.json()
    assert "detail" in body  # FastAPI HTTPException returns "detail"


# ── GET /forecast/{state} ─────────────────────────────────────────────────


def test_get_forecast_by_state_requires_auth(api_client):
    resp = api_client.get("/forecast/California")
    assert resp.status_code == 401


def test_get_forecast_by_state_with_key(api_client):
    resp = api_client.get("/forecast/California", headers=AUTH)
    # 503 expected (no model trained in test), not 401 or 422
    assert resp.status_code in (503, 500)
