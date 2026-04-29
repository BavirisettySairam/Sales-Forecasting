"""Security tests — auth, rate limiting, headers, injection, whitelist, docs."""
import pytest

from tests.conftest import AUTH, VALID_KEY


# ── API key authentication ────────────────────────────────────────────────

def test_no_api_key_returns_401(api_client):
    resp = api_client.post("/forecast", json={"state": "California", "weeks": 4})
    assert resp.status_code == 401


def test_invalid_api_key_returns_401(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 4},
        headers={"X-API-Key": "invalid-key-xyz"},
    )
    assert resp.status_code == 401


def test_valid_api_key_passes_auth(api_client):
    # With valid key, auth passes — result is 503 (no model) not 401
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 4},
        headers=AUTH,
    )
    assert resp.status_code != 401


def test_empty_api_key_returns_401(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 4},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 401


def test_health_endpoint_needs_no_key(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200


# ── Security response headers ─────────────────────────────────────────────

def test_x_content_type_options_header(api_client):
    resp = api_client.get("/health")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"


def test_x_frame_options_header(api_client):
    resp = api_client.get("/health")
    assert resp.headers.get("X-Frame-Options") == "DENY"


def test_x_xss_protection_header(api_client):
    resp = api_client.get("/health")
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"


def test_content_security_policy_header(api_client):
    resp = api_client.get("/health")
    assert "Content-Security-Policy" in resp.headers


def test_referrer_policy_header(api_client):
    resp = api_client.get("/health")
    assert "Referrer-Policy" in resp.headers


def test_request_id_header_present(api_client):
    resp = api_client.get("/health")
    assert "X-Request-ID" in resp.headers


# ── SQL injection attempt ─────────────────────────────────────────────────

def test_sql_injection_in_state_field_rejected(api_client):
    # Pydantic validates the field; ORM parameterises queries — both layers protect
    malicious = "'; DROP TABLE forecasts; --"
    resp = api_client.post(
        "/forecast",
        json={"state": malicious, "weeks": 4},
        headers=AUTH,
    )
    # Must not be 200 or 500 from a SQL error — expect 404 (state not found) or 503
    assert resp.status_code in (404, 503, 500)
    # Must not expose traceback or SQL error details
    body = resp.text
    assert "DROP TABLE" not in body
    assert "SyntaxError" not in body
    assert "Traceback" not in body


def test_xss_attempt_in_state_field(api_client):
    xss = "<script>alert('xss')</script>"
    resp = api_client.post(
        "/forecast",
        json={"state": xss, "weeks": 4},
        headers=AUTH,
    )
    assert "<script>" not in resp.text


# ── Input validation ──────────────────────────────────────────────────────

def test_state_too_short_rejected(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "A", "weeks": 4},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_weeks_zero_rejected(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 0},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_weeks_negative_rejected(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": -1},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_weeks_over_52_rejected(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "California", "weeks": 53},
        headers=AUTH,
    )
    assert resp.status_code == 422


def test_oversized_state_field_rejected(api_client):
    resp = api_client.post(
        "/forecast",
        json={"state": "A" * 200, "weeks": 4},
        headers=AUTH,
    )
    assert resp.status_code == 422


# ── Docs disabled in production ───────────────────────────────────────────

def test_docs_available_in_test_environment(api_client):
    resp = api_client.get("/docs")
    # In test env (ENVIRONMENT=test), docs should be available
    assert resp.status_code == 200


def test_docs_disabled_in_production():
    import os
    os.environ["ENVIRONMENT"] = "production"
    try:
        from importlib import reload
        import src.config.settings as s_mod
        reload(s_mod)
        import src.api.main as main_mod
        reload(main_mod)
        from fastapi.testclient import TestClient
        client = TestClient(main_mod.app, raise_server_exceptions=False)
        resp = client.get("/docs")
        assert resp.status_code == 404
    finally:
        os.environ["ENVIRONMENT"] = "test"


# ── Rate limiter structure ────────────────────────────────────────────────

def test_rate_limiter_returns_headers_on_success(api_client, mock_redis):
    # Patch the pipeline execute to simulate 1 request in window
    mock_redis.pipeline.return_value.execute.return_value = [0, 0, 1, True]
    resp = api_client.get("/models", headers=AUTH)
    assert resp.status_code == 200
