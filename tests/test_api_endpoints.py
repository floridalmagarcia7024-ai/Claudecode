"""Tests for API REST endpoints (Module 17)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.main import (
    create_dashboard_app,
    create_jwt_token,
    verify_jwt_token,
    set_components,
    JWT_SECRET,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a test client for the dashboard app."""
    from dashboard.main import dashboard_app
    return TestClient(dashboard_app)


@pytest.fixture
def auth_headers():
    """Generate valid JWT auth headers."""
    token = create_jwt_token("admin")
    return {"Authorization": f"Bearer {token}"}


# ── JWT Tests ─────────────────────────────────────────────────


class TestJWT:
    def test_create_token(self):
        token = create_jwt_token("test_user")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self):
        token = create_jwt_token("test_user")
        payload = verify_jwt_token(token)
        assert payload["sub"] == "test_user"

    def test_verify_invalid_token(self):
        with pytest.raises(Exception):
            verify_jwt_token("invalid.token.here")

    def test_verify_expired_token(self):
        import jwt as pyjwt
        from datetime import datetime, timedelta, timezone

        payload = {
            "sub": "test",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        token = pyjwt.encode(payload, JWT_SECRET, algorithm="HS256")
        with pytest.raises(Exception):
            verify_jwt_token(token)


# ── Auth Endpoint Tests ──────────────────────────────────────


class TestAuthEndpoints:
    def test_login_success(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "polymarket"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_wrong_username(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "wrong", "password": "polymarket"},
        )
        assert response.status_code == 401


# ── Health Endpoint ──────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_no_engine(self, client):
        set_components()  # Reset all to None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_unauthenticated(self, client):
        """Health endpoint should NOT require auth."""
        response = client.get("/health")
        assert response.status_code == 200


# ── Protected Endpoint Tests ─────────────────────────────────


class TestProtectedEndpoints:
    def test_status_requires_auth(self, client):
        response = client.get("/api/status")
        assert response.status_code in (401, 403)

    def test_positions_requires_auth(self, client):
        response = client.get("/api/positions")
        assert response.status_code in (401, 403)

    def test_trades_requires_auth(self, client):
        response = client.get("/api/trades")
        assert response.status_code in (401, 403)

    def test_metrics_requires_auth(self, client):
        response = client.get("/api/metrics")
        assert response.status_code in (401, 403)

    def test_config_requires_auth(self, client):
        response = client.post(
            "/api/config",
            json={"key": "zscore_threshold", "value": 2.0},
        )
        assert response.status_code in (401, 403)


# ── Config Endpoint Tests ────────────────────────────────────


class TestConfigEndpoint:
    def test_update_allowed_key(self, client, auth_headers):
        response = client.post(
            "/api/config",
            json={"key": "zscore_threshold", "value": 2.0},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"

    def test_update_disallowed_key(self, client, auth_headers):
        response = client.post(
            "/api/config",
            json={"key": "polymarket_api_key", "value": "hacked"},
            headers=auth_headers,
        )
        assert response.status_code == 400


# ── Dashboard HTML Tests ─────────────────────────────────────


class TestDashboardHTML:
    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Polymarket" in response.text

    def test_calibration_page(self, client):
        response = client.get("/calibration")
        assert response.status_code == 200
        assert "Calibration" in response.text

    def test_onboarding_page(self, client):
        response = client.get("/onboarding")
        assert response.status_code == 200
        assert "Setup" in response.text


# ── Regime Endpoint Tests ────────────────────────────────────


class TestRegimeEndpoint:
    def test_regime_no_detector(self, client, auth_headers):
        set_components()  # No regime detector
        response = client.get("/api/regime/test_market", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["regime"] == "unknown"


# ── Security Headers ─────────────────────────────────────────


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
