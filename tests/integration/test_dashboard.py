"""Unit test validating dashboard web endpoints."""

import pytest
from dashboard import app


class TestDashboardApp:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_dashboard_index(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"RMLTP" in response.data

    def test_dashboard_api_cluster(self, client):
        response = client.get("/api/cluster")
        assert response.status_code == 200
        data = response.get_json()
        assert "active_workers" in data
        assert "workers" in data
        assert "loss" in data

    def test_dashboard_train_step(self, client):
        response = client.post("/api/train_step")
        assert response.status_code == 200
        data = response.get_json()
        assert "step" in data
        assert "loss" in data

    def test_dashboard_simulate_crash(self, client):
        response = client.post("/api/simulate_crash")
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
