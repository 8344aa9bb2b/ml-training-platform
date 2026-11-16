"""Unit test validating dashboard web endpoints."""

from unittest.mock import patch
import pytest

from dashboard import app, _harness, get_harness


class TestDashboardApp:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with patch("src.network.socket_manager.SocketManager._bind"):
            # Ensure harness initialized with mocked bind
            get_harness()
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

    def test_dashboard_submit_job(self, client):
        payload = {
            "job_name": "sentiment-analysis",
            "dataset_location": "s3://imdb/reviews.csv",
            "num_workers": 2,
        }
        response = client.post("/api/jobs/submit", json=payload)
        assert response.status_code == 201
        data = response.get_json()
        assert data["name"] == "sentiment-analysis"
        assert data["workers"] == 2

    def test_dashboard_list_jobs(self, client):
        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 1
