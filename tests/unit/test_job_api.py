"""Unit tests for Job Submission API."""

import pytest
from unittest.mock import Mock

from src.controller.job_api import create_job_api, JobSubmission
from src.controller.controller import Controller


class TestJobSubmissionAPI:
    @pytest.fixture
    def mock_controller(self):
        controller = Mock(spec=Controller)
        controller.worker_registry = {"worker-1": {}, "worker-2": {}}
        controller.add_task_to_queue = Mock()
        return controller

    @pytest.fixture
    def client(self, mock_controller):
        app = create_job_api(mock_controller)
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_submit_job_valid(self, client):
        payload = {
            "job_name": "resnet-cifar10",
            "training_code": "model = torch.nn.Linear(10, 2)",
            "dataset_location": "s3://datasets/cifar10.tar.gz",
            "resource_requirements": {"cpu": 4, "gpu": 1, "memory_gb": 8},
            "num_workers": 2,
            "hyperparameters": {"lr": 0.01, "batch_size": 64},
        }

        res = client.post("/api/jobs/submit", json=payload)
        assert res.status_code == 201
        data = res.get_json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["num_workers"] == 2

    def test_submit_job_missing_fields(self, client):
        res = client.post("/api/jobs/submit", json={"job_name": "bad-job"})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_get_jobs_and_detail(self, client):
        # Submit a job
        payload = {
            "job_name": "test-job-list",
            "training_code": "train()",
            "dataset_location": "/data/train",
            "resource_requirements": {"cpu": 2},
            "num_workers": 1,
        }
        res = client.post("/api/jobs/submit", json=payload)
        job_id = res.get_json()["job_id"]

        # List jobs
        list_res = client.get("/api/jobs")
        assert list_res.status_code == 200
        jobs = list_res.get_json()
        assert len(jobs) >= 1

        # Get details
        detail_res = client.get(f"/api/jobs/{job_id}")
        assert detail_res.status_code == 200
        detail = detail_res.get_json()
        assert detail["job_id"] == job_id
        assert detail["job_name"] == "test-job-list"
