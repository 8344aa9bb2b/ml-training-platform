"""Job Submission REST API service for RMLTP."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request

from src.network.message_types import TaskMessage


@dataclass
class JobSubmission:
    """Represents a submitted distributed training job."""

    job_id: str
    job_name: str
    training_code: str
    dataset_location: str
    resource_requirements: Dict[str, Any]
    num_workers: int
    status: str = "pending"  # pending, scheduled, running, completed, failed, cancelled
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    assigned_workers: List[str] = field(default_factory=list)
    current_step: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


def create_job_api(controller) -> Flask:
    """Create Flask application exposing Job Submission API endpoints.

    Args:
        controller: The master Controller instance.

    Returns:
        Configured Flask app.
    """
    app = Flask(__name__)
    jobs_db: Dict[str, JobSubmission] = {}

    @app.route("/api/jobs/submit", methods=["POST"])
    def submit_job():
        data = request.get_json() or {}

        required_fields = ["training_code", "dataset_location", "resource_requirements", "num_workers"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        job_id = f"job-{uuid.uuid4().hex[:8]}"
        job = JobSubmission(
            job_id=job_id,
            job_name=data.get("job_name", f"job-{job_id}"),
            training_code=data["training_code"],
            dataset_location=data["dataset_location"],
            resource_requirements=data["resource_requirements"],
            num_workers=int(data["num_workers"]),
            hyperparameters=data.get("hyperparameters", {}),
        )

        jobs_db[job_id] = job

        # Create TaskMessages and enqueue in Controller
        task = TaskMessage(
            message_type="task_assignment",
            sender_id="job_api",
            task_id=job_id,
            task_type="training_job",
            parameters={
                "training_code": job.training_code,
                "dataset_location": job.dataset_location,
                "resource_requirements": job.resource_requirements,
                "num_workers": job.num_workers,
                "hyperparameters": job.hyperparameters,
            },
        )
        controller.add_task_to_queue(task)

        return jsonify(asdict(job)), 201

    @app.route("/api/jobs", methods=["GET"])
    def list_jobs():
        return jsonify([asdict(j) for j in jobs_db.values()]), 200

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def get_job(job_id: str):
        if job_id not in jobs_db:
            return jsonify({"error": f"Job {job_id} not found"}), 400
        return jsonify(asdict(jobs_db[job_id])), 200

    @app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
    def cancel_job(job_id: str):
        if job_id not in jobs_db:
            return jsonify({"error": f"Job {job_id} not found"}), 400
        job = jobs_db[job_id]
        job.status = "cancelled"
        return jsonify(asdict(job)), 200

    return app
