"""Demonstrable Web Dashboard for RMLTP Distributed ML Training Platform.

Run this with:
    python dashboard.py

Then open your browser at:
    http://localhost:5000
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import SGD
from flask import Flask, jsonify, render_template_string, request

from src.controller.controller import Controller
from src.node.worker import Worker
from src.model.runner import ModelRunner
from src.model.parallel import DataParallelExecutor
from src.model.gradient_sync import AllReduceSynchronizer
from src.network.message_types import Message, MessageTypes, WorkerRegistration
from src.scheduler.task_scheduler import Task
from core.logger import get_logger


# Minimal Toy Neural Network
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ClusterDemoHarness:
    """Simulates a live cluster for demonstration purposes."""

    def __init__(self):
        self.logger = get_logger("dashboard_demo")
        self.controller = Controller(
            config={
                "controller": {
                    "host": "127.0.0.1",
                    "port": 15590,
                    "heartbeat_interval": 2.0,
                    "max_task_retries": 3,
                },
                "monitoring": {
                    "checkpoint_dir": "./logs/demo_checkpoints",
                },
            },
            logger=self.logger,
        )

        # Initialize 2 Worker nodes with ModelRunners
        m1 = DemoModel()
        m2 = DemoModel()
        m2.load_state_dict(m1.state_dict())

        self.runner1 = ModelRunner(m1, SGD(m1.parameters(), lr=0.05), nn.MSELoss())
        self.runner2 = ModelRunner(m2, SGD(m2.parameters(), lr=0.05), nn.MSELoss())

        self.worker1 = Worker(
            config={"nodes": {"id": "worker-alpha", "port": 15591, "health_check": {"interval": 1.0}}},
            logger=self.logger,
            model_runner=self.runner1,
        )
        self.worker2 = Worker(
            config={"nodes": {"id": "worker-beta", "port": 15592, "health_check": {"interval": 1.0}}},
            logger=self.logger,
            model_runner=self.runner2,
        )

        # Register workers
        self._register_workers()

        # Data Parallelism Executor
        self.sync = AllReduceSynchronizer(expected_workers=["worker-0", "worker-1"])
        self.executor = DataParallelExecutor(runners=[self.runner1, self.runner2], synchronizer=self.sync)

        self.step = 0
        self.latest_loss = 0.85
        self.latest_throughput = 0.0
        self.jobs = [
            {
                "id": "job-init-01",
                "name": "cifar10-resnet50",
                "dataset": "s3://ml-datasets/cifar10",
                "workers": 2,
                "status": "running",
                "step": 0,
            }
        ]
        self.logs = ["Cluster initialized with 2 worker nodes (worker-alpha, worker-beta)."]

    def _register_workers(self):
        self.controller._route_message(
            WorkerRegistration(
                message_type=MessageTypes.WORKER_REGISTER,
                sender_id="worker-alpha",
                payload={"gpu": {"count": 1, "type": "NVIDIA RTX 4090"}},
            )
        )
        self.controller._route_message(
            WorkerRegistration(
                message_type=MessageTypes.WORKER_REGISTER,
                sender_id="worker-beta",
                payload={"gpu": {"count": 1, "type": "NVIDIA RTX 4090"}},
            )
        )

    def train_step(self):
        """Execute one distributed training step across active workers."""
        x = torch.randn(64, 4)
        y = torch.randn(64, 2)

        t0 = time.perf_counter()
        metrics = self.executor.train_step(x, y)
        elapsed = time.perf_counter() - t0

        self.step = metrics["step"]
        self.latest_loss = round(metrics["loss"], 4)
        self.latest_throughput = round(64.0 / max(0.0001, elapsed), 1)

        if self.jobs:
            self.jobs[0]["step"] = self.step
            self.jobs[0]["status"] = "running"

        # Report to Controller metrics
        self.controller.metrics.record_training_metrics(
            "worker-alpha",
            {"loss": self.latest_loss, "step": self.step, "throughput": self.latest_throughput},
        )
        self.controller.metrics.record_training_metrics(
            "worker-beta",
            {"loss": self.latest_loss, "step": self.step, "throughput": self.latest_throughput},
        )

        msg = f"[Step {self.step}] AllReduce Sync completed | Loss: {self.latest_loss:.4f} | Throughput: {self.latest_throughput:.1f} samples/sec"
        self.logs.insert(0, msg)
        if len(self.logs) > 15:
            self.logs.pop()

        return {
            "step": self.step,
            "loss": self.latest_loss,
            "throughput": self.latest_throughput,
        }

    def simulate_crash(self):
        """Simulate worker-beta missing heartbeats and crashing."""
        if "worker-beta" not in self.controller.worker_registry:
            # Revive
            self.controller.register_worker("worker-beta", {"gpu": {"count": 1, "type": "NVIDIA RTX 4090"}})
            self.controller.worker_registry["worker-beta"]["last_seen"] = time.time()
            self.controller.metrics.set_active_workers(len(self.controller.worker_registry))
            msg = "[Revival] worker-beta re-connected and registered with Controller."
            self.logs.insert(0, msg)
            return {"status": "revived", "message": msg}

        # Assign a task
        task = Task(task_id=f"job-{self.step + 100}", task_type="data_parallel_batch", priority=5)
        self.controller.scheduler.active_tasks["worker-beta"] = task
        self.controller.worker_active_tasks["worker-beta"] = [task]

        # Trigger heartbeat failure
        self.controller.worker_registry["worker-beta"]["last_seen"] = time.time() - 50.0
        dead = self.controller._check_worker_health()

        msg = f"[Failure Detected] {dead} missed 3 heartbeats. Task {task.task_id} automatically reassigned to worker-alpha."
        self.logs.insert(0, msg)
        return {"status": "failed", "message": msg}

    def submit_job(self, job_data: dict) -> dict:
        """Handle new user job submission."""
        job_id = f"job-{uuid.uuid4().hex[:6]}"
        job = {
            "id": job_id,
            "name": job_data.get("job_name", "custom-job"),
            "dataset": job_data.get("dataset_location", "data/default.csv"),
            "workers": int(job_data.get("num_workers", 2)),
            "status": "pending",
            "step": 0,
        }
        self.jobs.insert(0, job)
        msg = f"[Job Submitted] {job['name']} ({job['id']}) queued with {job['workers']} workers."
        self.logs.insert(0, msg)
        return job

    def get_cluster_status(self):
        # Refresh hardware usage
        res = self.controller.resource_monitor.get_resources()
        if res:
            self.controller.metrics.record_system_resources(res, node_id="controller")

        cpu = res.get("cpu", {}).get("percent", 24.5) if res else 24.5
        mem = res.get("memory", {}).get("rss_gb", 3.2) if res else 3.2

        workers = []
        for wid, info in self.controller.worker_registry.items():
            workers.append({
                "id": wid,
                "status": info.get("status", "active"),
                "last_seen": round(time.time() - info.get("last_seen", time.time()), 1),
                "gpu": info.get("gpu", {}).get("type", "NVIDIA RTX 4090"),
                "cpu_percent": round(cpu * (0.9 if wid == "worker-alpha" else 1.1), 1),
                "memory_gb": round(mem * 0.45, 2),
            })

        return {
            "active_workers": len(self.controller.worker_registry),
            "dead_workers": int(self.controller.metrics.dead_workers_total._value.get()),
            "tasks_reassigned": int(self.controller.metrics.tasks_reassigned_total._value.get()),
            "step": self.step,
            "loss": self.latest_loss,
            "throughput": self.latest_throughput,
            "workers": workers,
            "jobs": self.jobs,
            "logs": self.logs,
            "controller_cpu": round(cpu, 1),
            "controller_mem_gb": round(mem, 2),
        }


# Initialize App & Lazy Harness
app = Flask(__name__)
_harness: Optional[ClusterDemoHarness] = None


def get_harness() -> ClusterDemoHarness:
    """Lazy initialize demo harness so importing dashboard doesn't immediately bind ports."""
    global _harness
    if _harness is None:
        _harness = ClusterDemoHarness()
    return _harness

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RMLTP — Distributed ML Platform Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
        .metric-val { font-size: 2.2rem; font-weight: 700; color: #38bdf8; }
        .metric-label { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; }
        .badge-active { background-color: #10b981; }
        .badge-dead { background-color: #ef4444; }
        .log-box { background-color: #020617; border: 1px solid #1e293b; font-family: monospace; font-size: 0.85rem; height: 160px; overflow-y: auto; padding: 10px; border-radius: 8px; }
        .btn-custom { font-weight: 600; border-radius: 8px; padding: 8px 18px; }
        .form-control, .form-select { background-color: #0f172a; border: 1px solid #334155; color: #f8fafc; }
        .form-control:focus, .form-select:focus { background-color: #0f172a; border-color: #38bdf8; color: #f8fafc; box-shadow: none; }
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <!-- Header -->
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h2 class="fw-bold mb-1"><span class="text-primary">RMLTP</span> Distributed ML Platform</h2>
                <p class="text-secondary mb-0">Kubernetes-style Orchestration + Ray Scheduling + PyTorch Distributed Core</p>
            </div>
            <div>
                <button class="btn btn-outline-info btn-custom me-2" data-bs-toggle="modal" data-bs-target="#jobModal">➕ Submit Training Job</button>
                <button class="btn btn-primary btn-custom me-2" onclick="trainStep()">⚡ Run Distributed Step</button>
                <button class="btn btn-outline-danger btn-custom" onclick="simulateCrash()">💥 Toggle Worker Crash</button>
            </div>
        </div>

        <!-- Top Metric Counters -->
        <div class="row g-3 mb-4">
            <div class="col-md-2">
                <div class="card p-3 text-center">
                    <div class="metric-label">Active Nodes</div>
                    <div class="metric-val" id="val-nodes">2</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card p-3 text-center">
                    <div class="metric-label">Training Step</div>
                    <div class="metric-val text-info" id="val-step">0</div>
                </div>
            </div>
            <div class="col-md-2">
                <div class="card p-3 text-center">
                    <div class="metric-label">Current Loss</div>
                    <div class="metric-val text-warning" id="val-loss">0.8500</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="metric-label">Throughput (samples/s)</div>
                    <div class="metric-val text-success" id="val-tp">0.0</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card p-3 text-center">
                    <div class="metric-label">Failures Recovered</div>
                    <div class="metric-val text-danger" id="val-recovered">0</div>
                </div>
            </div>
        </div>

        <!-- Cluster Nodes Telemetry -->
        <div class="row g-4 mb-4">
            <div class="col-md-8">
                <div class="card p-4">
                    <h5 class="fw-bold mb-3">🖥️ Live Cluster Nodes Telemetry</h5>
                    <div class="table-responsive">
                        <table class="table table-dark table-hover align-middle mb-0">
                            <thead>
                                <tr class="text-secondary">
                                    <th>Node ID</th>
                                    <th>Status</th>
                                    <th>GPU Accelerator</th>
                                    <th>CPU Usage</th>
                                    <th>Memory (RAM)</th>
                                    <th>Last Heartbeat</th>
                                </tr>
                            </thead>
                            <tbody id="workers-tbody">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <!-- Controller Stats -->
            <div class="col-md-4">
                <div class="card p-4 h-100">
                    <h5 class="fw-bold mb-3">⚙️ Controller Master</h5>
                    <p class="mb-1 text-secondary">ZeroMQ Port: <strong class="text-light">15590</strong></p>
                    <p class="mb-1 text-secondary">Prometheus Port: <strong class="text-light">9090</strong></p>
                    <p class="mb-3 text-secondary">Placement: <strong class="text-light">Least Loaded / Capability Aware</strong></p>

                    <div class="mb-2">
                        <div class="d-flex justify-content-between small text-secondary">
                            <span>Controller CPU:</span>
                            <span id="ctrl-cpu">24%</span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-info" id="ctrl-cpu-bar" style="width: 24%"></div>
                        </div>
                    </div>
                    <div class="mb-3">
                        <div class="d-flex justify-content-between small text-secondary">
                            <span>Controller RAM:</span>
                            <span id="ctrl-mem">3.2 GB</span>
                        </div>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-warning" id="ctrl-mem-bar" style="width: 35%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Submitted Jobs Queue -->
        <div class="card p-4 mb-4">
            <h5 class="fw-bold mb-3">📋 Distributed Training Jobs</h5>
            <div class="table-responsive">
                <table class="table table-dark table-hover align-middle mb-0">
                    <thead>
                        <tr class="text-secondary">
                            <th>Job ID</th>
                            <th>Job Name</th>
                            <th>Dataset Location</th>
                            <th>Workers</th>
                            <th>Status</th>
                            <th>Step</th>
                        </tr>
                    </thead>
                    <tbody id="jobs-tbody">
                        <!-- Populated dynamically -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Activity Logs -->
        <div class="card p-4">
            <h5 class="fw-bold mb-2">📜 Real-Time Distributed Event Stream</h5>
            <div class="log-box" id="log-box"></div>
        </div>
    </div>

    <!-- Job Submission Modal -->
    <div class="modal fade" id="jobModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content card border-secondary text-light">
                <div class="modal-header border-secondary">
                    <h5 class="modal-title fw-bold">Submit Distributed Training Job</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label small text-secondary">Job Name</label>
                        <input type="text" id="input-job-name" class="form-control" value="bert-finetuning">
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-secondary">Dataset Location</label>
                        <input type="text" id="input-dataset" class="form-control" value="s3://datasets/nlp/imdb.parquet">
                    </div>
                    <div class="row g-2 mb-3">
                        <div class="col-6">
                            <label class="form-label small text-secondary">Number of Workers</label>
                            <input type="number" id="input-workers" class="form-control" value="2" min="1" max="8">
                        </div>
                        <div class="col-6">
                            <label class="form-label small text-secondary">GPU per Worker</label>
                            <input type="number" id="input-gpu" class="form-control" value="1" min="0">
                        </div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label small text-secondary">Parallelism Mode</label>
                        <select id="input-parallel" class="form-select">
                            <option value="data_parallel_allreduce">Data Parallel (AllReduce Synchronous)</option>
                            <option value="data_parallel_ps">Data Parallel (Parameter Server)</option>
                            <option value="pipeline_parallel">Pipeline Parallel Stages</option>
                        </select>
                    </div>
                </div>
                <div class="modal-footer border-secondary">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="submitJobModal()">Submit Job</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/api/cluster');
                const data = await res.json();
                
                document.getElementById('val-nodes').innerText = data.active_workers;
                document.getElementById('val-step').innerText = data.step;
                document.getElementById('val-loss').innerText = data.loss.toFixed(4);
                document.getElementById('val-tp').innerText = data.throughput.toFixed(1);
                document.getElementById('val-recovered').innerText = data.tasks_reassigned;

                document.getElementById('ctrl-cpu').innerText = data.controller_cpu + '%';
                document.getElementById('ctrl-cpu-bar').style.width = Math.min(100, data.controller_cpu) + '%';
                document.getElementById('ctrl-mem').innerText = data.controller_mem_gb + ' GB';

                let tbody = '';
                for (const w of data.workers) {
                    tbody += `<tr>
                        <td><strong>${w.id}</strong></td>
                        <td><span class="badge ${w.status === 'active' ? 'badge-active' : 'badge-dead'}">${w.status}</span></td>
                        <td><small>${w.gpu}</small></td>
                        <td>
                            <div class="d-flex align-items-center">
                                <span class="me-2 small">${w.cpu_percent}%</span>
                                <div class="progress flex-grow-1" style="height: 5px;">
                                    <div class="progress-bar bg-primary" style="width: ${w.cpu_percent}%"></div>
                                </div>
                            </div>
                        </td>
                        <td>${w.memory_gb} GB</td>
                        <td><small class="text-secondary">${w.last_seen}s ago</small></td>
                    </tr>`;
                }
                document.getElementById('workers-tbody').innerHTML = tbody;

                let jobsTbody = '';
                for (const j of data.jobs) {
                    jobsTbody += `<tr>
                        <td><code>${j.id}</code></td>
                        <td><strong>${j.name}</strong></td>
                        <td><small class="text-secondary">${j.dataset}</small></td>
                        <td>${j.workers}</td>
                        <td><span class="badge bg-primary">${j.status}</span></td>
                        <td>${j.step}</td>
                    </tr>`;
                }
                document.getElementById('jobs-tbody').innerHTML = jobsTbody;

                let logHtml = '';
                for (const line of data.logs) {
                    logHtml += `<div>${line}</div>`;
                }
                document.getElementById('log-box').innerHTML = logHtml;
            } catch (err) {
                console.error("Error fetching cluster status:", err);
            }
        }

        async function trainStep() {
            await fetch('/api/train_step', { method: 'POST' });
            fetchStatus();
        }

        async function simulateCrash() {
            await fetch('/api/simulate_crash', { method: 'POST' });
            fetchStatus();
        }

        async function submitJobModal() {
            const payload = {
                job_name: document.getElementById('input-job-name').value,
                dataset_location: document.getElementById('input-dataset').value,
                num_workers: parseInt(document.getElementById('input-workers').value),
                training_code: document.getElementById('input-parallel').value,
                resource_requirements: { gpu: parseInt(document.getElementById('input-gpu').value) }
            };
            await fetch('/api/jobs/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const modalEl = document.getElementById('jobModal');
            bootstrap.Modal.getInstance(modalEl).hide();
            fetchStatus();
        }

        // Auto poll every 1.5 seconds
        setInterval(fetchStatus, 1500);
        fetchStatus();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/cluster", methods=["GET"])
def get_cluster():
    return jsonify(get_harness().get_cluster_status())


@app.route("/api/train_step", methods=["POST"])
def run_step():
    res = get_harness().train_step()
    return jsonify(res)


@app.route("/api/simulate_crash", methods=["POST"])
def simulate_crash():
    res = get_harness().simulate_crash()
    return jsonify(res)


@app.route("/api/jobs/submit", methods=["POST"])
def submit_job():
    data = request.get_json() or {}
    job = get_harness().submit_job(data)
    return jsonify(job), 201


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify(get_harness().jobs), 200


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  🚀 RMLTP Distributed Machine Learning Platform Dashboard")
    print("  Running at: http://localhost:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
