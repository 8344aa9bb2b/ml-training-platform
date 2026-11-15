"""Unit tests for Prometheus metrics collection and reporting."""

import pytest
from prometheus_client import CollectorRegistry

from src.monitoring.metrics import PlatformMetrics


class TestPlatformMetrics:
    @pytest.fixture
    def metrics(self):
        # Use isolated registry per test to avoid global collision
        registry = CollectorRegistry()
        return PlatformMetrics(registry=registry)

    def test_worker_metrics(self, metrics):
        metrics.set_active_workers(5)
        text = metrics.generate_metrics_text()
        assert "rmltp_active_workers 5.0" in text

        metrics.record_dead_worker("worker-1")
        text = metrics.generate_metrics_text()
        assert "rmltp_dead_workers_total 1.0" in text

    def test_task_lifecycle_metrics(self, metrics):
        metrics.record_task_scheduled("task-1")
        metrics.record_task_scheduled("task-2")
        metrics.record_task_completed("task-1")
        metrics.record_task_failed("task-2")
        metrics.record_task_reassigned("task-2")

        text = metrics.generate_metrics_text()
        assert "rmltp_tasks_scheduled_total 2.0" in text
        assert "rmltp_tasks_completed_total 1.0" in text
        assert "rmltp_tasks_failed_total 1.0" in text
        assert "rmltp_tasks_reassigned_total 1.0" in text

    def test_training_metrics(self, metrics):
        metrics.record_training_metrics(
            worker_id="worker-0",
            metrics={
                "loss": 0.425,
                "step": 100,
                "gradient_norm": 1.25,
                "throughput": 512.0,
            },
        )
        text = metrics.generate_metrics_text()
        assert 'rmltp_training_loss{worker_id="worker-0"} 0.425' in text
        assert 'rmltp_training_step{worker_id="worker-0"} 100.0' in text
        assert 'rmltp_gradient_norm{worker_id="worker-0"} 1.25' in text
        assert 'rmltp_throughput_samples_sec{worker_id="worker-0"} 512.0' in text

    def test_system_resource_metrics(self, metrics):
        resources = {
            "cpu": {"percent": 45.2},
            "memory": {"rss_gb": 4.5},
            "gpu": {0: {"memory_allocated_gb": 3.8}},
        }
        metrics.record_system_resources(resources)
        text = metrics.generate_metrics_text()
        assert "rmltp_cpu_percent 45.2" in text
        assert "rmltp_memory_used_gb 4.5" in text
        assert 'rmltp_gpu_memory_allocated_gb{device="0"} 3.8' in text

    def test_http_server_lifecycle(self, metrics):
        # High port to avoid any collisions
        server = metrics.start_server(port=19099, host="127.0.0.1")
        assert metrics.is_server_running
        metrics.stop_server()
        assert not metrics.is_server_running
