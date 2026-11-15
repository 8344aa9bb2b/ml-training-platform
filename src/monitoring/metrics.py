"""Prometheus metrics exporter and monitoring collector."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
    start_http_server,
)


class PlatformMetrics:
    """Manages Prometheus metrics collection and HTTP exposition for RMLTP."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        """Initialize PlatformMetrics.

        Args:
            registry: Optional custom Prometheus CollectorRegistry. If None, uses a new isolated registry.
        """
        self.registry = registry or CollectorRegistry()

        # Cluster & Worker Metrics
        self.active_workers = Gauge(
            "rmltp_active_workers",
            "Number of currently active workers registered in cluster",
            registry=self.registry,
        )
        self.dead_workers_total = Counter(
            "rmltp_dead_workers_total",
            "Total dead or timed-out workers detected",
            registry=self.registry,
        )

        # Task Lifecycle Metrics
        self.tasks_scheduled_total = Counter(
            "rmltp_tasks_scheduled_total",
            "Total tasks accepted and scheduled",
            registry=self.registry,
        )
        self.tasks_completed_total = Counter(
            "rmltp_tasks_completed_total",
            "Total tasks completed successfully",
            registry=self.registry,
        )
        self.tasks_failed_total = Counter(
            "rmltp_tasks_failed_total",
            "Total tasks that encountered errors",
            registry=self.registry,
        )
        self.tasks_reassigned_total = Counter(
            "rmltp_tasks_reassigned_total",
            "Total tasks rescheduled due to worker death or failure",
            registry=self.registry,
        )

        # Training Metrics (labeled by worker_id)
        self.training_loss = Gauge(
            "rmltp_training_loss",
            "Latest training loss reported by worker",
            ["worker_id"],
            registry=self.registry,
        )
        self.training_step = Gauge(
            "rmltp_training_step",
            "Current training step reported by worker",
            ["worker_id"],
            registry=self.registry,
        )
        self.gradient_norm = Gauge(
            "rmltp_gradient_norm",
            "Latest gradient norm computed by worker",
            ["worker_id"],
            registry=self.registry,
        )
        self.throughput = Gauge(
            "rmltp_throughput_samples_sec",
            "Training throughput in samples per second",
            ["worker_id"],
            registry=self.registry,
        )

        # System & Hardware Metrics
        self.cpu_percent = Gauge(
            "rmltp_cpu_percent",
            "System CPU utilization percentage",
            registry=self.registry,
        )
        self.node_cpu_percent = Gauge(
            "rmltp_node_cpu_percent",
            "Per-node CPU utilization percentage",
            ["node_id"],
            registry=self.registry,
        )
        self.memory_used_gb = Gauge(
            "rmltp_memory_used_gb",
            "System RSS memory used in GB",
            registry=self.registry,
        )
        self.node_memory_used_gb = Gauge(
            "rmltp_node_memory_used_gb",
            "Per-node RSS memory used in GB",
            ["node_id"],
            registry=self.registry,
        )
        self.gpu_memory_allocated_gb = Gauge(
            "rmltp_gpu_memory_allocated_gb",
            "GPU memory allocated in GB",
            ["device"],
            registry=self.registry,
        )
        self.node_gpu_memory_allocated_gb = Gauge(
            "rmltp_node_gpu_memory_allocated_gb",
            "Per-node GPU memory allocated in GB",
            ["node_id", "device"],
            registry=self.registry,
        )

        # Server handle
        self._server = None
        self._server_thread = None
        self._lock = threading.Lock()

    @property
    def is_server_running(self) -> bool:
        with self._lock:
            return self._server is not None

    def start_server(self, port: int = 9090, host: str = "0.0.0.0") -> None:
        """Start the Prometheus metrics HTTP export server.

        Args:
            port: Port to bind Prometheus HTTP endpoint.
            host: Interface address.
        """
        with self._lock:
            if self._server is not None:
                return

            server, thread = start_http_server(
                port=port,
                addr=host,
                registry=self.registry,
            )
            self._server = server
            self._server_thread = thread

    def stop_server(self) -> None:
        """Shut down the Prometheus metrics HTTP server."""
        with self._lock:
            if self._server is not None:
                try:
                    self._server.shutdown()
                    self._server.server_close()
                except Exception:
                    pass
                self._server = None
                self._server_thread = None

    def set_active_workers(self, count: int) -> None:
        """Update active workers gauge."""
        self.active_workers.set(count)

    def record_dead_worker(self, worker_id: str) -> None:
        """Increment dead worker counter."""
        self.dead_workers_total.inc()

    def record_task_scheduled(self, task_id: str) -> None:
        """Increment tasks scheduled counter."""
        self.tasks_scheduled_total.inc()

    def record_task_completed(self, task_id: str) -> None:
        """Increment tasks completed counter."""
        self.tasks_completed_total.inc()

    def record_task_failed(self, task_id: str) -> None:
        """Increment tasks failed counter."""
        self.tasks_failed_total.inc()

    def record_task_reassigned(self, task_id: str) -> None:
        """Increment tasks reassigned counter."""
        self.tasks_reassigned_total.inc()

    def record_training_metrics(self, worker_id: str, metrics: Dict[str, Any]) -> None:
        """Record batch training metrics received from worker."""
        if "loss" in metrics:
            self.training_loss.labels(worker_id=worker_id).set(float(metrics["loss"]))
        if "step" in metrics:
            self.training_step.labels(worker_id=worker_id).set(float(metrics["step"]))
        if "gradient_norm" in metrics:
            self.gradient_norm.labels(worker_id=worker_id).set(float(metrics["gradient_norm"]))
        if "throughput" in metrics:
            self.throughput.labels(worker_id=worker_id).set(float(metrics["throughput"]))

    def record_system_resources(self, resources: Dict[str, Any], node_id: str = "controller") -> None:
        """Record system and hardware utilization metrics for a given node.

        Args:
            resources: Dictionary containing CPU, memory, and GPU usage.
            node_id: Identifier of the node (e.g. 'controller', 'worker-1').
        """
        if not resources:
            return

        cpu = resources.get("cpu", {})
        if "percent" in cpu:
            val = float(cpu["percent"])
            self.cpu_percent.set(val)
            self.node_cpu_percent.labels(node_id=node_id).set(val)

        memory = resources.get("memory", {})
        if "rss_gb" in memory:
            val = float(memory["rss_gb"])
            self.memory_used_gb.set(val)
            self.node_memory_used_gb.labels(node_id=node_id).set(val)

        gpu = resources.get("gpu", {})
        if isinstance(gpu, dict) and gpu != {"none": 0.0}:
            for dev_id, usage in gpu.items():
                if isinstance(usage, dict) and "memory_allocated_gb" in usage:
                    mem_val = float(usage["memory_allocated_gb"])
                    self.gpu_memory_allocated_gb.labels(device=str(dev_id)).set(mem_val)
                    self.node_gpu_memory_allocated_gb.labels(node_id=node_id, device=str(dev_id)).set(mem_val)

    def generate_metrics_text(self) -> str:
        """Generate Prometheus exposition text format string."""
        return generate_latest(self.registry).decode("utf-8")
