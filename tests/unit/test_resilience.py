"""Unit tests for Controller resilience, fault detection, and task reassignment."""

import time
from unittest.mock import Mock, patch
import pytest

from src.controller.controller import Controller
from src.network.message_types import Message, MessageTypes, TaskMessage, WorkerRegistration
from src.scheduler.task_scheduler import Task
from src.monitoring.metrics import PlatformMetrics
from prometheus_client import CollectorRegistry


class TestControllerResilience:
    @pytest.fixture
    def mock_logger(self):
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.fixture
    def controller(self, mock_logger, tmp_path):
        registry = CollectorRegistry()
        metrics = PlatformMetrics(registry=registry)

        with patch("signal.signal"):
            ctrl = Controller(
                config={
                    "controller": {
                        "host": "127.0.0.1",
                        "port": 15555,
                        "heartbeat_interval": 1.0,
                        "max_task_retries": 2,
                    },
                    "monitoring": {
                        "checkpoint_dir": str(tmp_path / "ckpts"),
                    },
                },
                logger=mock_logger,
            )
            ctrl.metrics = metrics
            yield ctrl
            ctrl.stop()

    def test_worker_death_triggers_task_reassignment(self, controller):
        # 1. Register worker
        reg_msg = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-dead",
            payload={"compute": "gpu"},
        )
        controller._route_message(reg_msg)
        assert "worker-dead" in controller.worker_registry
        assert controller.metrics.active_workers._value.get() == 1

        # 2. Assign a task to the worker
        task = Task(
            task_id="task-important",
            task_type="forward_pass",
            priority=10,
        )
        controller.scheduler.active_tasks["worker-dead"] = task
        controller.worker_active_tasks["worker-dead"] = [task]

        # 3. Simulate passage of time to trigger worker death
        controller.worker_registry["worker-dead"]["last_seen"] = time.time() - 10.0

        dead_workers = controller._check_worker_health()
        assert "worker-dead" in dead_workers
        assert "worker-dead" not in controller.worker_registry
        assert controller.metrics.active_workers._value.get() == 0
        assert controller.metrics.dead_workers_total._value.get() == 1

        # Verify task was requeued
        assert len(controller.task_queue) == 1
        reassigned_task = controller.task_queue[0]
        assert reassigned_task.task_id == "task-important"
        assert controller.task_retry_counts["task-important"] == 1
        assert controller.metrics.tasks_reassigned_total._value.get() == 1

    def test_max_retries_exhaustion(self, controller):
        reg_msg = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-fail",
        )
        controller._route_message(reg_msg)

        task = Task(task_id="task-doomed", task_type="train")
        controller.scheduler.active_tasks["worker-fail"] = task
        controller.worker_active_tasks["worker-fail"] = [task]

        # Pre-set retry count to max
        controller.task_retry_counts["task-doomed"] = 2  # max is 2
        controller.worker_registry["worker-fail"]["last_seen"] = time.time() - 10.0

        dead_workers = controller._check_worker_health()
        assert "worker-fail" in dead_workers

        # Should NOT requeue because retries exceeded max
        assert len(controller.task_queue) == 0
        assert controller.metrics.tasks_failed_total._value.get() == 1

    def test_metrics_reporting_integration(self, controller):
        metric_msg = Message(
            message_type=MessageTypes.METRIC,
            sender_id="worker-1",
            payload={
                "loss": 0.35,
                "step": 20,
                "gradient_norm": 0.8,
            },
        )
        controller._route_message(metric_msg)

        text = controller.metrics.generate_metrics_text()
        assert 'rmltp_training_loss{worker_id="worker-1"} 0.35' in text
        assert 'rmltp_training_step{worker_id="worker-1"} 20.0' in text
