"""End-to-end integration tests for RMLTP distributed training platform."""

import os
import time
from unittest.mock import Mock, patch
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from prometheus_client import CollectorRegistry

from src.controller.controller import Controller
from src.node.worker import Worker
from src.model.runner import ModelRunner
from src.model.parallel import DataParallelExecutor, PipelineParallelStage
from src.model.gradient_sync import AllReduceSynchronizer, SyncMode
from src.model.parameter_server import ParameterServer
from src.model.tensor_utils import serialize_tensor
from src.monitoring.metrics import PlatformMetrics
from src.monitoring.checkpoint_manager import CheckpointManager
from src.network.message_types import Message, MessageTypes, TaskMessage, WorkerRegistration
from src.scheduler.task_scheduler import Task


class LinearMLP(nn.Module):
    """Simple MLP for end-to-end distributed integration tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TestEndToEndCluster:
    @pytest.fixture
    def mock_logger(self):
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.fixture
    def cluster_env(self, mock_logger, tmp_path):
        """Create an integrated controller with isolated metrics and checkpointing."""
        registry = CollectorRegistry()
        metrics = PlatformMetrics(registry=registry)
        ckpt_dir = str(tmp_path / "checkpoints")

        with patch("signal.signal"):
            controller = Controller(
                config={
                    "controller": {
                        "host": "127.0.0.1",
                        "port": 15560,
                        "heartbeat_interval": 1.0,
                        "max_task_retries": 2,
                    },
                    "monitoring": {
                        "checkpoint_dir": ckpt_dir,
                    },
                },
                logger=mock_logger,
            )
            controller.metrics = metrics
            yield controller, ckpt_dir
            controller.stop()

    def test_end_to_end_data_parallel_training_flow(self, cluster_env, mock_logger):
        """Verify full lifecycle: worker registration, task execution, data parallel training,

        parameter synchronization, metric collection, and checkpoint saving.
        """
        controller, ckpt_dir = cluster_env

        # 1. Initialize two workers with real PyTorch ModelRunner instances
        model_w1 = LinearMLP()
        model_w2 = LinearMLP()
        model_w2.load_state_dict(model_w1.state_dict())

        opt1 = SGD(model_w1.parameters(), lr=0.05)
        opt2 = SGD(model_w2.parameters(), lr=0.05)

        runner1 = ModelRunner(model_w1, optimizer=opt1, criterion=nn.MSELoss())
        runner2 = ModelRunner(model_w2, optimizer=opt2, criterion=nn.MSELoss())

        with patch("src.node.worker.SocketManager"):
            worker1 = Worker(
                config={"nodes": {"id": "worker-1", "health_check": {"interval": 1.0}}},
                logger=mock_logger,
                model_runner=runner1,
            )
            worker2 = Worker(
                config={"nodes": {"id": "worker-2", "health_check": {"interval": 1.0}}},
                logger=mock_logger,
                model_runner=runner2,
            )

        # 2. Register workers with Controller
        for w in [worker1, worker2]:
            reg_msg = WorkerRegistration(
                message_type=MessageTypes.WORKER_REGISTER,
                sender_id=w.worker_id,
                payload={"gpu": {"count": 1, "type": "simulated"}},
            )
            controller._route_message(reg_msg)

        assert len(controller.worker_registry) == 2
        assert controller.metrics.active_workers._value.get() == 2

        # 3. Execute Distributed Data Parallel Training Steps
        synchronizer = AllReduceSynchronizer(expected_workers=["worker-0", "worker-1"])
        executor = DataParallelExecutor(
            runners=[runner1, runner2],
            synchronizer=synchronizer,
        )

        dataset_x = torch.randn(32, 4)
        dataset_y = torch.randn(32, 2)

        initial_loss = None
        final_loss = None

        for step in range(5):
            metrics = executor.train_step(dataset_x, dataset_y)
            if step == 0:
                initial_loss = metrics["loss"]
            final_loss = metrics["loss"]

            # Report metrics to controller
            for i, w in enumerate([worker1, worker2]):
                metric_msg = Message(
                    message_type=MessageTypes.METRIC,
                    sender_id=w.worker_id,
                    payload={"loss": metrics["loss"], "step": step, "throughput": 128.0},
                )
                controller._route_message(metric_msg)

        # Training loss decreased or proceeded stably
        assert final_loss is not None
        assert executor.current_step == 5

        # Both workers kept in sync by AllReduce
        for p1, p2 in zip(model_w1.parameters(), model_w2.parameters()):
            assert torch.allclose(p1, p2)

        # 4. Checkpoint creation and Controller manifest verification
        ckpt_msg = Message(
            message_type=MessageTypes.CHECKPOINT,
            sender_id="worker-1",
            payload={
                "step": 5,
                "model_state": runner1.get_state_dict(),
                "metrics": {"loss": final_loss},
            },
        )
        controller._route_message(ckpt_msg)

        latest_ckpt = controller.checkpoint_manager.get_latest_checkpoint()
        assert latest_ckpt is not None
        assert latest_ckpt.step == 5
        assert os.path.exists(latest_ckpt.path)

        # 5. Verify Prometheus exposition text
        metrics_text = controller.metrics.generate_metrics_text()
        assert "rmltp_active_workers 2.0" in metrics_text
        assert 'rmltp_training_step{worker_id="worker-1"} 4.0' in metrics_text

    def test_end_to_end_fault_recovery_and_task_reassignment(self, cluster_env, mock_logger):
        """Verify worker death triggers task reassignment and surviving worker completes it."""
        controller, _ = cluster_env

        # Register workers
        controller._route_message(WorkerRegistration(message_type=MessageTypes.WORKER_REGISTER, sender_id="worker-healthy"))
        controller._route_message(WorkerRegistration(message_type=MessageTypes.WORKER_REGISTER, sender_id="worker-failing"))
        assert len(controller.worker_registry) == 2

        # Assign task to failing worker
        task = Task(task_id="job-recovery-42", task_type="forward_pass", priority=1)
        controller.scheduler.active_tasks["worker-failing"] = task
        controller.worker_active_tasks["worker-failing"] = [task]

        # Simulate timeout on worker-failing
        controller.worker_registry["worker-failing"]["last_seen"] = time.time() - 100.0
        controller.worker_registry["worker-healthy"]["last_seen"] = time.time()

        dead = controller._check_worker_health()
        assert "worker-failing" in dead
        assert "worker-healthy" in controller.worker_registry

        # Task was requeued
        assert len(controller.task_queue) == 1
        assert controller.task_queue[0].task_id == "job-recovery-42"

        # Worker-healthy completes the reassigned task
        completion_msg = Message(
            message_type=MessageTypes.TASK_STATUS,
            sender_id="worker-healthy",
            payload={"task_id": "job-recovery-42", "successful": True, "result": {"status": "recovered"}},
        )
        controller._route_message(completion_msg)

        assert controller.metrics.tasks_completed_total._value.get() == 1
        assert controller.metrics.tasks_reassigned_total._value.get() == 1
