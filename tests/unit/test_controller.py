"""Unit tests for controller component."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.controller.controller import Controller, TaskMessage
from src.network.message_types import Message, MessageTypes, WorkerRegistration


class TestController:
    """Test suite for controller."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.fixture
    def controller(self, mock_logger):
        """Create a controller instance."""
        with patch('signal.signal'):
            controller = Controller(
                config={"controller": {"host": "127.0.0.1", "port": 15550}},
                logger=mock_logger
            )
            yield controller
            controller.stop()

    def test_worker_registration(self, controller):
        """Test worker registration and health monitoring."""
        # Create registration message
        reg_msg = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-1",
            payload={"gpu": {"count": 1, "type": "nvidia-rtx-3080", "memory": "10GiB"}},
        )

        # Handle registration
        controller._route_message(reg_msg)

        # Verify registration
        assert "worker-1" in controller.worker_registry
        assert controller.worker_registry["worker-1"]["gpu"]["count"] == 1

        # Send heartbeat to keep alive
        heartbeat = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="worker-1",
            timestamp=time.time()
        )
        controller._route_message(heartbeat)

        # Verify heartbeat updates registry
        assert controller.worker_registry["worker-1"]["last_seen"] > 0

        # Test health check
        time.sleep(0.5)  # Wait for health monitor to run
        controller._check_worker_health()

        # Verify still alive
        assert "worker-1" in controller.worker_registry

        # Simulate dead worker
        # Remove last seen time
        controller.worker_registry["worker-1"]["last_seen"] = time.time() - 40
        controller._check_worker_health()

        # Verify removed
        assert "worker-1" not in controller.worker_registry

    def test_health_monitoring(self, controller):
        """Test health monitoring functionality."""
        # Create a worker that's alive
        alive_msg = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="worker-2",
            timestamp=time.time()
        )
        controller._route_message(alive_msg)

        # Verify alive
        assert "worker-2" in controller.worker_registry

        # Simulate dead worker
        controller.worker_registry["worker-2"]["last_seen"] = time.time() - 35
        controller._check_worker_health()

        # Verify removed
        assert "worker-2" not in controller.worker_registry

    def test_task_queue(self, controller):
        """Test task queue management."""
        # Create task message
        task_msg = TaskMessage(
            message_type=MessageTypes.TASK_ASSIGNMENT,
            sender_id="controller",
            recipient_id="worker-1",
            task_id="task-123",
            task_type="forward_pass",
            parameters={"batch_size": 32},
        )

        # Add to queue
        controller.add_task_to_queue(task_msg)

        # Verify queue
        assert len(controller.task_queue) == 1
        assert controller.task_queue[0].task_id == "task-123"

    def test_worker_pool(self, controller):
        """Test worker pool management."""
        # Register two workers
        reg_msg1 = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-1",
            payload={"gpu": {"count": 1}},
        )
        controller._route_message(reg_msg1)

        reg_msg2 = WorkerRegistration(
            message_type=MessageTypes.WORKER_REGISTER,
            sender_id="worker-2",
            payload={"gpu": {"count": 2}},
        )
        controller._route_message(reg_msg2)

        # Test available workers
        available = controller.get_available_workers()
        assert len(available) == 2
        assert "worker-1" in available
        assert "worker-2" in available

        # Simulate worker going down
        controller.worker_registry["worker-1"]["last_seen"] = time.time() - 35
        controller._check_worker_health()

        available = controller.get_available_workers()
        assert len(available) == 1
        assert "worker-1" not in available
        assert "worker-2" in available

    def test_controller_shutdown(self, controller):
        """Test controller graceful shutdown."""
        controller.stop()
        assert controller.running is False

    def test_resource_monitor(self, controller):
        """Test resource monitoring."""
        # In production, this would interact with hardware APIs
        # For now, just verify the method exists
        assert hasattr(controller, "_resource_monitor_loop")

    def test_workload_scheduler(self, controller):
        """Test workload scheduler."""
        assert hasattr(controller, "_workload_scheduler")

    def test_job_execution(self, controller):
        """Test job execution."""
        # In production, this would run the actual job
        # For now, just verify the method exists
        assert hasattr(controller, "run_job")