"""Unit tests for Worker node."""

import os
from unittest.mock import Mock, patch
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.node.worker import Worker
from src.network.message_types import TaskMessage, ControlMessage
from src.model.runner import ModelRunner
from src.model.tensor_utils import serialize_tensor


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestWorker:
    @pytest.fixture
    def mock_logger(self):
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        return logger

    @pytest.fixture
    def simulated_worker(self, mock_logger):
        with patch("src.node.worker.SocketManager"):
            worker = Worker(
                config={"nodes": {"id": "worker-sim", "health_check": {"interval": 1.0}}},
                logger=mock_logger,
            )
            return worker

    @pytest.fixture
    def model_worker(self, mock_logger):
        model = SimpleNet()
        optimizer = SGD(model.parameters(), lr=0.01)
        runner = ModelRunner(model=model, optimizer=optimizer, criterion=nn.MSELoss())

        with patch("src.node.worker.SocketManager"):
            worker = Worker(
                config={"nodes": {"id": "worker-model", "health_check": {"interval": 1.0}}},
                logger=mock_logger,
                model_runner=runner,
            )
            return worker

    def test_simulated_worker_tasks(self, simulated_worker):
        res_fwd = simulated_worker._execute_forward_pass({"batch_size": 16})
        assert res_fwd["status"] == "completed"
        assert res_fwd["batch_size"] == 16
        assert "loss" in res_fwd

        res_bwd = simulated_worker._execute_backward_pass({})
        assert res_bwd["status"] == "completed"
        assert "gradient_norm" in res_bwd

        res_opt = simulated_worker._execute_optimizer_step({"learning_rate": 0.05})
        assert res_opt["status"] == "completed"
        assert res_opt["learning_rate"] == 0.05

    def test_model_worker_tasks(self, model_worker, tmp_path):
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        # Forward pass with real tensors
        params = {
            "inputs": serialize_tensor(x),
            "targets": serialize_tensor(y),
            "batch_size": 8,
        }
        res_fwd = model_worker._execute_forward_pass(params)
        assert res_fwd["status"] == "completed"
        assert res_fwd["batch_size"] == 8
        assert res_fwd["loss"] >= 0.0

        # Optimizer step
        res_opt = model_worker._execute_optimizer_step({"learning_rate": 0.01})
        assert res_opt["status"] == "completed"

        # Checkpoint save
        ckpt_dir = str(tmp_path / "ckpts")
        res_ckpt = model_worker._save_checkpoint({"save_directory": ckpt_dir, "step": 5})
        assert res_ckpt["status"] == "completed"
        assert os.path.exists(res_ckpt["checkpoint_path"])

    def test_handle_task_assignment(self, model_worker):
        task = TaskMessage(
            message_type="task_assignment",
            sender_id="controller",
            task_id="task-001",
            task_type="forward_pass",
            parameters={"batch_size": 4},
        )

        model_worker.comm_manager.send_message = Mock()
        model_worker._handle_task_assignment(task)

        # Check that report_task_status called send_message
        assert model_worker.comm_manager.send_message.called
        sent_msg = model_worker.comm_manager.send_message.call_args[0][0]
        assert sent_msg.payload["successful"] is True
        assert sent_msg.payload["task_id"] == "task-001"

    def test_handle_control_stop(self, simulated_worker):
        simulated_worker.running = True
        ctrl = ControlMessage(message_type="control", sender_id="controller", control_type="stop")
        simulated_worker._handle_control_message(ctrl)
        assert simulated_worker.running is False

    def test_connect_to_controller_tcp_handshake(self, mock_logger):
        """Verify real TCP registration handshake between Worker and Controller socket."""
        import time
        from src.network.socket_manager import SocketManager
        from src.network.message_types import MessageTypes

        server_port = 15582
        server = SocketManager(
            host="127.0.0.1",
            port=server_port,
            logger=mock_logger,
            pattern="reqrep",
            is_server=True,
        )

        worker = Worker(
            config={"nodes": {"id": "worker-test-tcp", "port": 15583}},
            logger=mock_logger,
        )

        try:
            worker.connect_to_controller(f"tcp://127.0.0.1:{server_port}")
            time.sleep(0.05)

            received = server.listen(timeout_ms=1000)
            assert len(received) == 1
            reg = received[0]
            assert reg.message_type == MessageTypes.WORKER_REGISTER
            assert reg.sender_id == "worker-test-tcp"
            assert "cpu" in reg.payload
            assert reg.payload["cpu"]["count"] >= 1
        finally:
            worker.stop()
            server.stop()
