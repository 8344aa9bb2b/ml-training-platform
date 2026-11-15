"""Unit tests for network message types."""

import pytest
import time
import torch
import numpy as np
from src.network.message_types import (
    Message,
    MessageTypes,
    TaskMessage,
    TensorMessage,
    ControlMessage,
    MetricMessage,
)


class TestMessageTypes:
    """Test suite for base message types."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            message_type=MessageTypes.HEARTBEAT,
            sender_id="controller",
        )
        assert msg.message_type == MessageTypes.HEARTBEAT
        assert msg.sender_id == "controller"
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, float)

    def test_message_serialization_deserialization(self):
        """Test message serialization and deserialization."""
        msg = Message(
            message_type=MessageTypes.TASK_ASSIGNMENT,
            sender_id="scheduler",
            recipient_id="worker-1",
            message_id="msg-123",
        )
        msg_dict = msg.__dict__
        reconstructed = Message(**msg_dict)
        assert reconstructed.message_type == msg.message_type
        assert reconstructed.sender_id == reconstructed.sender_id


class TestTaskMessage:
    """Test suite for task messages."""

    def test_task_message_creation(self):
        """Test task message creation."""
        task_msg = TaskMessage(
            message_type=MessageTypes.TASK_ASSIGNMENT,
            sender_id="scheduler",
            recipient_id="worker-1",
            task_id="task-42",
            task_type="forward_pass",
            parameters={"batch_size": 32},
            priority=1,
        )
        assert task_msg.task_id == "task-42"
        assert task_msg.task_type == "forward_pass"
        assert task_msg.parameters == {"batch_size": 32}
        assert task_msg.priority == 1

    def test_default_priority(self):
        """Test that priority defaults to 0."""
        task_msg = TaskMessage(
            message_type=MessageTypes.TASK_ASSIGNMENT,
            sender_id="scheduler",
        )
        assert task_msg.priority == 0


class TestTensorMessage:
    """Test suite for tensor messages."""

    def test_tensor_message_with_torch(self):
        """Test tensor message with PyTorch tensor."""
        tensor = torch.randn(10, 10)
        tensor_msg = TensorMessage(
            message_type=MessageTypes.TENSOR_DATA,
            sender_id="worker",
            recipient_id="controller",
            tensor_name="gradients",
            tensor_data=tensor,
            shape=list(tensor.shape),
            dtype="float32",
            compression="none",
        )
        assert tensor_msg.tensor_name == "gradients"
        assert tensor_msg.shape == [10, 10]
        assert tensor_msg.dtype == "float32"

    def test_tensor_message_with_numpy(self):
        """Test tensor message with NumPy array."""
        array = np.random.randn(5, 5)
        tensor_msg = TensorMessage(
            message_type=MessageTypes.TENSOR_DATA,
            sender_id="worker",
            recipient_id="controller",
            tensor_name="activations",
            tensor_data=array,
            shape=list(array.shape),
            dtype="float64",
            compression="none",
        )
        assert tensor_msg.tensor_name == "activations"

    def test_tensor_message_compression(self):
        """Test tensor message with compression specified."""
        tensor_msg = TensorMessage(
            message_type=MessageTypes.TENSOR_DATA,
            sender_id="worker",
            recipient_id="controller",
            tensor_name="weights",
            tensor_data=torch.randn(50, 50),
            shape=[50, 50],
            dtype="float32",
            compression="gzip",
        )
        assert tensor_msg.compression == "gzip"


class TestControlMessage:
    """Test suite for control messages."""

    def test_sync_barrier_message(self):
        """Test control message for synchronization barrier."""
        control_msg = ControlMessage(
            message_type=MessageTypes.CONTROL,
            sender_id="scheduler",
            recipient_id="worker",
            control_type="sync_barrier",
            expected_barriers=10,
            barrier_id="barrier-001",
        )
        assert control_msg.control_type == "sync_barrier"
        assert control_msg.expected_barriers == 10

    def test_reset_message(self):
        """Test control message for reset."""
        control_msg = ControlMessage(
            message_type=MessageTypes.CONTROL,
            sender_id="scheduler",
            control_type="reset",
            parameters={"step": 500, "model_state": "before_epoch_5"},
        )
        assert control_msg.parameters is not None
        assert control_msg.parameters["step"] == 500

    def test_no_expected_barriers(self):
        """Test control message with no barrier expectations."""
        control_msg = ControlMessage(
            message_type=MessageTypes.CONTROL,
            sender_id="worker",
            control_type="checkpoints",
        )
        assert control_msg.expected_barriers == 0


class TestMetricMessage:
    """Test suite for metric messages."""

    def test_metric_message_basic(self):
        """Test basic metric message."""
        metrics = {"loss": 0.5, "accuracy": 0.8}
        metric_msg = MetricMessage(
            message_type=MessageTypes.METRIC,
            sender_id="worker-1",
            recipient_id="controller",
            metrics=metrics,
            step=100,
            epoch=5,
            worker_id="worker-1",
        )
        assert metric_msg.metrics == metrics
        assert metric_msg.step == 100
        assert metric_msg.epoch == 5

    def test_metric_message_no_epoch(self):
        """Test metric message without epoch information."""
        metric_msg = MetricMessage(
            message_type=MessageTypes.METRIC,
            sender_id="worker-2",
            metrics={"loss": 0.6},
            step=200,
        )
        assert metric_msg.epoch is None
        assert "loss" in metric_msg.metrics

    def test_metric_timestamp(self):
        """Test that metric messages have timestamps."""
        start_time = time.time()
        metric_msg = MetricMessage(
            message_type=MessageTypes.METRIC,
            sender_id="worker-3",
            metrics={"loss": 0.7},
            step=300,
        )
        assert metric_msg.timestamp >= start_time
        assert (metric_msg.timestamp - start_time) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])