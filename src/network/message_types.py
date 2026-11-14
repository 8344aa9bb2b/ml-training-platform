"""Message types for inter-component communication."""

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class MessageTypes(str, Enum):
    """Supported message types for distributed communication."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_STATUS = "task_status"
    TENSOR_DATA = "tensor_data"
    CONTROL = "control"
    METRIC = "metric"
    HEARTBEAT = "heartbeat"
    CHECKPOINT = "checkpoint"


@dataclass
class Message:
    """Base message class for inter-component communication."""

    message_type: MessageTypes
    sender_id: str
    recipient_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


@dataclass
class TaskMessage(Message):
    """Message for task assignment and status updates."""

    task_id: Optional[str] = None
    task_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 0


@dataclass
class TensorMessage(Message):
    """Message for transferring model tensors (gradients, activations)."""

    tensor_name: str = ""
    tensor_data: Any = None
    shape: list = field(default_factory=list)
    dtype: str = "float32"
    compression: str = "none"  # none, gzip, lz4


@dataclass
class ControlMessage(Message):
    """Control messages for synchronization and coordination."""

    control_type: str = "sync"  # sync_barrier, reset, stop, etc.
    parameters: Optional[Dict[str, Any]] = None
    expected_barriers: int = 0
    barrier_id: str = ""


@dataclass
class MetricMessage(Message):
    """Message for reporting training metrics."""

    metrics: Dict[str, float] = field(default_factory=dict)
    step: int = 0
    epoch: Optional[int] = None
    worker_id: str = ""


__all__ = [
    "MessageTypes",
    "Message",
    "TaskMessage",
    "TensorMessage",
    "ControlMessage",
    "MetricMessage",
]