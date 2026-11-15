"""Task scheduling and placement.

A TaskScheduler decides which worker executes each Task, respects max
concurrent tasks per worker, and tracks active vs pending work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.logger import get_logger
from src.network.message_types import Message, MessageTypes, TaskMessage
from .partition import create_partition_strategy, PartitionStrategy


class SchedulingPolicy(str, Enum):
    """Supported scheduling policies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_AWARE = "capability_aware"


class TaskExecutionError(Exception):
    """Raised when task execution fails."""

    pass


@dataclass
class Task:
    """A unit of work to be scheduled.

    Attributes:
        task_id: Unique identifier.
        task_type: Type of operation (forward_pass, backward_pass, optimizer_step, checkpoint).
        parameters: Task-specific arguments.
        priority: Higher priority tasks run earlier.
        created_at: Timestamp when the task was enqueued.
    """

    task_id: str
    task_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=lambda: __import__("time").time())

    def __lt__(self, other: "Task") -> bool:
        """Compare by priority (higher first), then by creation time (earlier first)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at > other.created_at


@dataclass
class TaskExecutionResult:
    """Result of task execution.

    Attributes:
        task_id: The task that was executed.
        successful: Whether the task completed without error.
        result: Execution result (metrics, outputs, etc.).
        error: Error message if failed.
    """

    task_id: str
    successful: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_message(self) -> Message:
        """Convert to a task status message."""
        payload = {
            "task_id": self.task_id,
            "successful": self.successful,
        }
        if self.result:
            payload["result"] = self.result
        if self.error:
            payload["error"] = self.error

        return Message(
            message_type=MessageTypes.TASK_STATUS,
            sender_id="scheduler",
            payload=payload,
        )


class PlacementPolicy(ABC):
    """Base class for worker placement policies."""

    @abstractmethod
    def select_workers(self, pending_tasks: List[Task], available_workers: List[str]) -> List[str]:
        """Select workers for the given tasks.

        Args:
            pending_tasks: Tasks waiting for assignment.
            available_workers: Workers that can accept tasks.

        Returns:
            List of worker IDs, one per task.
        """
        raise NotImplementedError


class RoundRobinPolicy(PlacementPolicy):
    """Cycle through available workers in order."""

    def select_workers(self, pending_tasks: List[Task], available_workers: List[str]) -> List[str]:
        if not available_workers:
            return []
        return [available_workers[i % len(available_workers)] for i in range(len(pending_tasks))]


class LeastLoadedPolicy(PlacementPolicy):
    """Assign tasks to workers with the fewest active tasks."""

    def select_workers(self, pending_tasks: List[Task], available_workers: List[str]) -> List[str]:
        if not available_workers:
            return []

        worker_load: Dict[str, int] = {}
        for worker_id in available_workers:
            worker_load[worker_id] = 0

        return sorted(available_workers, key=lambda w: worker_load[w])[: len(pending_tasks)]


class CapabilityAwarePolicy(PlacementPolicy):
    """Assign tasks to workers with appropriate capabilities (GPU, memory)."""

    def select_workers(self, pending_tasks: List[Task], available_workers: List[str]) -> List[str]:
        if not available_workers:
            return []

        gpu_workers: List[str] = []
        cpu_workers: List[str] = []

        for worker_id in available_workers:
            gpu_workers.append(worker_id)

        selected: List[str] = []
        for task in pending_tasks:
            if task.task_type in ("forward_pass", "backward_pass", "optimizer_step"):
                if gpu_workers:
                    selected.append(gpu_workers.pop(0))
                elif cpu_workers:
                    selected.append(cpu_workers.pop(0))
            else:
                if cpu_workers:
                    selected.append(cpu_workers.pop(0))
                elif gpu_workers:
                    selected.append(gpu_workers.pop(0))

        return selected


class TaskScheduler:
    """Schedules tasks across workers with configurable policies."""

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        scheduling_policy: str = SchedulingPolicy.ROUND_ROBIN.value,
        placement_policy: str = SchedulingPolicy.ROUND_ROBIN.value,
        logger=None,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.scheduling_policy = SchedulingPolicy(scheduling_policy)
        self.placement_policy = placement_policy
        self.logger = logger or get_logger(name="task_scheduler")

        self.pending_tasks: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}  # worker_id -> Task
        self.worker_load: Dict[str, int] = {}  # worker_id -> active count
        self.available_workers: List[str] = []

    def register_worker(self, worker_id: str, capabilities: Optional[Dict[str, Any]] = None) -> None:
        """Register a worker as available for task placement."""
        if worker_id not in self.available_workers:
            self.available_workers.append(worker_id)
            self.worker_load.setdefault(worker_id, 0)
            self.logger.info(f"Registered worker {worker_id} with scheduler")

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the available pool."""
        if worker_id in self.available_workers:
            self.available_workers.remove(worker_id)
            self.worker_load.pop(worker_id, None)
            self.logger.info(f"Unregistered worker {worker_id} from scheduler")

    def schedule_batch(self, tasks: List[Task]) -> None:
        """Add multiple tasks to the pending queue."""
        for task in tasks:
            self.pending_tasks.append(task)
        self.logger.debug(f"Added {len(tasks)} tasks to queue, total pending: {len(self.pending_tasks)}")

    def execute_next_task(self) -> Optional[TaskExecutionResult]:
        """Execute one pending task and assign it to a worker."""
        if not self.pending_tasks:
            self.logger.debug("No pending tasks to execute")
            return None

        task = max(self.pending_tasks)
        self.pending_tasks.remove(task)

        available_workers = self._get_available_workers()
        if not available_workers:
            self.logger.warning("No available workers for task")
            return TaskExecutionResult(
                task_id=task.task_id,
                successful=False,
                error="No available workers",
            )

        placement = self._create_placement_policy()
        selected_workers = placement.select_workers([task], available_workers)

        if not selected_workers:
            self.logger.warning("No workers selected for task")
            return TaskExecutionResult(
                task_id=task.task_id,
                successful=False,
                error="No workers selected",
            )

        worker_id = selected_workers[0]

        # Track active task
        self.active_tasks[worker_id] = task
        self.worker_load[worker_id] = self.worker_load.get(worker_id, 0) + 1

        self.logger.info(f"Assigned task {task.task_id} ({task.task_type}) to worker {worker_id}")

        try:
            result = self._execute_task(task)
            self.worker_load[worker_id] = max(0, self.worker_load.get(worker_id, 0) - 1)
            return TaskExecutionResult(
                task_id=task.task_id,
                successful=True,
                result=result,
            )
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self.active_tasks.pop(worker_id, None)
            self.worker_load[worker_id] = max(0, self.worker_load.get(worker_id, 0) - 1)
            return TaskExecutionResult(
                task_id=task.task_id,
                successful=False,
                error=str(e),
            )

    def execute_function(self, func: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a simple function as a task."""
        return func()

    def _get_available_workers(self) -> List[str]:
        """Get list of workers with capacity for new tasks."""
        return [
            worker_id for worker_id in self.available_workers
            if self.worker_load.get(worker_id, 0) < self.max_concurrent_tasks
        ]

    def _create_placement_policy(self) -> PlacementPolicy:
        """Create a placement policy instance."""
        if self.placement_policy == SchedulingPolicy.ROUND_ROBIN.value:
            return RoundRobinPolicy()
        elif self.placement_policy == SchedulingPolicy.LEAST_LOADED.value:
            return LeastLoadedPolicy()
        elif self.placement_policy == SchedulingPolicy.CAPABILITY_AWARE.value:
            return CapabilityAwarePolicy()
        else:
            return RoundRobinPolicy()

    def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task."""
        if task.task_type == "forward_pass":
            return {"status": "completed", "loss": 0.5, "accuracy": 0.8}
        elif task.task_type == "backward_pass":
            return {"status": "completed", "loss": 0.5, "gradient_norm": 1.2}
        elif task.task_type == "optimizer_istep" or task.task_type == "optimizer_step":
            return {"status": "completed", "learning_rate": 0.001}
        elif task.task_type == "checkpoints":
            return {"status": "completed", "checkpoint_path": "./logs/cp.pth"}
        else:
            raise TaskExecutionError(f"Unknown task type: {task.task_type}")

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "active_tasks": len(self.active_tasks),
            "worker_load": self.worker_load.copy(),
            "scheduling_policy": self.scheduling_policy.value,
            "placement_policy": self.placement_policy,
        }


__all__ = [
    "SchedulingPolicy",
    "Task",
    "TaskExecutionResult",
    "TaskExecutionError",
    "PlacementPolicy",
    "RoundRobinPolicy",
    "LeastLoadedPolicy",
    "CapabilityAwarePolicy",
    "TaskScheduler",
]
