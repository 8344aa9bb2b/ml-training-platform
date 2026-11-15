"""Unit tests for scheduler component."""

import pytest
import time
from unittest.mock import Mock
from src.scheduler.task_scheduler import (
    TaskScheduler,
    Task,
    TaskExecutionResult,
    SchedulingPolicy,
    PlacementPolicy,
    RoundRobinPolicy,
    LeastLoadedPolicy,
    CapabilityAwarePolicy,
    TaskExecutionError,
)
from src.scheduler.partition import (
    Partition,
    PartitionResult,
    DataParallelPartition,
    PipelineParallelPartition,
    create_partition_strategy,
    PartitionError,
)


class TestTask:
    """Test suite for Task dataclass."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="task-1",
            task_type="forward_pass",
            parameters={"batch_size": 32},
            priority=1,
        )
        assert task.task_id == "task-1"
        assert task.task_type == "forward_pass"
        assert task.priority == 1
        assert task.created_at > 0

    def test_task_default_priority(self):
        """Test that priority defaults to 0."""
        task = Task(
            task_id="task-2",
            task_type="backward_pass",
        )
        assert task.priority == 0

    def test_task_lt_comparison(self):
        """Test task ordering by priority, then creation time."""
        t1 = Task("t1", "forward_pass", priority=1, created_at=100.0)
        t2 = Task("t2", "forward_pass", priority=1, created_at=101.0)
        t3 = Task("t3", "forward_pass", priority=2, created_at=100.0)

        # t1 created earlier than t2 (so t1 > t2 in __lt__ when priority equal)
        # t3 has higher priority than t1 (t3 > t1)
        assert t2 < t1  # t2 created later -> smaller in max-heap ordering
        assert t1 < t3  # t3 higher priority -> larger in max-heap ordering


class TestTaskExecutionResult:
    """Test suite for TaskExecutionResult."""

    def test_successful_result(self):
        """Test successful task execution result."""
        result = TaskExecutionResult(
            task_id="task-1",
            successful=True,
            result={"loss": 0.5, "accuracy": 0.8},
        )
        assert result.task_id == "task-1"
        assert result.successful
        assert result.result == {"loss": 0.5, "accuracy": 0.8}
        assert result.error is None

    def test_failed_result(self):
        """Test failed task execution result."""
        result = TaskExecutionResult(
            task_id="task-2",
            successful=False,
            error="Task failed",
        )
        assert result.task_id == "task-2"
        assert not result.successful
        assert result.error == "Task failed"
        assert result.result is None

    def test_to_message(self):
        """Test conversion to message."""
        result = TaskExecutionResult(
            task_id="task-3",
            successful=True,
            result={"loss": 0.5},
        )
        msg = result.to_message()
        assert msg.message_type == "task_status"
        assert msg.payload["task_id"] == "task-3"
        assert msg.payload["successful"] is True
        assert msg.payload["result"] == {"loss": 0.5}


class TestPlacementPolicy:
    """Test suite for placement policies."""

    def test_round_robin_policy(self):
        """Test round-robin worker selection."""
        policy = RoundRobinPolicy()
        workers = ["worker-1", "worker-2", "worker-3"]
        tasks = [Task(f"task-{i}", "forward_pass") for i in range(5)]

        selected = policy.select_workers(tasks, workers)
        assert len(selected) == 5
        assert selected == ["worker-1", "worker-2", "worker-3", "worker-1", "worker-2"]

    def test_least_loaded_policy(self):
        """Test least-loaded worker selection."""
        policy = LeastLoadedPolicy()
        workers = ["worker-1", "worker-2"]
        tasks = [Task(f"task-{i}", "forward_pass") for i in range(2)]

        selected = policy.select_workers(tasks, workers)
        assert len(selected) == 2
        assert set(selected).issubset(set(workers))

    def test_capability_aware_policy(self):
        """Test capability-aware worker selection."""
        policy = CapabilityAwarePolicy()
        workers = ["worker-1", "worker-2", "worker-3"]
        tasks = [
            Task("task-1", "forward_pass"),
            Task("task-2", "backward_pass"),
            Task("task-3", "optimizer_step"),
        ]

        selected = policy.select_workers(tasks, workers)
        assert len(selected) == 3


class TestTaskScheduler:
    """Test suite for task scheduler."""

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
    def scheduler(self, mock_logger):
        """Create a scheduler instance."""
        scheduler = TaskScheduler(
            max_concurrent_tasks=5,
            scheduling_policy="round_robin",
            placement_policy="least_loaded",
            logger=mock_logger,
        )
        return scheduler

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization with default settings."""
        assert scheduler.max_concurrent_tasks == 5
        assert scheduler.scheduling_policy == SchedulingPolicy.ROUND_ROBIN
        assert scheduler.placement_policy == "least_loaded"
        assert scheduler.pending_tasks == []
        assert scheduler.active_tasks == {}
        assert scheduler.worker_load == {}

    def test_task_scheduling(self, scheduler):
        """Test task assignment and scheduling."""
        # Register a worker
        scheduler.register_worker("worker-1", {"gpu": True})

        tasks = [
            Task("task-1", "forward_pass"),
            Task("task-2", "backward_pass"),
            Task("task-3", "optimizer_step"),
        ]

        scheduler.schedule_batch(tasks)

        assert len(scheduler.pending_tasks) == 3
        assert len(scheduler.active_tasks) == 0

        # Execute first task
        result = scheduler.execute_next_task()
        assert result is not None
        assert result.task_id == "task-1"
        assert result.successful
        assert len(scheduler.active_tasks) > 0

    def test_execute_function(self, scheduler):
        """Test task execution flow with function."""
        def simple_task():
            return {"status": "completed", "task_id": "test-task"}

        result = scheduler.execute_function(simple_task)
        assert result["status"] == "completed"
        assert result["task_id"] == "test-task"

    def test_get_stats(self, scheduler):
        """Test scheduler statistics."""
        stats = scheduler.get_stats()
        assert "pending_tasks" in stats
        assert "active_tasks" in stats
        assert "worker_load" in stats
        assert "scheduling_policy" in stats


class TestPartitionStrategy:
    """Test suite for model partitioning strategies."""

    def test_data_parallel_contiguous(self):
        """Test data parallel partitioning with contiguous mode."""
        partition = DataParallelPartition()
        result = partition.partition(total=100, parts=4, mode="contiguous")

        assert result.num_partitions == 4
        assert sum(result.sizes()) == 100
        assert len(result.partitions) == 4

        # Check that partitions are contiguous
        for i, partition_result in enumerate(result.partitions):
            if i < len(result.partitions) - 1:
                assert partition_result.items[-1] < result.partitions[i + 1].items[0]

    def test_data_parallel_strided(self):
        """Test data parallel partitioning with strided mode."""
        partition = DataParallelPartition()
        result = partition.partition(total=100, parts=4, mode="strided")

        assert result.num_partitions == 4
        assert sum(result.sizes()) == 100

        # Check that all items are present
        all_items = set()
        for p in result.partitions:
            all_items.update(p.items)

        assert len(all_items) == 100

    def test_pipeline_parallel(self):
        """Test pipeline parallel partitioning."""
        partition = PipelineParallelPartition()
        result = partition.partition(total=10, parts=3)

        assert result.num_partitions == 3
        assert sum(result.sizes()) == 10
        assert len(result.partitions) == 3

        # Check that all layers 0-9 are covered exactly once
        all_layers = []
        for p in result.partitions:
            all_layers.extend(p.items)
        assert sorted(all_layers) == list(range(10))

    def test_partition_result_validation(self):
        """Test that partition results are validated."""
        # Valid partition
        result = PartitionResult(strategy="test", partitions=[
            Partition(index=0, items=[0, 1, 2]),
            Partition(index=1, items=[3, 4, 5]),
        ])
        result.validate()  # Should not raise

        # Invalid partition (duplicate items)
        result = PartitionResult(strategy="test", partitions=[
            Partition(index=0, items=[0, 1, 2]),
            Partition(index=1, items=[2, 3, 4]),  # Duplicate 2
        ])
        with pytest.raises(PartitionError):
            result.validate()

    def test_invalid_partition_inputs(self):
        """Test that invalid partition inputs are rejected."""
        partition = DataParallelPartition()

        # Negative total
        with pytest.raises(PartitionError):
            partition.partition(total=-10, parts=2)

        # Negative parts
        with pytest.raises(PartitionError):
            partition.partition(total=10, parts=-2)

        # More parts than total (pipeline parallel)
        with pytest.raises(PartitionError):
            partition.partition(total=5, parts=10)

        # Unknown mode
        with pytest.raises(PartitionError):
            partition.partition(total=100, parts=4, mode="unknown")


class TestCreatePartitionStrategy:
    """Test suite for partition strategy factory."""

    def test_create_data_parallel(self):
        """Test creating data parallel partition."""
        strategy = create_partition_strategy("data_parallel")
        assert isinstance(strategy, DataParallelPartition)

    def test_create_pipeline_parallel(self):
        """Test creating pipeline parallel partition."""
        strategy = create_partition_strategy("pipeline_parallel")
        assert isinstance(strategy, PipelineParallelPartition)

    def test_unknown_strategy(self):
        """Test error for unknown strategy name."""
        with pytest.raises(PartitionError):
            create_partition_strategy("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
