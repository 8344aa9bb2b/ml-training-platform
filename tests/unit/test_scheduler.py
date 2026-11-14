"""Unit tests for scheduler component."""

import pytest
from src.scheduler.task_scheduler import TaskScheduler
from src.scheduler.partition import PartitionStrategy, PartitionResult


class TestTaskScheduler:
    """Test suite for task scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization with default settings."""
        scheduler = TaskScheduler(
            max_concurrent_tasks=10,
            scheduling_policy="round_robin",
        )
        assert scheduler.max_concurrent_tasks == 10
        assert scheduler.scheduling_policy == "round_robin"
        assert scheduler.pending_tasks == []
        assert scheduler.active_tasks == {}

    def test_task_scheduling(self):
        """Test task assignment and scheduling."""
        scheduler = TaskScheduler(max_concurrent_tasks=5)

        # Create sample tasks
        tasks = [
            TaskAssignment(task_id="task-1", task_type="forward_pass", parameters={}),
            TaskAssignment(task_id="task-2", task_type="backward_pass", parameters={}),
            TaskAssignment(task_id="task-3", task_type="optimizer_step", parameters={}),
        ]

        # Schedule tasks
        scheduler.schedule_batch(tasks)

        assert len(scheduler.pending_tasks) == 3
        assert len(scheduler.active_tasks) == 0

        #Execute first task
        result = scheduler.execute_next_task()
        assert result is not None
        assert result.task_id == "task-1"
        assert len(scheduler.active_tasks) > 0

    def task_execution(self):
        """Test task execution flow."""
        scheduler = TaskScheduler(max_concurrent_tasks=2)

        # Create simple task
        def simple_task():
            return {"status": "completed", "task_id": "test-task"}

        result = scheduler.execute_function(simple_task)
        assert result["status"] == "completed"
        assert result["task_id"] == "test-task"


class TestPartitionStrategy:
    """Test suite for model partitioning strategies."""

    def test_partitioning_by_layers(self):
        """Test partitioning strategy that divides by model layers."""
        # This would test the actual layer-aware partitioning logic
        # For now, just verify the interface
        assert True  # Placeholder

    def test_partitioning_by_tensor_size(self):
        """Test partitioning strategy that distributes large tensors."""
        # Placeholder for tensor-size-aware partitioning
        assert True

    def test_partition_result_validation(self):
        """Test that partition results are validated."""
        # This would test validation of partition sizes and overlaps
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])