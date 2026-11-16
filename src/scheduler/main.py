"""Main entry point for the scheduler component."""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scheduler.task_scheduler import TaskScheduler
from src.scheduler.partition import DataParallelPartition, PipelineParallelPartition
from core.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed ML Training Platform - Scheduler")

    parser.add_argument("--config", type=str, default="config/scheduler.yaml",
                        help="Path to scheduler configuration file")
    parser.add_argument("--mode", type=str, choices=["partition", "test"],
                        default="partition", help="Operation mode: partition jobs or run tests")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        scheduler_config = config.get("scheduler", {})
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Setup logging
    logger = get_logger(
        name="scheduler",
        config=config.get("logging", {})
    )

    logger.info(f"Starting scheduler in {args.mode} mode")

    if args.mode == "partition":
        # Run partitioning tests
        logger.info("Running partitioning tests")
        run_partition_tests(scheduler_config, logger)
    elif args.mode == "test":
        # Run scheduler tests
        logger.info("Running scheduler tests")
        run_scheduler_tests(scheduler_config, logger)


def run_partition_tests(config: dict, logger):
    """Run partitioning strategy tests."""
    logger.info("Testing partitioning strategies")

    # Test data parallel partitioning
    dp_partition = DataParallelPartition()
    result = dp_partition.partition(total=100, parts=4, mode="contiguous")

    logger.info(f"Data parallel (contiguous): {result.num_partitions} partitions, sizes: {result.sizes()}")
    assert len(result.partitions) == 4
    assert sum(result.sizes()) == 100

    result = dp_partition.partition(total=100, parts=4, mode="strided")
    logger.info(f"Data parallel (strided): {result.num_partitions} partitions, sizes: {result.sizes()}")
    assert len(result.partitions) == 4

    # Test pipeline parallel partitioning
    pp_partition = PipelineParallelPartition()
    result = pp_partition.partition(total=10, parts=3)

    logger.info(f"Pipeline parallel: {result.num_partitions} partitions, sizes: {result.sizes()}")
    assert len(result.partitions) == 3
    assert sum(result.sizes()) == 10

    # Test invalid partition
    try:
        pp_partition.partition(total=5, parts=10)
        logger.error("Invalid partition should have raised an error")
        sys.exit(1)
    except Exception as e:
        logger.info(f"Correctly rejected invalid partition: {type(e).__name__}")

    logger.info("All partitioning tests passed!")


def run_scheduler_tests(config: dict, logger):
    """Run scheduler tests."""
    logger.info("Testing task scheduler")

    # Test initialization
    scheduler = TaskScheduler(
        max_concurrent_tasks=5,
        scheduling_policy="round_robin",
        placement_policy="least_loaded",
    )

    logger.info(f"Scheduler initialized: {scheduler.get_stats()}")

    # Test task scheduling
    tasks = [
        Task(task_id="task-1", task_type="forward_pass", parameters={}),
        Task(task_id="task-2", task_type="backward_pass", parameters={}),
        Task(task_id="task-3", task_type="optimizer_step", parameters={}),
    ]

    scheduler.schedule_batch(tasks)
    logger.info(f"Added {len(tasks)} tasks to queue")

    # Execute tasks
    for i in range(3):
        result = scheduler.execute_next_task()
        if result:
            logger.info(f"Task {result.task_id} executed: {'success' if result.successful else 'failed'}")

    # Test execution function
    def simple_task():
        return {"status": "completed", "task_id": "test-task"}

    result = scheduler.execute_function(simple_task)
    logger.info(f"Function task result: {result}")

    logger.info("All scheduler tests passed!")


if __name__ == "__main__":
    main()
