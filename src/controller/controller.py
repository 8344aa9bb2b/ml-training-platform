"""Controller - main orchestrator component."""

import threading
import time
import signal
from typing import Dict, List, Optional
from collections import defaultdict

from src.network.message_types import Message, MessageTypes, TaskMessage
from src.network.socket_manager import SocketManager
from src.scheduler.task_scheduler import TaskScheduler, Task
from src.monitoring.metrics import PlatformMetrics
from src.monitoring.checkpoint_manager import CheckpointManager
from core.config_loader import load_yaml_config
from core.logger import get_logger
from core.resource_monitor import ResourceMonitor


class Controller:
    """Main orchestrator for the distributed training platform."""

    def __init__(self, config: dict, logger):
        """
        Initialize the controller.

        Args:
            config: Controller configuration dictionary
            logger: Structured logger instance
        """
        self.config = config
        self.logger = logger

        # Component configurations
        self.controller_config = config.get("controller", {})
        self.monitoring_config = config.get("monitoring", {})
        self.distributed_config = config.get("distributed", {})

        # State
        self.running = False
        self.worker_registry: Dict[str, dict] = defaultdict(dict)
        self.task_queue: List[TaskMessage] = []
        self.job_status: Dict[str, dict] = {}
        self.health_check_interval = self.controller_config.get("heartbeat_interval", 10.0)

        # Observability & Metrics
        self.metrics = PlatformMetrics()
        ckpt_dir = self.monitoring_config.get("checkpoint_dir", "./logs/checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir)

        # Resilience & Fault Tolerance
        self.max_task_retries = int(self.controller_config.get("max_task_retries", 3))
        self.task_retry_counts: Dict[str, int] = defaultdict(int)
        self.worker_active_tasks: Dict[str, List[Task]] = defaultdict(list)

        # Initialize task scheduler
        self.scheduler = TaskScheduler(
            max_concurrent_tasks=self.controller_config.get("max_workers", 100),
            scheduling_policy=self.controller_config.get("scheduling_policy", "round_robin"),
            placement_policy=self.controller_config.get("placement_policy", "least_loaded"),
            logger=self.logger
        )

        # Component initialization
        self._init_components()
        self._register_signal_handlers()

    def _init_components(self):
        """Initialize subsystem components."""
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(
            config=self.controller_config,
            logger=self.logger
        )

        # Initialize communication manager
        self.comm_manager = SocketManager(
            host=self.controller_config.get("host", "0.0.0.0"),
            port=self.controller_config.get("port", 8000),
            logger=self.logger
        )

        # Start Prometheus server if enabled
        if self.monitoring_config.get("prometheus_enabled", False):
            prom_port = int(self.controller_config.get("prometheus_port", 9090))
            self.metrics.start_server(port=prom_port)

        self.logger.info("Controller components initialized")

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def serve(self):
        """
        Serve the controller as an ongoing service.
        - Accepts worker registrations
        - Dispatches tasks to workers
        - Collects metrics and status reports
        """
        self.running = True
        self.logger.info(f"Starting controller on {self.comm_manager.host}:{self.comm_manager.port}")

        # Start monitoring resources
        monitoring_thread = threading.Thread(
            target=self._resource_monitor_loop,
            daemon=True
        )
        monitoring_thread.start()

        # Start message listener
        listener_thread = threading.Thread(
            target=self._message_listener,
            daemon=True
        )
        listener_thread.start()

        # Start worker health monitor
        health_thread = threading.Thread(
            target=self._worker_health_monitor,
            daemon=True
        )
        health_thread.start()

        # Start task scheduler (background thread)
        scheduler_thread = threading.Thread(
            target=self._workload_scheduler,
            daemon=True
        )
        scheduler_thread.start()

        # Run main event loop
        try:
            while self.running:
                time.sleep(1)
        finally:
            self.stop()

    def _message_listener(self):
        """Listen for messages from workers and scheduler."""
        self.logger.info("Message listener started")

        while self.running:
            try:
                # Listen for messages (non-blocking)
                messages = self.comm_manager.listen()
                for msg in messages:
                    self.logger.debug(f"Received message: {msg.message_type} from {msg.sender_id}")

                    # Route message to appropriate handler
                    self._route_message(msg)

            except Exception as e:
                self.logger.error(f"Error in message listener: {e}")

    def _route_message(self, message: Message):
        """Route incoming messages to appropriate handlers."""
        handler_map = {
            MessageTypes.HEARTBEAT: self._handle_heartbeat,
            MessageTypes.TASK_STATUS: self._handle_task_status,
            MessageTypes.METRIC: self._handle_metrics,
            MessageTypes.CHECKPOINT: self._handle_checkpoint,
            MessageTypes.WORKER_REGISTER: self._handle_worker_registration,
        }

        handler = handler_map.get(message.message_type)
        if handler:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"Error handling {message.message_type}: {e}")

    def _handle_heartbeat(self, message: Message):
        """Handle worker heartbeat messages."""
        worker_id = message.sender_id
        self.logger.debug(f"Received heartbeat from worker {worker_id}")

        # Update worker registry
        self.worker_registry[worker_id].update({
            "last_seen": time.time(),
            "status": "active"
        })
        self.metrics.set_active_workers(len(self.worker_registry))

    def _handle_task_status(self, message: TaskMessage):
        """Handle task status updates from workers."""
        worker_id = message.sender_id
        payload = message.payload or {}
        task_id = payload.get("task_id", getattr(message, "task_id", "unknown"))
        successful = payload.get("successful", False)

        if successful:
            self.metrics.record_task_completed(str(task_id))
            self.logger.info(f"Task {task_id} completed: {payload}")
        else:
            self.metrics.record_task_failed(str(task_id))
            self.logger.warning(f"Task {task_id} failed: {payload.get('error')}")

        # Remove from active tasks tracking
        if worker_id in self.worker_active_tasks:
            self.worker_active_tasks[worker_id] = [
                t for t in self.worker_active_tasks[worker_id]
                if getattr(t, "task_id", None) != task_id
            ]

    def _handle_metrics(self, message: Message):
        """Handle training metrics from workers."""
        worker_id = message.sender_id
        metrics_data = message.payload or getattr(message, "metrics", {})
        self.metrics.record_training_metrics(worker_id, metrics_data)
        self.logger.debug(f"Received metrics from {worker_id}")

    def _handle_checkpoint(self, message: Message):
        """Handle checkpoint updates from workers."""
        worker_id = message.sender_id
        payload = message.payload or {}
        step = payload.get("step", 0)
        state_dict = payload.get("model_state", {})
        ckpt_metrics = payload.get("metrics", {})
        if state_dict:
            self.checkpoint_manager.save_checkpoint(
                step=step,
                state_dict=state_dict,
                metrics=ckpt_metrics,
                worker_id=worker_id,
            )
        self.logger.info(f"Checkpoint saved by worker {worker_id} at step {step}")

    def _handle_worker_registration(self, message: Message):
        """Handle worker registration messages."""
        worker_id = message.sender_id
        worker_info = dict(message.payload or {})

        if worker_id in self.worker_registry:
            self.logger.info(f"Worker {worker_id} already registered")
        else:
            self.worker_registry[worker_id] = worker_info
            self.scheduler.register_worker(worker_id, worker_info)
            self.logger.info(f"Registered new worker: {worker_id}")

        # Registration counts as proof of life
        self.worker_registry[worker_id]["last_seen"] = time.time()
        self.worker_registry[worker_id]["status"] = "active"
        self.metrics.set_active_workers(len(self.worker_registry))

    def _check_worker_health(self) -> List[str]:
        """
        Run a single health sweep, removing workers that missed heartbeats and
        reassigning orphaned tasks.

        Returns:
            List of worker IDs removed as dead.
        """
        now = time.time()
        cutoff = now - self.health_check_interval

        dead_workers = [
            wid for wid, info in self.worker_registry.items()
            if info.get("last_seen", 0) < cutoff
        ]

        for dead in dead_workers:
            self.metrics.record_dead_worker(dead)
            self.logger.warning(f"Dead worker detected: {dead}")

            # Retrieve active tasks assigned to this worker
            orphaned_tasks = []
            if dead in self.scheduler.active_tasks:
                orphaned_tasks.append(self.scheduler.active_tasks[dead])
            if dead in self.worker_active_tasks:
                for t in self.worker_active_tasks[dead]:
                    if t not in orphaned_tasks:
                        orphaned_tasks.append(t)

            # Reassign or mark permanently failed
            for task in orphaned_tasks:
                tid = getattr(task, "task_id", str(task))
                self.task_retry_counts[tid] += 1
                attempts = self.task_retry_counts[tid]

                if attempts <= self.max_task_retries:
                    self.logger.warning(
                        f"Reassigning task {tid} from dead worker {dead} "
                        f"(attempt {attempts}/{self.max_task_retries})"
                    )
                    self.task_queue.append(task)
                    self.metrics.record_task_reassigned(tid)
                else:
                    self.logger.error(
                        f"Task {tid} exceeded max retries ({self.max_task_retries}) "
                        f"after death of worker {dead}; marking permanently failed"
                    )
                    self.metrics.record_task_failed(tid)

            # Cleanup worker state
            del self.worker_registry[dead]
            self.scheduler.unregister_worker(dead)
            self.worker_active_tasks.pop(dead, None)

        self.metrics.set_active_workers(len(self.worker_registry))
        return dead_workers

    def _worker_health_monitor(self):
        """Periodically check worker health and remove dead workers."""
        self.logger.info("Worker health monitor started")

        while self.running:
            try:
                self._check_worker_health()
                time.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                time.sleep(5)

    def _resource_monitor_loop(self):
        """Periodically monitor resource availability."""
        interval = self.controller_config.get("resource_monitor_interval", 5.0)

        while self.running:
            try:
                self.logger.debug("Running resource check")
                resources = self.resource_monitor.get_resources()
                if resources:
                    self.metrics.record_system_resources(resources)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")

    def _workload_scheduler(self):
        """Background thread for workload scheduling."""
        self.logger.info("Workload scheduler started")

        while self.running:
            try:
                if self.task_queue:
                    # Move tasks from queue to scheduler
                    for task_msg in self.task_queue:
                        self.scheduler.schedule_batch([task_msg])
                    self.task_queue = []
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error in workload scheduler: {e}")

    def run_job(self, job_config: dict):
        """
        Run a single training job.

        Args:
            job_config: Job configuration dictionary
        """
        self.logger.info(f"Starting job: {job_config.get('job', {}).get('name')}")

        try:
            # Job execution logic
            # 1. Validate job configuration
            # 2. Allocate resources
            # 3. Dispatch tasks to workers
            # 4. Monitor job progress
            # 5. Collect results and checkpoints
            # 6. Cleanup

            self.logger.info("Job started successfully")

        except Exception as e:
            self.logger.error(f"Job execution failed: {e}")
            raise
        finally:
            self.logger.info("Job completed")

    def add_task_to_queue(self, task: TaskMessage):
        """Add a task to the scheduling queue."""
        self.task_queue.append(task)
        self.logger.debug(f"Added task {task.task_id} to queue")

    def register_worker(self, worker_id: str, worker_info: dict):
        """Register a new worker with the controller."""
        self.worker_registry[worker_id].update(worker_info)
        self.logger.info(f"Registered worker: {worker_id}")

    def get_available_workers(self) -> List[str]:
        """Get list of currently active workers (seen within the health window)."""
        now = time.time()
        cutoff = now - self.health_check_interval

        available = [
            wid for wid, info in self.worker_registry.items()
            if info.get("last_seen", 0) > cutoff
        ]
        return available

    def stop(self):
        """Stop the controller."""
        self.logger.info("Stopping controller...")
        self.running = False
        if hasattr(self, "metrics") and self.metrics:
            try:
                self.metrics.stop_server()
            except Exception:
                pass
        if hasattr(self, "comm_manager") and self.comm_manager:
            try:
                self.comm_manager.stop()
            except Exception:
                pass

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass