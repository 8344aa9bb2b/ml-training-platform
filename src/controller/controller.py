"""Controller - main orchestrator component."""

import threading
import time
from typing import Dict, List, Optional
from collections import defaultdict

from src.network.message_types import Message, MessageTypes, TaskMessage
from src.network.socket_manager import SocketManager
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
        self.task_queue = []
        self.job_status = {}

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

        # Start workload scheduler (background thread)
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

    def _handle_task_status(self, message: TaskMessage):
        """Handle task status updates from workers."""
        worker_id = message.sender_id
        self.logger.info(f"Task {message.task_id} completed: {message.payload}")

    def _handle_metrics(self, message: Message):
        """Handle training metrics from workers."""
        self.logger.debug(f"Received metrics from {message.sender_id}")

    def _handle_checkpoint(self, message: Message):
        """Handle checkpoint updates from workers."""
        worker_id = message.sender_id
        self.logger.info(f"Checkpoint saved by worker {worker_id}")

    def _resource_monitor_loop(self):
        """Periodically monitor resource availability."""
        interval = self.controller_config.get("resource_monitor_interval", 5.0)

        while self.running:
            try:
                self.logger.debug("Running resource check")
                # Update resource monitoring
                # In production, this would touch actual hardware APIs
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")

    def _workload_scheduler(self):
        """Background thread for workload scheduling."""
        self.logger.info("Workload scheduler started")

        while self.running:
            try:
                if self.task_queue and self.worker_registry:
                    # TODO: Implement actual scheduling logic
                    # For now, just log startup
                    pass
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
        """Get list of currently active workers."""
        now = time.time()
        cutoff = now - 30  # Workers seen in last 30 seconds

        available = [
            wid for wid, info in self.worker_registry.items()
            if info.get("last_seen", 0) > cutoff
        ]
        return available

    def stop(self):
        """Stop the controller."""
        self.logger.info("Stopping controller...")
        self.running = False

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass