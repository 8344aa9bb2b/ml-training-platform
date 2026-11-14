"""Worker node - executes training tasks and reports metrics."""

import time
import numpy as np
import torch
from typing import Dict, List, Optional
import threading

from src.network.message_types import (
    Message,
    MessageTypes,
    TaskMessage,
    TensorMessage,
    ControlMessage,
    MetricMessage,
)
from src.network.socket_manager import SocketManager
from core.logger import get_logger


class Worker:
    """Worker node that executes training tasks."""

    def __init__(self, config: dict, logger):
        """
        Initialize the worker node.

        Args:
            config: Worker configuration dictionary
            logger: Structured logger instance
        """
        self.config = config
        self.logger = logger

        # Component configurations
        self.nodes_config = config.get("nodes", {})
        self.worker_id = self.nodes_config.get("id", "worker-unknown")

        # Component initialization
        self._init_components()

        # State
        self.running = False
        self.current_task = None
        self.health_status = "healthy"
        self.metrics_history = []

        # GPU setup
        self._setup_gpu()

    def _init_components(self):
        """Initialize worker components."""
        # Initialize communication manager
        self.comm_manager = SocketManager(
            host="0.0.0.0",
            port=8001,  # Separate port for workers
            logger=self.logger
        )

        self.logger.info(f"Worker {self.worker_id} components initialized")

    def _setup_gpu(self):
        """Setup GPU device for training."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Available GPUs: {device_count}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.logger.warning("CUDA not available, using CPU")
            self.device = torch.device("cpu")

    def connect_to_controller(self, controller_address: str):
        """
        Connect to the controller.

        Args:
            controller_address: Controller address in format "host:port"
        """
        self.logger.info(f"Connecting to controller at {controller_address}")
        # TODO: Implement connection logic
        # 1. Create connection to controller's message socket
        # 2. Register worker with controller
        # 3. Start listening for task assignments

    def serve_standalone(self):
        """Serve in standalone mode (no controller)."""
        self.logger.info("Worker serving in standalone mode")

        with threading.Lock():
            self.running = True

        try:
            while self.running:
                time.sleep(1)

        finally:
            self.stop()

    def serve(self):
        """Serve as a connected worker."""
        self.logger.info(f"Worker {self.worker_id} serving")

        with threading.Lock():
            self.running = True

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        heartbeat_thread.start()

        # Start message listener
        listener_thread = threading.Thread(
            target=self._message_listener,
            daemon=True
        )
        listener_thread.start()

        # Run main loop
        try:
            while self.running:
                time.sleep(1)
        finally:
            self.stop()

    def _message_listener(self):
        """Listen for messages from controller."""
        self.logger.info("Worker message listener started")

        def message_handler(message: Message):
            """Handle incoming messages."""
            self.logger.debug(f"Worker received: {message.message_type}")

            # Route to appropriate handler
            if message.message_type == MessageTypes.TASK_ASSIGNMENT:
                self._handle_task_assignment(message)
            elif message.message_type == MessageTypes.CONTROL:
                self._handle_control_message(message)
            elif message.message_type == MessageTypes.HEARTBEAT:
                self._handle_heartbeat_response(message)

        # Start listening
        self.comm_manager.start_listener(
            message_handler=message_handler,
            interval_ms=100
        )

    def _handle_task_assignment(self, message: TaskMessage):
        """Handle task assignment from controller."""
        self.current_task = message
        self.logger.info(f"Received task assignment: {message.task_id} ({message.task_type})")

        # Execute the task
        try:
            result = self._execute_task(message)
            self._report_task_status(message.task_id, successful=True, result=result)
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self._report_task_status(message.task_id, successful=False, error=str(e))

    def _execute_task(self, task: TaskMessage) -> Dict:
        """
        Execute a training task.

        Args:
            task: Task message containing task assignment

        Returns:
            Task execution result
        """
        task_type = task.task_type

        if task_type == "forward_pass":
            return self._execute_forward_pass(task.parameters)
        elif task_type == "backward_pass":
            return self._execute_backward_pass(task.parameters)
        elif task_type == "optimizer_step":
            return self._execute_optimizer_step(task.parameters)
        elif task_type == "checkpoints":
            return self._save_checkpoint(task.parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _execute_forward_pass(self, params: dict) -> dict:
        """Execute a forward pass through the model."""
        # TODO: Implement actual model forward pass
        batch_size = params.get("batch_size", 32)
        loss_value = np.random.randn()  # Simulated loss
        accuracy = np.random.rand()

        self.logger.debug(f"Forward pass: batch_size={batch_size}, loss={loss_value:.4f}, accuracy={accuracy:.2%}")

        return {
            "status": "completed",
            "loss": float(loss_value),
            "accuracy": float(accuracy),
            "batch_size": batch_size
        }

    def _execute_backward_pass(self, params: dict) -> dict:
        """Execute backward pass and compute gradients."""
        # TODO: Implement actual backward pass
        loss_value = np.random.randn()
        gradient_norm = np.random.rand()

        self.logger.debug(f"Backward pass: loss={loss_value:.4f}, grad_norm={gradient_norm:.4f}")

        return {
            "status": "completed",
            "loss": float(loss_value),
            "gradient_norm": float(gradient_norm)
        }

    def _execute_optimizer_step(self, params: dict) -> dict:
        """Execute optimizer step."""
        # TODO: Implement actual optimizer step
        learning_rate = params.get("learning_rate", 0.001)
        weight_update = np.random.randn(10)

        self.logger.debug(f"Optimizer step: lr={learning_rate}")

        return {
            "status": "completed",
            "learning_rate": learning_rate,
            "weight_update_shapes": [w.shape for w in [weight_update]]
        }

    def _save_checkpoint(self, params: dict) -> dict:
        """Save model checkpoint."""
        checkpoint_dir = params.get("save_directory", "./logs/checkpoints")

        checkpoint_path = f"{checkpoint_dir}/step_{params.get('step', 'unknown')}.pth"

        self.logger.info(f"Saving checkpoint: {checkpoint_path}")

        # TODO: Implement actual checkpoint saving
        torch.save({"step": params.get("step")}, checkpoint_path)

        return {
            "status": "completed",
            "checkpoint_path": checkpoint_path
        }

    def _handle_control_message(self, message: ControlMessage):
        """Handle control messages."""
        control_type = message.control_type

        if control_type == "sync_barrier":
            self.logger.debug("Executing synchronization barrier")
            # TODO: Wait for all workers to reach barrier
        elif control_type == "reset":
            self.logger.warning("Received reset command")
            # TODO: Implement reset logic
        elif control_type == "stop":
            self.logger.info("Received stop command")
            self.running = False
        else:
            self.logger.warning(f"Unknown control type: {control_type}")

    def _handle_heartbeat_response(self, message: Message):
        """Handle controller heartbeat response."""
        self.logger.debug("Heartbeat confirmed")

    def _heartbeat_loop(self):
        """Periodically send heartbeat to controller."""
        heartbeat_interval = self.nodes_config.get(
            "health_check", {}
        ).get("interval", 5.0)

        while self.running:
            try:
                heartbeat = Message(
                    message_type=MessageTypes.HEARTBEAT,
                    sender_id=self.worker_id,
                    timestamp=time.time()
                )

                self.comm_manager.send_message(heartbeat)
                self.logger.debug("Heartbeat sent")
                time.sleep(heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                break

    def _report_task_status(self, task_id: str, successful: bool, result: dict = None, error: str = None):
        """
        Report task status to controller.

        Args:
            task_id: ID of the completed task
            successful: Whether the task completed successfully
            result: Task execution result
            error: Error message if task failed
        """
        status_msg = Message(
            message_type=MessageTypes.TASK_STATUS,
            sender_id=self.worker_id,
            payload={
                "task_id": task_id,
                "successful": successful,
                "result": result,
                "error": error,
            }
        )

        self.comm_manager.send_message(status_msg)

    def send_metrics(self, metrics: dict, step: int, epoch: Optional[int] = None):
        """
        Send training metrics to controller.

        Args:
            metrics: Dictionary of metrics {name: value}
            step: Current training step
            epoch: Current epoch (if applicable)
        """
        metric_msg = MetricMessage(
            message_type=MessageTypes.METRIC,
            sender_id=self.worker_id,
            metrics=metrics,
            step=step,
            epoch=epoch,
        )

        self.comm_manager.send_message(metric_msg)

    def stop(self):
        """Stop the worker node."""
        self.logger.info("Worker stopping...")
        self.running = False

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except Exception:
            pass