"""Central Parameter Server implementation for distributed training."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Set, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.model.gradient_sync import SyncMode, reduce_gradient_dict
from src.model.tensor_utils import apply_gradients


class ParameterServer:
    """Central parameter server maintaining authoritative model weights and optimizer.

    Supports both Synchronous SGD (accumulate gradients from all active workers before stepping)
    and Asynchronous SGD (immediate parameter updates with staleness mitigation).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        sync_mode: SyncMode = SyncMode.SYNCHRONOUS,
        min_workers: int = 1,
        staleness_bound: int = 10,
    ) -> None:
        """Initialize parameter server.

        Args:
            model: PyTorch model holding master parameters.
            optimizer: Optional optimizer updating master parameters.
            sync_mode: SYNCHRONOUS or ASYNCHRONOUS update policy.
            min_workers: Minimum worker gradient submissions required per synchronous step.
            staleness_bound: Maximum step lag tolerated in asynchronous updates before discarding.
        """
        self.model = model
        self.optimizer = optimizer
        self.sync_mode = sync_mode
        self.min_workers = max(1, min_workers)
        self.staleness_bound = staleness_bound

        self.current_step: int = 0
        self._lock = threading.Lock()
        self._registered_workers: Set[str] = set()

        # Step -> worker_id -> gradient_dict
        self._sync_buffer: Dict[str, Dict[str, torch.Tensor]] = {}

    @property
    def registered_workers(self) -> Set[str]:
        with self._lock:
            return set(self._registered_workers)

    def register_worker(self, worker_id: str) -> None:
        """Register a worker with the parameter server."""
        with self._lock:
            self._registered_workers.add(worker_id)

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        with self._lock:
            self._registered_workers.discard(worker_id)
            self._sync_buffer.pop(worker_id, None)

    def pull(self, worker_id: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], int]:
        """Pull the latest model state dictionary and current step count.

        Args:
            worker_id: Optional worker requesting parameters.

        Returns:
            Tuple of (cloned_state_dict, current_step).
        """
        with self._lock:
            state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            return state, self.current_step

    def push(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> bool:
        """Push worker gradients to the parameter server.

        Args:
            worker_id: Identifier of pushing worker.
            gradients: Gradients computed by the worker.
            step: The model step at which the worker computed these gradients.

        Returns:
            True if an optimizer step was applied, False otherwise.
        """
        with self._lock:
            if self.sync_mode == SyncMode.SYNCHRONOUS:
                return self._push_synchronous(worker_id, gradients, step)
            else:
                return self._push_asynchronous(worker_id, gradients, step)

    def _push_synchronous(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> bool:
        """Handle synchronous gradient accumulation."""
        # Buffer gradient from worker
        self._sync_buffer[worker_id] = {k: v.detach().cpu().clone() for k, v in gradients.items()}

        required_count = max(self.min_workers, len(self._registered_workers))
        if len(self._sync_buffer) >= required_count:
            # Aggregate and apply
            all_grads = list(self._sync_buffer.values())
            avg_grads = reduce_gradient_dict(all_grads, op="avg")

            apply_gradients(self.model, avg_grads)
            if self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self._sync_buffer.clear()
            self.current_step += 1
            return True

        return False

    def _push_asynchronous(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> bool:
        """Handle asynchronous parameter update with staleness damping."""
        staleness = self.current_step - step
        if staleness < 0:
            staleness = 0

        # Discard updates that exceed the staleness bound
        if staleness > self.staleness_bound:
            return False

        # Apply staleness damping factor: 1.0 / (1.0 + staleness)
        factor = 1.0 / (1.0 + float(staleness))
        damped_grads = {
            k: (v.detach().cpu() * factor) for k, v in gradients.items()
        }

        apply_gradients(self.model, damped_grads)
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.current_step += 1
        return True

    def save_checkpoint(self, path: str) -> None:
        """Persist parameter server state to a file checkpoint."""
        with self._lock:
            payload = {
                "step": self.current_step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "sync_mode": self.sync_mode.value,
            }
            torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        """Restore parameter server state from a checkpoint file."""
        with self._lock:
            checkpoint = torch.load(path, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state"])
            if self.optimizer is not None and checkpoint.get("optimizer_state") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.current_step = checkpoint.get("step", 0)
