"""Gradient synchronization primitives and AllReduce synchronizer."""

from __future__ import annotations

from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set
import torch


class SyncMode(str, Enum):
    """Synchronization mode across training workers."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"


class SyncStrategy(str, Enum):
    """Distributed gradient communication strategy."""
    ALL_REDUCE = "all_reduce"
    PARAMETER_SERVER = "parameter_server"


def reduce_gradient_dict(
    grads_list: Sequence[Dict[str, torch.Tensor]],
    op: str = "avg",
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Reduce a list of gradient dictionaries across workers.

    Args:
        grads_list: Non-empty sequence of gradient dictionaries mapping parameter names to tensors.
        op: Reduction operation, either 'avg' or 'sum'.
        weights: Optional sequence of weights (e.g. per-worker batch sizes) for weighted averaging.

    Returns:
        A dictionary of reduced gradient tensors.

    Raises:
        ValueError: If grads_list is empty, keys mismatch, or shapes mismatch.
    """
    if not grads_list:
        raise ValueError("Cannot reduce empty gradient list")

    if op not in ("avg", "sum"):
        raise ValueError(f"Unsupported reduction op: {op}. Expected 'avg' or 'sum'")

    reference = grads_list[0]
    keys = set(reference.keys())

    # Validate keys across all workers
    for i, g in enumerate(grads_list[1:], start=1):
        if set(g.keys()) != keys:
            raise ValueError(f"Worker {i} gradient keys mismatch reference keys")

    # Validate shapes and compute reduction
    if weights is not None:
        if len(weights) != len(grads_list):
            raise ValueError(f"Weights length ({len(weights)}) does not match grads ({len(grads_list)})")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")
        normalized_weights = [w / total_weight for w in weights]
    else:
        normalized_weights = [1.0 / len(grads_list)] * len(grads_list)

    reduced: Dict[str, torch.Tensor] = {}
    for key in keys:
        ref_tensor = reference[key]
        for i, g in enumerate(grads_list[1:], start=1):
            if g[key].shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{key}': "
                    f"expected {ref_tensor.shape}, got {g[key].shape} from worker {i}"
                )

        if op == "sum":
            acc = sum(g[key].to(dtype=ref_tensor.dtype) for g in grads_list)
            reduced[key] = acc
        else:  # avg
            acc = sum(
                g[key].to(dtype=torch.float32) * weight
                for g, weight in zip(grads_list, normalized_weights)
            )
            reduced[key] = acc.to(dtype=ref_tensor.dtype)

    return reduced


class AllReduceSynchronizer:
    """Synchronizes gradients across workers using collective AllReduce semantics.

    Supports synchronous barrier execution where all workers submit gradients for a
    training step and block until the averaged gradient tensor is available.
    """

    def __init__(self, expected_workers: Optional[Sequence[str]] = None) -> None:
        """Initialize AllReduce synchronizer.

        Args:
            expected_workers: Optional sequence of initial worker identifiers.
        """
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._registered_workers: Set[str] = set(expected_workers) if expected_workers else set()
        self._worker_weights: Dict[str, float] = {w: 1.0 for w in self._registered_workers}

        # Step -> worker_id -> gradients
        self._pending: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        # Step -> reduced gradients
        self._reduced: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def registered_workers(self) -> Set[str]:
        with self._lock:
            return set(self._registered_workers)

    def register_worker(self, worker_id: str, weight: float = 1.0) -> None:
        """Register a worker participating in AllReduce."""
        with self._lock:
            self._registered_workers.add(worker_id)
            self._worker_weights[worker_id] = float(weight)

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        with self._lock:
            self._registered_workers.discard(worker_id)
            self._worker_weights.pop(worker_id, None)

    def submit_gradient(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Submit worker gradients without blocking.

        Returns:
            Reduced gradients dictionary if this submission completed the barrier,
            otherwise None.
        """
        with self._cv:
            if worker_id not in self._registered_workers:
                self._registered_workers.add(worker_id)
                self._worker_weights[worker_id] = 1.0

            if step not in self._pending:
                self._pending[step] = {}

            if worker_id in self._pending[step]:
                raise ValueError(f"Worker '{worker_id}' already submitted gradients for step {step}")

            # Store gradient submission (cloned to CPU for safety)
            self._pending[step][worker_id] = {k: v.detach().cpu().clone() for k, v in gradients.items()}

            # Check if all registered workers have submitted for this step
            expected_count = len(self._registered_workers)
            if len(self._pending[step]) >= expected_count:
                worker_ids = list(self._pending[step].keys())
                grads_list = [self._pending[step][w] for w in worker_ids]
                weights = [self._worker_weights.get(w, 1.0) for w in worker_ids]

                reduced = reduce_gradient_dict(grads_list, op="avg", weights=weights)
                self._reduced[step] = reduced
                self._cv.notify_all()
                return {k: v.clone() for k, v in reduced.items()}

            return None

    def submit_and_sync(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
        timeout: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Submit local gradients for a step and wait for all workers to complete AllReduce.

        Args:
            worker_id: Identifier of the worker submitting gradients.
            gradients: Gradient dictionary for the current step.
            step: Training step number.
            timeout: Maximum wait time in seconds (or None to block indefinitely).

        Returns:
            Dictionary of reduced/averaged gradients for the step.

        Raises:
            TimeoutError: If the barrier times out before all workers submit.
            ValueError: If worker is not registered or submits duplicate gradients for step.
        """
        start_time = time.time()

        with self._cv:
            result = self.submit_gradient(worker_id, gradients, step)
            if result is not None:
                return result

            # Wait for completion
            while step not in self._reduced:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise TimeoutError(f"AllReduce timed out for step {step}")
                    signaled = self._cv.wait(timeout=remaining)
                    if not signaled and step not in self._reduced:
                        raise TimeoutError(f"AllReduce timed out for step {step}")
                else:
                    self._cv.wait()

            return {k: v.clone() for k, v in self._reduced[step].items()}

    def clear_step(self, step: int) -> None:
        """Purge stored gradients for completed steps to prevent memory growth."""
        with self._lock:
            self._pending.pop(step, None)
            self._reduced.pop(step, None)
