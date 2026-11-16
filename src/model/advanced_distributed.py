"""Advanced distributed training primitives: gradient compression, distributed optimizer, fault-tolerant AllReduce, and elastic scaling."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.model.gradient_sync import reduce_gradient_dict
from src.model.tensor_utils import apply_gradients, extract_gradients
from src.scheduler.partition import DataParallelPartition


class TopKCompressor:
    """Sparsifies gradients by transmitting only top-k absolute values."""

    @staticmethod
    def compress(gradient: torch.Tensor, ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """Compress a gradient tensor by keeping the top (ratio * numel) values.

        Args:
            gradient: The full dense gradient tensor.
            ratio: Fraction of elements to retain in [0.0, 1.0].

        Returns:
            Tuple of (top_values, top_indices, original_shape).
        """
        orig_shape = gradient.shape
        flat = gradient.reshape(-1)
        k = max(1, int(flat.numel() * ratio))

        abs_flat = torch.abs(flat)
        topk = torch.topk(abs_flat, k=k, sorted=False)
        indices = topk.indices
        values = flat[indices]

        return values, indices, orig_shape

    @staticmethod
    def decompress(values: torch.Tensor, indices: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """Reconstruct a dense tensor from sparse top-k values and indices."""
        flat = torch.zeros(orig_shape.numel(), dtype=values.dtype, device=values.device)
        flat.scatter_(0, indices, values)
        return flat.reshape(orig_shape)


class FaultTolerantAllReduce:
    """AllReduce synchronizer that dynamically recovers if a worker dies mid-step."""

    def __init__(self, expected_workers: Optional[Sequence[str]] = None) -> None:
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._expected: Set[str] = set(expected_workers) if expected_workers else set()
        self._pending: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        self._reduced: Dict[int, Dict[str, torch.Tensor]] = {}

    @property
    def expected_workers(self) -> Set[str]:
        with self._lock:
            return set(self._expected)

    def register_worker(self, worker_id: str) -> None:
        with self._lock:
            self._expected.add(worker_id)

    def drop_worker(self, worker_id: str) -> None:
        """Drop a dead or lagging worker and resolve any pending step barriers."""
        with self._cv:
            self._expected.discard(worker_id)
            # Check all pending steps to see if removing this worker unblocks the barrier
            for step, submissions in list(self._pending.items()):
                if step in self._reduced:
                    continue
                # If all surviving expected workers have submitted, reduce now
                surviving_submissions = {
                    w: grads for w, grads in submissions.items() if w in self._expected
                }
                if surviving_submissions and len(surviving_submissions) >= len(self._expected):
                    reduced = reduce_gradient_dict(list(surviving_submissions.values()), op="avg")
                    self._reduced[step] = reduced
                    self._cv.notify_all()

    def submit_gradient(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        step: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Submit gradient without blocking. Returns reduced gradients if all active workers submitted."""
        with self._cv:
            self._expected.add(worker_id)
            if step not in self._pending:
                self._pending[step] = {}

            self._pending[step][worker_id] = {k: v.detach().cpu().clone() for k, v in gradients.items()}

            if len(self._pending[step]) >= len(self._expected):
                reduced = reduce_gradient_dict(list(self._pending[step].values()), op="avg")
                self._reduced[step] = reduced
                self._cv.notify_all()
                return {k: v.clone() for k, v in reduced.items()}

            return None

    def get_reduced_gradients(self, step: int) -> Optional[Dict[str, torch.Tensor]]:
        with self._lock:
            return self._reduced.get(step)


class DistributedOptimizer:
    """Wraps any standard PyTorch optimizer and synchronizes gradients across workers on step()."""

    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        synchronizer: Optional[FaultTolerantAllReduce] = None,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.synchronizer = synchronizer or FaultTolerantAllReduce()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, worker_id: str = "w0", step: int = 0) -> None:
        """Synchronize gradients and step the base optimizer."""
        grads = extract_gradients(self.model)
        reduced = self.synchronizer.submit_gradient(worker_id, grads, step=step)

        if reduced is None:
            reduced = self.synchronizer.get_reduced_gradients(step)

        if reduced is not None:
            apply_gradients(self.model, reduced)

        self.optimizer.step()
        self.zero_grad()


class ElasticClusterManager:
    """Handles dynamic scale-up and scale-down of workers with elastic dataset reshuffling."""

    def __init__(self, initial_workers: Optional[Sequence[str]] = None) -> None:
        self._workers: List[str] = list(initial_workers) if initial_workers else []
        self._lock = threading.Lock()
        self.partitioner = DataParallelPartition()

    @property
    def current_workers(self) -> List[str]:
        with self._lock:
            return list(self._workers)

    def register_worker(self, worker_id: str) -> None:
        with self._lock:
            if worker_id not in self._workers:
                self._workers.append(worker_id)

    def remove_worker(self, worker_id: str) -> None:
        with self._lock:
            if worker_id in self._workers:
                self._workers.remove(worker_id)

    def repartition_dataset(self, total_samples: int, mode: str = "contiguous") -> List[List[int]]:
        """Recalculate dataset partitions dynamically based on active worker count."""
        with self._lock:
            count = len(self._workers)
            if count == 0:
                return []
            result = self.partitioner.partition(total=total_samples, parts=count, mode=mode)
            return [p.items for p in result.partitions]
