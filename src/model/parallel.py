"""Data parallelism and pipeline parallelism execution modules."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.scheduler.partition import DataParallelPartition, PipelineParallelPartition
from src.model.runner import ModelRunner
from src.model.gradient_sync import AllReduceSynchronizer, reduce_gradient_dict
from src.model.parameter_server import ParameterServer
from src.model.tensor_utils import apply_gradients


class DataParallelExecutor:
    """Coordinates data-parallel training across multiple worker ModelRunners.

    Partitions incoming data batches across workers, executes local forward/backward passes,
    and synchronizes gradients using AllReduce or a Parameter Server.
    """

    def __init__(
        self,
        runners: Sequence[ModelRunner],
        synchronizer: Optional[AllReduceSynchronizer] = None,
        parameter_server: Optional[ParameterServer] = None,
        partition_mode: str = "contiguous",
    ) -> None:
        """Initialize DataParallelExecutor.

        Args:
            runners: List of worker ModelRunner instances.
            synchronizer: Optional AllReduceSynchronizer instance.
            parameter_server: Optional ParameterServer instance.
            partition_mode: Data partitioning mode ('contiguous' or 'strided').
        """
        if not runners:
            raise ValueError("DataParallelExecutor requires at least one ModelRunner")

        self.runners = list(runners)
        self.num_workers = len(runners)
        self.synchronizer = synchronizer
        self.parameter_server = parameter_server
        self.partition_mode = partition_mode
        self.partition_strategy = DataParallelPartition()
        self.current_step = 0

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict[str, float]:
        """Execute a synchronous distributed training step across all runners.

        Args:
            batch_x: Full batch inputs of shape (N, ...).
            batch_y: Full batch targets of shape (N, ...).

        Returns:
            Dict containing aggregated metrics: 'loss', 'total_samples'.
        """
        total_samples = batch_x.size(0)
        if total_samples < self.num_workers:
            raise ValueError(
                f"Batch size ({total_samples}) is smaller than number of workers ({self.num_workers})"
            )

        # 1. Partition data across workers
        partition_result = self.partition_strategy.partition(
            total=total_samples,
            parts=self.num_workers,
            mode=self.partition_mode,
        )

        step_losses = []
        worker_grads = []
        weights = []

        # 2. Local forward and backward passes (without local optimizer stepping)
        for i, (runner, partition) in enumerate(zip(self.runners, partition_result.partitions)):
            indices = torch.tensor(partition.items, dtype=torch.long)
            sub_x = batch_x[indices]
            sub_y = batch_y[indices]

            metrics = runner.train_step(sub_x, sub_y, step_optimizer=False)
            step_losses.append(metrics["loss"] * len(indices))
            worker_grads.append(runner.get_gradients())
            weights.append(float(len(indices)))

        # 3. Synchronize gradients
        if self.parameter_server is not None:
            # Push gradients to parameter server
            for i, grads in enumerate(worker_grads):
                worker_id = f"worker-{i}"
                self.parameter_server.push(worker_id, grads, step=self.current_step)

            # Pull authoritative weights from parameter server and load into runners
            updated_state, _ = self.parameter_server.pull()
            for runner in self.runners:
                runner.load_state_dict(updated_state)

        elif self.synchronizer is not None:
            # AllReduce across workers
            reduced_grads = None
            for i, grads in enumerate(worker_grads):
                worker_id = f"worker-{i}"
                res = self.synchronizer.submit_gradient(
                    worker_id=worker_id,
                    gradients=grads,
                    step=self.current_step,
                )
                if res is not None:
                    reduced_grads = res

            if reduced_grads is None:
                raise RuntimeError("AllReduce did not complete for all workers")

            # Apply reduced gradients and step each runner's optimizer
            for runner in self.runners:
                runner.set_gradients(reduced_grads)
                runner.step()

        else:
            # Direct weighted average reduction
            reduced_grads = reduce_gradient_dict(worker_grads, op="avg", weights=weights)
            for runner in self.runners:
                runner.set_gradients(reduced_grads)
                runner.step()

        self.current_step += 1
        avg_loss = sum(step_losses) / total_samples

        return {
            "loss": float(avg_loss),
            "total_samples": total_samples,
            "step": self.current_step,
        }


class PipelineParallelStage:
    """Represents a sequential stage in a pipeline-parallel execution model.

    Holds a sub-module of layers, runs forward activation computation, and
    propagates backward activation gradients to adjacent stages.
    """

    def __init__(
        self,
        stage_index: int,
        num_stages: int,
        submodule: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize PipelineParallelStage.

        Args:
            stage_index: Zero-based index of this stage.
            num_stages: Total number of stages in the pipeline.
            submodule: Sequential neural network module for this stage.
            optimizer: Optional optimizer for parameters in this stage.
            criterion: Loss function (only used if stage is the final stage).
            device: Execution device.
        """
        if stage_index < 0 or stage_index >= num_stages:
            raise ValueError(f"stage_index {stage_index} out of range [0, {num_stages - 1}]")

        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device or torch.device("cpu")
        self.submodule = submodule.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or (nn.MSELoss() if self.is_last_stage else None)

        self._saved_inputs: Optional[torch.Tensor] = None
        self._saved_outputs: Optional[torch.Tensor] = None

    @property
    def is_first_stage(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage_index == self.num_stages - 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute forward pass for this pipeline stage.

        Args:
            inputs: Raw batch input (for stage 0) or activations from stage - 1.

        Returns:
            Output activations to pass to stage + 1 (or final predictions).
        """
        x = inputs.to(self.device)
        if not self.is_first_stage:
            # Activations between stages are independent leaf tensors with requires_grad=True
            x = x.detach().clone().requires_grad_(True)

        self._saved_inputs = x
        outputs = self.submodule(x)
        self._saved_outputs = outputs
        return outputs

    def backward(self, target_or_grad: torch.Tensor) -> Optional[torch.Tensor]:
        """Execute backward pass for this pipeline stage.

        Args:
            target_or_grad: Targets y (if last stage) or output activation gradients
                            from stage + 1.

        Returns:
            Input activation gradients to pass backward to stage - 1, or None if stage 0.
        """
        if self._saved_outputs is None:
            raise RuntimeError("Cannot call backward before forward pass")

        if self.is_last_stage:
            if self.criterion is None:
                raise RuntimeError("Last stage requires a loss criterion")
            target = target_or_grad.to(self.device)
            loss = self.criterion(self._saved_outputs, target)
            loss.backward()
        else:
            grad_output = target_or_grad.to(self.device)
            self._saved_outputs.backward(grad_output)

        if not self.is_first_stage and self._saved_inputs is not None:
            return self._saved_inputs.grad
        return None

    def step(self) -> None:
        """Step local stage optimizer and zero gradients."""
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.submodule.zero_grad()
        self._saved_inputs = None
        self._saved_outputs = None
