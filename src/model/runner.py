"""Model runner executing forward/backward passes and local optimization."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.model.tensor_utils import apply_gradients, extract_gradients


class ModelRunner:
    """Executes model training operations on a worker node.

    Manages forward pass, loss calculation, backward propagation, mixed precision
    scaling, and local gradient extraction / weight loading.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        mixed_precision: bool = False,
    ) -> None:
        """Initialize ModelRunner.

        Args:
            model: PyTorch neural network module.
            optimizer: Optional optimizer for parameter updates.
            criterion: Optional loss function (defaults to MSELoss).
            device: Execution device (CPU or CUDA).
            mixed_precision: Whether to enable FP16/AMP mixed precision training.
        """
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.mixed_precision = mixed_precision

        # Mixed precision setup
        self._scaler = None
        if self.mixed_precision:
            if self.device.type == "cuda":
                self._scaler = torch.amp.GradScaler("cuda")
            else:
                # CPU mixed precision uses bfloat16 without loss scaling
                self._scaler = None

    def _autocast_context(self):
        """Return the appropriate autocast context manager based on device."""
        if not self.mixed_precision:
            return contextlib.nullcontext()
        if self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through the model."""
        x = batch_x.to(self.device)
        with self._autocast_context():
            return self.model(x)

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute training loss given predictions and ground truth."""
        y = targets.to(self.device)
        with self._autocast_context():
            return self.criterion(predictions, y)

    def backward(self, loss: torch.Tensor) -> None:
        """Compute backward gradients with optional AMP scaling."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self) -> None:
        """Step the local optimizer and update gradient scaler if present."""
        if self.optimizer is None:
            raise RuntimeError("Cannot step without an optimizer configured")

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

    def zero_grad(self) -> None:
        """Clear gradients on the model and optimizer."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.model.zero_grad()

    def train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        step_optimizer: bool = True,
    ) -> Dict[str, float]:
        """Perform a complete training step: forward, loss, backward, and optional optimizer step.

        Args:
            batch_x: Input features.
            batch_y: Target labels or values.
            step_optimizer: If True, steps optimizer immediately. If False, retains
                            gradients for external synchronization (AllReduce/PS).

        Returns:
            Dict containing step metrics (loss, batch_size).
        """
        self.zero_grad()
        preds = self.forward(batch_x)
        loss = self.compute_loss(preds, batch_y)
        loss_val = float(loss.item())

        self.backward(loss)

        if step_optimizer and self.optimizer is not None:
            self.step()

        return {
            "loss": loss_val,
            "batch_size": int(batch_x.size(0)),
        }

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract and unscale (if scaled) model gradients."""
        if self._scaler is not None and self.optimizer is not None:
            # Unscale before extracting gradients for synchronization
            self._scaler.unscale_(self.optimizer)
        return extract_gradients(self.model)

    def set_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Apply external gradients directly to model parameters."""
        apply_gradients(self.model, gradients)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get cloned CPU state dictionary of model parameters."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load state dictionary into model."""
        self.model.load_state_dict(state_dict)
