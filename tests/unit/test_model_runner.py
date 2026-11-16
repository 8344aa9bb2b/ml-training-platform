"""Unit tests for ModelRunner."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.model.runner import ModelRunner


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestModelRunner:
    def test_forward_and_loss(self):
        model = ToyNet()
        runner = ModelRunner(model=model, criterion=nn.MSELoss())

        x = torch.randn(5, 4)
        y = torch.randn(5, 2)

        preds = runner.forward(x)
        assert preds.shape == (5, 2)

        loss = runner.compute_loss(preds, y)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_train_step_with_optimizer(self):
        model = ToyNet()
        optimizer = SGD(model.parameters(), lr=0.1)
        runner = ModelRunner(model=model, optimizer=optimizer, criterion=nn.MSELoss())

        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        w_before = model.fc.weight.clone()
        metrics = runner.train_step(x, y, step_optimizer=True)

        assert "loss" in metrics
        assert metrics["batch_size"] == 8
        assert not torch.equal(model.fc.weight, w_before)

    def test_train_step_without_optimizer_step(self):
        model = ToyNet()
        optimizer = SGD(model.parameters(), lr=0.1)
        runner = ModelRunner(model=model, optimizer=optimizer, criterion=nn.MSELoss())

        x = torch.randn(4, 4)
        y = torch.randn(4, 2)

        w_before = model.fc.weight.clone()
        metrics = runner.train_step(x, y, step_optimizer=False)

        # Weights shouldn't have changed yet
        assert torch.equal(model.fc.weight, w_before)

        # Gradients are extracted
        grads = runner.get_gradients()
        assert "fc.weight" in grads
        assert grads["fc.weight"] is not None

        # Manual step changes weights
        runner.step()
        assert not torch.equal(model.fc.weight, w_before)

    def test_mixed_precision_execution(self):
        model = ToyNet()
        optimizer = SGD(model.parameters(), lr=0.01)
        runner = ModelRunner(
            model=model,
            optimizer=optimizer,
            mixed_precision=True,
        )

        x = torch.randn(4, 4)
        y = torch.randn(4, 2)

        metrics = runner.train_step(x, y, step_optimizer=True)
        assert "loss" in metrics
        assert metrics["batch_size"] == 4

    def test_state_dict_operations(self):
        m1 = ToyNet()
        m2 = ToyNet()
        r1 = ModelRunner(m1)
        r2 = ModelRunner(m2)

        state = r1.get_state_dict()
        r2.load_state_dict(state)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)
