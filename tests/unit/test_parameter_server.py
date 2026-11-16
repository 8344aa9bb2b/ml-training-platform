"""Unit tests for ParameterServer."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.model.gradient_sync import SyncMode
from src.model.parameter_server import ParameterServer


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(1.0)


class TestParameterServer:
    def test_sync_mode_execution(self):
        model = LinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        ps = ParameterServer(
            model=model,
            optimizer=optimizer,
            sync_mode=SyncMode.SYNCHRONOUS,
            min_workers=2,
        )
        ps.register_worker("w1")
        ps.register_worker("w2")

        # Worker 1 pushes grads: [[2.0, 2.0]]
        stepped = ps.push("w1", {"fc.weight": torch.tensor([[2.0, 2.0]])}, step=0)
        assert not stepped
        assert ps.current_step == 0
        assert torch.allclose(model.fc.weight, torch.tensor([[1.0, 1.0]]))

        # Worker 2 pushes grads: [[4.0, 4.0]]
        # Average grads: [[3.0, 3.0]]
        # SGD update: w = 1.0 - 0.1 * 3.0 = 0.7
        stepped = ps.push("w2", {"fc.weight": torch.tensor([[4.0, 4.0]])}, step=0)
        assert stepped
        assert ps.current_step == 1
        assert torch.allclose(model.fc.weight, torch.tensor([[0.7, 0.7]]))

        state, step = ps.pull("w1")
        assert step == 1
        assert torch.allclose(state["fc.weight"], torch.tensor([[0.7, 0.7]]))

    def test_async_mode_execution_and_staleness(self):
        model = LinearModel()
        optimizer = SGD(model.parameters(), lr=0.1)
        ps = ParameterServer(
            model=model,
            optimizer=optimizer,
            sync_mode=SyncMode.ASYNCHRONOUS,
            staleness_bound=2,
        )

        # Immediate update at step 0: staleness 0 -> factor 1.0
        # Grads: [[2.0, 2.0]], update: 1.0 - 0.1 * 2.0 = 0.8
        stepped = ps.push("w1", {"fc.weight": torch.tensor([[2.0, 2.0]])}, step=0)
        assert stepped
        assert ps.current_step == 1
        assert torch.allclose(model.fc.weight, torch.tensor([[0.8, 0.8]]))

        # Next update with step 0: staleness = 1 - 0 = 1 -> factor 1 / (1 + 1) = 0.5
        # Grads: [[4.0, 4.0]] * 0.5 = [[2.0, 2.0]], update: 0.8 - 0.1 * 2.0 = 0.6
        stepped = ps.push("w2", {"fc.weight": torch.tensor([[4.0, 4.0]])}, step=0)
        assert stepped
        assert ps.current_step == 2
        assert torch.allclose(model.fc.weight, torch.tensor([[0.6, 0.6]]))

        # Stale update with step 0 when current_step is 2: staleness = 2 (within bound=2)
        # Advance step further to exceed bound
        ps.current_step = 5
        # Step 0 with current_step 5 has staleness 5 > bound 2 -> dropped
        rejected = ps.push("w3", {"fc.weight": torch.tensor([[10.0, 10.0]])}, step=0)
        assert not rejected
        assert ps.current_step == 5

    def test_checkpoint_save_and_load(self, tmp_path):
        model = LinearModel()
        optimizer = SGD(model.parameters(), lr=0.05)
        ps = ParameterServer(model=model, optimizer=optimizer, sync_mode=SyncMode.SYNCHRONOUS)
        ps.push("w1", {"fc.weight": torch.tensor([[10.0, 10.0]])}, step=0)

        ckpt_path = str(tmp_path / "ps_ckpt.pt")
        ps.save_checkpoint(ckpt_path)

        # Mutate model
        with torch.no_grad():
            model.fc.weight.zero_()
        ps.current_step = 99

        # Reload
        ps.load_checkpoint(ckpt_path)
        assert ps.current_step == 1
        assert torch.allclose(model.fc.weight, torch.tensor([[0.5, 0.5]]))
