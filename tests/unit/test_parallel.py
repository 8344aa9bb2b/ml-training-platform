"""Unit tests for DataParallelExecutor and PipelineParallelStage."""

import copy
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.model.runner import ModelRunner
from src.model.gradient_sync import AllReduceSynchronizer, SyncMode
from src.model.parameter_server import ParameterServer
from src.model.parallel import DataParallelExecutor, PipelineParallelStage


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestDataParallelExecutor:
    def test_data_parallel_with_allreduce(self):
        m1 = Net()
        m2 = copy.deepcopy(m1)

        opt1 = SGD(m1.parameters(), lr=0.1)
        opt2 = SGD(m2.parameters(), lr=0.1)

        r1 = ModelRunner(m1, opt1)
        r2 = ModelRunner(m2, opt2)

        sync = AllReduceSynchronizer(expected_workers=["worker-0", "worker-1"])
        executor = DataParallelExecutor(runners=[r1, r2], synchronizer=sync)

        x = torch.randn(10, 4)
        y = torch.randn(10, 2)

        metrics = executor.train_step(x, y)
        assert metrics["total_samples"] == 10
        assert metrics["step"] == 1
        assert "loss" in metrics

        # Both workers should have synchronized weights
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_data_parallel_with_parameter_server(self):
        master_model = Net()
        m1 = copy.deepcopy(master_model)
        m2 = copy.deepcopy(master_model)

        ps_opt = SGD(master_model.parameters(), lr=0.1)
        ps = ParameterServer(master_model, ps_opt, sync_mode=SyncMode.SYNCHRONOUS, min_workers=2)

        r1 = ModelRunner(m1)
        r2 = ModelRunner(m2)

        executor = DataParallelExecutor(runners=[r1, r2], parameter_server=ps)

        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        metrics = executor.train_step(x, y)
        assert metrics["step"] == 1
        assert ps.current_step == 1

        # Both workers should match the master model
        for p_master, p1, p2 in zip(master_model.parameters(), m1.parameters(), m2.parameters()):
            assert torch.allclose(p_master, p1)
            assert torch.allclose(p_master, p2)

    def test_data_parallel_batch_too_small(self):
        r1 = ModelRunner(Net())
        r2 = ModelRunner(Net())
        executor = DataParallelExecutor(runners=[r1, r2])

        x = torch.randn(1, 4)
        y = torch.randn(1, 2)
        with pytest.raises(ValueError, match="smaller than number of workers"):
            executor.train_step(x, y)


class TestPipelineParallelStage:
    def test_pipeline_execution_flow(self):
        layer0 = nn.Linear(4, 3)
        layer1 = nn.Linear(3, 2)

        opt0 = SGD(layer0.parameters(), lr=0.1)
        opt1 = SGD(layer1.parameters(), lr=0.1)

        stage0 = PipelineParallelStage(
            stage_index=0,
            num_stages=2,
            submodule=layer0,
            optimizer=opt0,
        )
        stage1 = PipelineParallelStage(
            stage_index=1,
            num_stages=2,
            submodule=layer1,
            optimizer=opt1,
            criterion=nn.MSELoss(),
        )

        assert stage0.is_first_stage
        assert not stage0.is_last_stage
        assert not stage1.is_first_stage
        assert stage1.is_last_stage

        # Forward pass
        x = torch.randn(4, 4)
        y = torch.randn(4, 2)

        act0 = stage0.forward(x)
        assert act0.shape == (4, 3)

        preds = stage1.forward(act0)
        assert preds.shape == (4, 2)

        # Backward pass
        w0_before = layer0.weight.clone()
        w1_before = layer1.weight.clone()

        grad_act0 = stage1.backward(y)
        assert grad_act0.shape == (4, 3)

        grad_input = stage0.backward(grad_act0)
        assert grad_input is None  # Stage 0 is first stage

        # Step optimizers
        stage0.step()
        stage1.step()

        assert not torch.equal(layer0.weight, w0_before)
        assert not torch.equal(layer1.weight, w1_before)

    def test_invalid_stage_index(self):
        with pytest.raises(ValueError, match="out of range"):
            PipelineParallelStage(stage_index=2, num_stages=2, submodule=nn.Linear(2, 2))
