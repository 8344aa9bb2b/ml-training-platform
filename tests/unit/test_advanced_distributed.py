"""Unit tests for advanced distributed features (compression, distributed optimizer, fault-tolerant AllReduce, elasticity)."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from src.model.advanced_distributed import (
    TopKCompressor,
    DistributedOptimizer,
    FaultTolerantAllReduce,
    ElasticClusterManager,
)


class TestAdvancedDistributed:
    def test_topk_compression(self):
        tensor = torch.tensor([0.1, 5.0, 0.05, -10.0, 0.01])
        # Keep top 40% (2 items out of 5)
        vals, indices, orig_shape = TopKCompressor.compress(tensor, ratio=0.4)
        assert len(vals) == 2
        assert len(indices) == 2

        decompressed = TopKCompressor.decompress(vals, indices, orig_shape)
        assert decompressed.shape == tensor.shape
        # Top 2 values were 5.0 and -10.0
        assert 5.0 in decompressed
        assert -10.0 in decompressed
        assert decompressed[0] == 0.0  # Zeroed out

    def test_distributed_optimizer(self):
        m0 = nn.Linear(4, 2)
        m1 = nn.Linear(4, 2)
        m1.load_state_dict(m0.state_dict())

        opt0 = SGD(m0.parameters(), lr=0.1)
        opt1 = SGD(m1.parameters(), lr=0.1)

        sync = FaultTolerantAllReduce(expected_workers=["w0", "w1"])
        dist_opt0 = DistributedOptimizer(opt0, model=m0, synchronizer=sync)
        dist_opt1 = DistributedOptimizer(opt1, model=m1, synchronizer=sync)

        x = torch.randn(4, 4)
        m0(x).sum().backward()
        m1(x).sum().backward()

        w_before = m0.weight.clone()
        dist_opt0.step(worker_id="w0", step=0)
        dist_opt1.step(worker_id="w1", step=0)

        assert not torch.equal(m0.weight, w_before)

    def test_fault_tolerant_allreduce_worker_drop(self):
        sync = FaultTolerantAllReduce(expected_workers=["w0", "w1", "w2"])

        # w0 and w1 submit gradients
        sync.submit_gradient("w0", {"w": torch.tensor([2.0])}, step=0)
        sync.submit_gradient("w1", {"w": torch.tensor([4.0])}, step=0)

        # w2 dies before submitting
        sync.drop_worker("w2")

        # After dropping w2, the barrier completes with w0 and w1
        reduced = sync.get_reduced_gradients(step=0)
        assert reduced is not None
        assert torch.allclose(reduced["w"], torch.tensor([3.0]))  # avg of 2.0 and 4.0

    def test_elastic_cluster_scaling(self):
        elastic = ElasticClusterManager(initial_workers=["w0", "w1"])
        assert elastic.current_workers == ["w0", "w1"]

        # Scale up
        elastic.register_worker("w2")
        assert len(elastic.current_workers) == 3

        # Shard recalculation
        shards = elastic.repartition_dataset(total_samples=30)
        assert len(shards) == 3
        assert len(shards[0]) == 10

        # Scale down
        elastic.remove_worker("w1")
        assert elastic.current_workers == ["w0", "w2"]
        shards_new = elastic.repartition_dataset(total_samples=30)
        assert len(shards_new) == 2
        assert len(shards_new[0]) == 15
