"""Unit tests for distributed dataset sharding."""

import pytest
import torch
from torch.utils.data import TensorDataset

from src.model.dataset_sharding import ShardedDataset, shard_dataset


class TestDatasetSharding:
    def test_sharded_dataset_partitioning(self):
        # 100 samples
        x = torch.arange(100).unsqueeze(1).float()
        y = torch.arange(100)
        base_dataset = TensorDataset(x, y)

        # Split across 4 workers
        shards = [
            ShardedDataset(base_dataset, rank=i, num_replicas=4, mode="contiguous")
            for i in range(4)
        ]

        # Each worker should have 25 samples
        for shard in shards:
            assert len(shard) == 25

        # Verify no overlap between shards
        all_samples = []
        for shard in shards:
            for item in shard:
                all_samples.append(int(item[1].item()))

        assert sorted(all_samples) == list(range(100))

    def test_strided_sharding(self):
        x = torch.arange(10).unsqueeze(1).float()
        base_dataset = TensorDataset(x, x)

        s0 = ShardedDataset(base_dataset, rank=0, num_replicas=2, mode="strided")
        s1 = ShardedDataset(base_dataset, rank=1, num_replicas=2, mode="strided")

        assert len(s0) == 5
        assert len(s1) == 5

        s0_indices = [int(item[0].item()) for item in s0]
        s1_indices = [int(item[0].item()) for item in s1]

        assert s0_indices == [0, 2, 4, 6, 8]
        assert s1_indices == [1, 3, 5, 7, 9]

    def test_shard_dataset_helper(self):
        data = list(range(10))
        sharded = shard_dataset(data, rank=0, num_replicas=2)
        assert len(sharded) == 5
        assert sharded == [0, 1, 2, 3, 4]
