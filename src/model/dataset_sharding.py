"""Distributed dataset sharding utilities for data-parallel training."""

from __future__ import annotations

from typing import Any, List, Sequence
from torch.utils.data import Dataset

from src.scheduler.partition import DataParallelPartition


class ShardedDataset(Dataset):
    """Wraps any PyTorch Dataset or Sequence and exposes a disjoint slice for a specific worker rank."""

    def __init__(
        self,
        dataset: Sequence[Any],
        rank: int,
        num_replicas: int,
        mode: str = "contiguous",
        seed: int = 42,
    ) -> None:
        """Initialize ShardedDataset.

        Args:
            dataset: The underlying dataset or list of samples.
            rank: Worker rank index [0, num_replicas - 1].
            num_replicas: Total number of worker replicas.
            mode: Partition mode ('contiguous' or 'strided').
            seed: Optional seed for reproducible partitioning.
        """
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} must be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.mode = mode
        self.seed = seed

        partitioner = DataParallelPartition()
        result = partitioner.partition(total=len(dataset), parts=num_replicas, mode=mode)
        self.indices = result.partitions[rank].items

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        real_idx = self.indices[idx]
        return self.dataset[real_idx]


def shard_dataset(
    data: Sequence[Any],
    rank: int,
    num_replicas: int,
    mode: str = "contiguous",
) -> List[Any]:
    """Slice a python list or sequence for a given worker rank.

    Args:
        data: Sequence of data samples.
        rank: Worker rank index.
        num_replicas: Total number of workers.
        mode: Sharding mode ('contiguous' or 'strided').

    Returns:
        List containing the subset of items for this rank.
    """
    sharded = ShardedDataset(data, rank=rank, num_replicas=num_replicas, mode=mode)
    return [sharded[i] for i in range(len(sharded))]
