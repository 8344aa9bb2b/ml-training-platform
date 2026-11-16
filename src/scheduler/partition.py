"""Workload partitioning strategies.

A partition splits a job's work into per-worker slices. Two granularities are
supported today:

- Data parallel: the input space (dataset samples) is divided across workers.
- Pipeline parallel: the model's layers are divided into sequential stages.

Partitioning is pure computation: it takes counts and returns index ranges. It
never talks to workers, sockets, or the controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class PartitionError(ValueError):
    """Raised when a partition request is invalid or produces an invalid split."""


@dataclass
class Partition:
    """A single worker's slice of a job.

    Attributes:
        index: Zero-based position of this partition in the ordered strategy.
        items: The items assigned to the partition (sample indices or layer indices).
        worker_id: Worker the partition was placed on, if assigned yet.
        metadata: Strategy-specific annotations (e.g. contiguous vs strided).
    """

    index: int
    items: List[int] = field(default_factory=list)
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class PartitionResult:
    """The full outcome of partitioning a job.

    Attributes:
        strategy: Name of the strategy that produced this result.
        partitions: Ordered worker slices.
    """

    strategy: str
    partitions: List[Partition] = field(default_factory=list)

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)

    def sizes(self) -> List[int]:
        """Return the item count of each partition, in order."""
        return [len(p) for p in self.partitions]

    def validate(self) -> None:
        """Check that partitions cover every item exactly once.

        Raises:
            PartitionError: If partitions overlap, skip items, or are empty.
        """
        if not self.partitions:
            raise PartitionError("PartitionResult has no partitions")

        seen = set()
        for partition in self.partitions:
            for item in partition.items:
                if item in seen:
                    raise PartitionError(f"Item {item} appears in more than one partition")
                seen.add(item)

        if len(seen) != sum(len(p) for p in self.partitions):
            raise PartitionError("Partitions contain duplicates")


class PartitionStrategy(ABC):
    """Base class for workload partitioning strategies."""

    name: str = "base"

    @abstractmethod
    def partition(self, total: int, parts: int, **kwargs: Any) -> PartitionResult:
        """Split ``total`` units of work into ``parts`` partitions.

        Args:
            total: Number of items (samples or layers) to divide.
            parts: Number of partitions (typically the number of workers).
            **kwargs: Strategy-specific options.

        Returns:
            A validated PartitionResult.
        """
        raise NotImplementedError

    @staticmethod
    def _check_inputs(total: int, parts: int) -> None:
        """Validate the common total/parts invariants."""
        if parts <= 0:
            raise PartitionError(f"parts must be positive, got {parts}")
        if total <= 0:
            raise PartitionError(f"total must be positive, got {total}")


def _split_evenly(total: int, parts: int) -> List[int]:
    """Return per-partition sizes as evenly as possible, remainder first.

    Sizes differ by at most one and always sum to ``total``. ``parts`` must not
    exceed ``total``.
    """
    base, remainder = divmod(total, parts)
    return [base + 1 if i < remainder else base for i in range(parts)]


class DataParallelPartition(PartitionStrategy):
    """Divide dataset samples across workers.

    Each partition receives a disjoint set of sample indices. With
    ``mode="contiguous"`` (default) every partition gets one contiguous range;
    with ``mode="strided"`` samples are dealt round-robin so each partition
    sees an even mix of the dataset.
    """

    name = "data_parallel"

    def partition(self, total: int, parts: int, mode: str = "contiguous", **_: Any) -> PartitionResult:
        self._check_inputs(total, parts)
        if parts > total:
            raise PartitionError(f"parts ({parts}) exceeds total samples ({total})")
        if mode not in ("contiguous", "strided"):
            raise PartitionError(f"Unknown data-parallel mode: {mode!r}")

        if mode == "strided":
            items_per_part = [list(range(i, total, parts)) for i in range(parts)]
        else:
            items_per_part = []
            start = 0
            for size in _split_evenly(total, parts):
                items_per_part.append(list(range(start, start + size)))
                start += size

        result = PartitionResult(
            strategy=self.name,
            partitions=[
                Partition(index=i, items=items, metadata={"mode": mode})
                for i, items in enumerate(items_per_part)
            ],
        )
        result.validate()
        return result


class PipelineParallelPartition(PartitionStrategy):
    """Divide model layers into sequential pipeline stages.

    Layers stay in their original order so stage *n* consumes stage *n-1*'s
    output. When layer count does not divide evenly the leading stages receive
    one extra layer, balancing the pipeline as much as possible.
    """

    name = "pipeline_parallel"

    def partition(self, total: int, parts: int, **_: Any) -> PartitionResult:
        self._check_inputs(total, parts)
        if parts > total:
            raise PartitionError(
                f"Cannot build {parts} pipeline stages from {total} layers; "
                "more stages than layers would leave stages empty"
            )

        partitions: List[Partition] = []
        start = 0
        for index, size in enumerate(_split_evenly(total, parts)):
            items = list(range(start, start + size))
            partitions.append(
                Partition(index=index, items=items, metadata={"stage": index})
            )
            start += size

        result = PartitionResult(strategy=self.name, partitions=partitions)
        result.validate()
        return result


_STRATEGIES = {
    DataParallelPartition.name: DataParallelPartition,
    PipelineParallelPartition.name: PipelineParallelPartition,
}


def create_partition_strategy(name: str) -> PartitionStrategy:
    """Instantiate a partition strategy by name.

    Args:
        name: One of ``data_parallel`` or ``pipeline_parallel``.

    Raises:
        PartitionError: If the strategy is unknown.
    """
    try:
        return _STRATEGIES[name]()
    except KeyError:
        raise PartitionError(
            f"Unknown partition strategy: {name!r}. "
            f"Choose from {sorted(_STRATEGIES)}"
        ) from None


__all__ = [
    "PartitionError",
    "Partition",
    "PartitionResult",
    "PartitionStrategy",
    "DataParallelPartition",
    "PipelineParallelPartition",
    "create_partition_strategy",
]
