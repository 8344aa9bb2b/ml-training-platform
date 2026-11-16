"""Unit tests validating benchmark execution pipeline."""

import pytest
from benchmarks.benchmark_distributed import (
    benchmark_tensor_serialization,
    benchmark_gradient_sync,
    benchmark_data_parallel_scaling,
)


class TestBenchmarks:
    def test_tensor_serialization_benchmark(self):
        res = benchmark_tensor_serialization(num_elements=1000, iterations=2)
        assert "none" in res
        assert "gzip" in res
        assert res["none"]["serialize_ms"] > 0
        assert res["none"]["deserialize_ms"] > 0

    def test_gradient_sync_benchmark(self):
        res = benchmark_gradient_sync(num_workers=2, iterations=2)
        assert "allreduce_avg_ms" in res
        assert "ps_sync_step_ms" in res
        assert "ps_async_step_ms" in res

    def test_data_parallel_scaling_benchmark(self):
        res = benchmark_data_parallel_scaling(worker_counts=[1, 2], batch_size=8, steps=2)
        assert 1 in res
        assert 2 in res
        assert res[1] > 0
        assert res[2] > 0
