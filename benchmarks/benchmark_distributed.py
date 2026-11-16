"""Performance benchmarks for distributed ML operations in RMLTP."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import SGD

from src.model.tensor_utils import serialize_tensor, deserialize_tensor
from src.model.runner import ModelRunner
from src.model.gradient_sync import AllReduceSynchronizer, reduce_gradient_dict, SyncMode
from src.model.parameter_server import ParameterServer
from src.model.parallel import DataParallelExecutor


class BenchmarkModel(nn.Module):
    """Standard MLP benchmark model."""

    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


def benchmark_tensor_serialization(num_elements: int = 100_000, iterations: int = 10) -> Dict[str, Dict[str, float]]:
    """Benchmark tensor serialization and deserialization with and without compression."""
    tensor = torch.randn(num_elements, dtype=torch.float32)
    raw_size_bytes = tensor.numel() * 4

    results = {}
    for compression in ["none", "gzip"]:
        # Warmup
        _ = deserialize_tensor(serialize_tensor(tensor, compression=compression))

        # Serialization time
        t0 = time.perf_counter()
        for _ in range(iterations):
            payload = serialize_tensor(tensor, compression=compression)
        ser_time = (time.perf_counter() - t0) / iterations

        # Payload size
        payload_bytes = len(payload["data"].encode("ascii"))
        compression_ratio = raw_size_bytes / max(1, payload_bytes)

        # Deserialization time
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = deserialize_tensor(payload)
        deser_time = (time.perf_counter() - t0) / iterations

        results[compression] = {
            "serialize_ms": ser_time * 1000.0,
            "deserialize_ms": deser_time * 1000.0,
            "payload_kb": payload_bytes / 1024.0,
            "compression_ratio": compression_ratio,
        }

    return results


def benchmark_gradient_sync(num_workers: int = 4, iterations: int = 20) -> Dict[str, float]:
    """Benchmark AllReduce vs Parameter Server synchronization overhead."""
    model = BenchmarkModel()
    dummy_grads = {name: torch.randn_like(param) for name, param in model.named_parameters()}

    results = {}

    # 1. Direct AllReduce Reduction
    t0 = time.perf_counter()
    for _ in range(iterations):
        worker_grads = [copy.deepcopy(dummy_grads) for _ in range(num_workers)]
        _ = reduce_gradient_dict(worker_grads, op="avg")
    allreduce_time = (time.perf_counter() - t0) / iterations
    results["allreduce_avg_ms"] = allreduce_time * 1000.0

    # 2. Synchronous Parameter Server
    ps_model = BenchmarkModel()
    ps_opt = SGD(ps_model.parameters(), lr=0.01)
    ps = ParameterServer(ps_model, ps_opt, sync_mode=SyncMode.SYNCHRONOUS, min_workers=num_workers)
    for i in range(num_workers):
        ps.register_worker(f"w-{i}")

    t0 = time.perf_counter()
    for step in range(iterations):
        for i in range(num_workers):
            ps.push(f"w-{i}", dummy_grads, step=step)
    ps_sync_time = (time.perf_counter() - t0) / iterations
    results["ps_sync_step_ms"] = ps_sync_time * 1000.0

    # 3. Asynchronous Parameter Server
    ps_async_model = BenchmarkModel()
    ps_async_opt = SGD(ps_async_model.parameters(), lr=0.01)
    ps_async = ParameterServer(ps_async_model, ps_async_opt, sync_mode=SyncMode.ASYNCHRONOUS)

    t0 = time.perf_counter()
    for step in range(iterations):
        for i in range(num_workers):
            ps_async.push(f"w-{i}", dummy_grads, step=step)
    ps_async_time = (time.perf_counter() - t0) / iterations
    results["ps_async_step_ms"] = ps_async_time * 1000.0

    return results


def benchmark_data_parallel_scaling(
    worker_counts: List[int] = [1, 2, 4],
    batch_size: int = 64,
    steps: int = 10,
) -> Dict[int, float]:
    """Benchmark data-parallel training throughput with varying worker counts."""
    throughput_results = {}

    for count in worker_counts:
        runners = [
            ModelRunner(BenchmarkModel(), SGD(BenchmarkModel().parameters(), lr=0.01))
            for _ in range(count)
        ]
        sync = AllReduceSynchronizer(expected_workers=[f"worker-{i}" for i in range(count)])
        executor = DataParallelExecutor(runners=runners, synchronizer=sync)

        x = torch.randn(batch_size, 128)
        y = torch.randn(batch_size, 10)

        # Warmup
        _ = executor.train_step(x, y)

        t0 = time.perf_counter()
        for _ in range(steps):
            _ = executor.train_step(x, y)
        elapsed = time.perf_counter() - t0

        total_samples = batch_size * steps
        throughput = total_samples / elapsed
        throughput_results[count] = throughput

    return throughput_results


def run_all_benchmarks(quick: bool = False) -> None:
    """Execute all platform performance benchmarks and print formatted summary."""
    iters = 3 if quick else 10
    num_elems = 10_000 if quick else 100_000

    print("=" * 60)
    print("           RMLTP DISTRIBUTED BENCHMARKS")
    print("=" * 60)

    # 1. Serialization Benchmark
    print("\n[1] Tensor Serialization Benchmark:")
    ser_res = benchmark_tensor_serialization(num_elements=num_elems, iterations=iters)
    for comp, stats in ser_res.items():
        print(f"  Compression: {comp:<6} | Serialize: {stats['serialize_ms']:6.2f} ms | "
              f"Deserialize: {stats['deserialize_ms']:6.2f} ms | "
              f"Payload: {stats['payload_kb']:6.1f} KB | Ratio: {stats['compression_ratio']:.2f}x")

    # 2. Gradient Synchronization Benchmark
    print("\n[2] Gradient Synchronization Overhead (4 workers):")
    sync_res = benchmark_gradient_sync(num_workers=4, iterations=iters)
    for k, v in sync_res.items():
        print(f"  {k:<20}: {v:6.2f} ms / step")

    # 3. Scaling Benchmark
    print("\n[3] Data Parallel Training Throughput:")
    counts = [1, 2] if quick else [1, 2, 4]
    scaling_res = benchmark_data_parallel_scaling(worker_counts=counts, batch_size=32, steps=iters)
    for count, tp in scaling_res.items():
        print(f"  {count} Worker(s): {tp:8.1f} samples/sec")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMLTP Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run fast iterations")
    args = parser.parse_args()
    run_all_benchmarks(quick=args.quick)
