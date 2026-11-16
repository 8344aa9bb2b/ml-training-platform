"""Unit tests for gradient synchronization and AllReduce."""

import threading
import pytest
import torch

from src.model.gradient_sync import (
    SyncMode,
    SyncStrategy,
    reduce_gradient_dict,
    AllReduceSynchronizer,
)


class TestGradientSync:
    def test_sync_enums(self):
        assert SyncMode.SYNCHRONOUS == "synchronous"
        assert SyncMode.ASYNCHRONOUS == "asynchronous"
        assert SyncStrategy.ALL_REDUCE == "all_reduce"
        assert SyncStrategy.PARAMETER_SERVER == "parameter_server"

    def test_reduce_gradient_dict_avg(self):
        grads1 = {
            "w": torch.tensor([1.0, 3.0]),
            "b": torch.tensor([2.0]),
        }
        grads2 = {
            "w": torch.tensor([3.0, 5.0]),
            "b": torch.tensor([4.0]),
        }

        reduced = reduce_gradient_dict([grads1, grads2], op="avg")
        assert torch.allclose(reduced["w"], torch.tensor([2.0, 4.0]))
        assert torch.allclose(reduced["b"], torch.tensor([3.0]))

    def test_reduce_gradient_dict_sum(self):
        grads1 = {"w": torch.tensor([1.0, 2.0])}
        grads2 = {"w": torch.tensor([3.0, 4.0])}

        reduced = reduce_gradient_dict([grads1, grads2], op="sum")
        assert torch.allclose(reduced["w"], torch.tensor([4.0, 6.0]))

    def test_reduce_gradient_dict_weighted(self):
        grads1 = {"w": torch.tensor([10.0])}  # weight 1
        grads2 = {"w": torch.tensor([20.0])}  # weight 3
        # weighted avg: (10*1 + 20*3) / 4 = 70 / 4 = 17.5
        reduced = reduce_gradient_dict([grads1, grads2], op="avg", weights=[1.0, 3.0])
        assert torch.allclose(reduced["w"], torch.tensor([17.5]))

    def test_reduce_gradient_dict_validation_errors(self):
        with pytest.raises(ValueError, match="Cannot reduce empty"):
            reduce_gradient_dict([])

        with pytest.raises(ValueError, match="Unsupported reduction op"):
            reduce_gradient_dict([{"w": torch.tensor([1.0])}], op="max")

        with pytest.raises(ValueError, match="keys mismatch"):
            reduce_gradient_dict([{"w": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}])

        with pytest.raises(ValueError, match="Shape mismatch"):
            reduce_gradient_dict([{"w": torch.tensor([1.0])}, {"w": torch.tensor([1.0, 2.0])}])

    def test_allreduce_synchronizer_multithreaded(self):
        sync = AllReduceSynchronizer(expected_workers=["worker-0", "worker-1"])
        assert "worker-0" in sync.registered_workers
        assert "worker-1" in sync.registered_workers

        results = {}

        def worker_fn(worker_id, val):
            grads = {"param": torch.tensor([val], dtype=torch.float32)}
            reduced = sync.submit_and_sync(worker_id, grads, step=0, timeout=5.0)
            results[worker_id] = reduced

        t0 = threading.Thread(target=worker_fn, args=("worker-0", 10.0))
        t1 = threading.Thread(target=worker_fn, args=("worker-1", 30.0))

        t0.start()
        t1.start()
        t0.join(timeout=3.0)
        t1.join(timeout=3.0)

        assert not t0.is_alive()
        assert not t1.is_alive()
        # Avg of 10.0 and 30.0 is 20.0
        assert torch.allclose(results["worker-0"]["param"], torch.tensor([20.0]))
        assert torch.allclose(results["worker-1"]["param"], torch.tensor([20.0]))

    def test_allreduce_timeout(self):
        sync = AllReduceSynchronizer(expected_workers=["w1", "w2"])
        with pytest.raises(TimeoutError):
            sync.submit_and_sync("w1", {"param": torch.tensor([1.0])}, step=0, timeout=0.1)
