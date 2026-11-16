"""Unit tests for tensor serialization and utility functions."""

import gzip
import pytest
import torch
import torch.nn as nn

from src.model.tensor_utils import (
    serialize_tensor,
    deserialize_tensor,
    extract_gradients,
    apply_gradients,
    serialize_gradients,
    deserialize_gradients,
    flatten_tensors,
    unflatten_tensors,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestTensorUtils:
    def test_serialize_deserialize_tensor_none_compression(self):
        tensor = torch.tensor([[1.0, 2.5, -3.2], [4.1, 0.0, 5.5]], dtype=torch.float32)
        payload = serialize_tensor(tensor, compression="none")

        assert payload["shape"] == [2, 3]
        assert payload["dtype"] == "float32"
        assert payload["compression"] == "none"
        assert isinstance(payload["data"], str)

        restored = deserialize_tensor(payload)
        assert torch.allclose(tensor, restored)
        assert restored.dtype == torch.float32

    def test_serialize_deserialize_tensor_gzip_compression(self):
        tensor = torch.randn(10, 20, dtype=torch.float32)
        payload = serialize_tensor(tensor, compression="gzip")

        assert payload["compression"] == "gzip"
        restored = deserialize_tensor(payload)
        assert torch.allclose(tensor, restored)

    def test_serialize_deserialize_tensor_dtypes(self):
        int_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        restored_int = deserialize_tensor(serialize_tensor(int_tensor))
        assert torch.equal(int_tensor, restored_int)
        assert restored_int.dtype == torch.int64

    def test_extract_and_apply_gradients(self):
        model = SimpleModel()
        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()

        grads = extract_gradients(model)
        assert "fc1.weight" in grads
        assert "fc1.bias" in grads
        assert "fc2.weight" in grads
        assert "fc2.bias" in grads
        assert grads["fc1.weight"].shape == model.fc1.weight.grad.shape
        assert torch.allclose(grads["fc1.weight"], model.fc1.weight.grad)

        # Create another model and apply grads
        model2 = SimpleModel()
        apply_gradients(model2, grads)
        assert torch.allclose(model2.fc1.weight.grad, grads["fc1.weight"])
        assert torch.allclose(model2.fc2.bias.grad, grads["fc2.bias"])

    def test_serialize_deserialize_gradients(self):
        model = SimpleModel()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        serialized = serialize_gradients(model)
        assert "fc1.weight" in serialized
        assert "shape" in serialized["fc1.weight"]

        restored_grads = deserialize_gradients(serialized)
        assert "fc1.weight" in restored_grads
        assert torch.allclose(restored_grads["fc1.weight"], model.fc1.weight.grad)

    def test_flatten_and_unflatten_tensors(self):
        t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = torch.tensor([5.0, 6.0, 7.0])
        t3 = torch.tensor(8.0)

        flat, metadata = flatten_tensors([t1, t2, t3])
        assert flat.numel() == 4 + 3 + 1
        assert torch.allclose(flat, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

        unflattened = unflatten_tensors(flat, metadata)
        assert len(unflattened) == 3
        assert torch.equal(unflattened[0], t1)
        assert torch.equal(unflattened[1], t2)
        assert torch.equal(unflattened[2], t3)
