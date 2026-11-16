"""Tensor serialization, extraction, and tensor utility operations."""

from __future__ import annotations

import base64
import gzip
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


# Map PyTorch dtypes to string identifiers and numpy dtypes
_DTYPE_MAP = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_STR_TO_TORCH_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def serialize_tensor(tensor: torch.Tensor, compression: str = "none") -> Dict[str, Any]:
    """Serialize a PyTorch tensor to a JSON-compatible dictionary.

    Args:
        tensor: The torch.Tensor to serialize.
        compression: Compression algorithm ('none' or 'gzip').

    Returns:
        Dict containing shape, dtype, compression type, and base64-encoded buffer.
    """
    detached = tensor.detach().cpu()
    shape = list(detached.shape)
    dtype_str = _DTYPE_MAP.get(detached.dtype, str(detached.dtype).replace("torch.", ""))

    # Convert to contiguous numpy buffer or raw bytes
    np_arr = detached.numpy()
    raw_bytes = np_arr.tobytes()

    if compression == "gzip":
        data_bytes = gzip.compress(raw_bytes)
    elif compression == "none":
        data_bytes = raw_bytes
    else:
        raise ValueError(f"Unsupported compression algorithm: {compression}")

    encoded_str = base64.b64encode(data_bytes).decode("ascii")

    return {
        "shape": shape,
        "dtype": dtype_str,
        "compression": compression,
        "data": encoded_str,
    }


def deserialize_tensor(data: Dict[str, Any]) -> torch.Tensor:
    """Deserialize a dictionary created by serialize_tensor back into a PyTorch tensor.

    Args:
        data: Dictionary with 'shape', 'dtype', 'compression', and 'data'.

    Returns:
        torch.Tensor with original values, shape, and dtype.
    """
    raw_b64 = data["data"].encode("ascii")
    data_bytes = base64.b64decode(raw_b64)

    compression = data.get("compression", "none")
    if compression == "gzip":
        data_bytes = gzip.decompress(data_bytes)
    elif compression != "none":
        raise ValueError(f"Unsupported compression algorithm: {compression}")

    shape = data["shape"]
    dtype_str = data["dtype"]
    torch_dtype = _STR_TO_TORCH_DTYPE.get(dtype_str, getattr(torch, dtype_str))

    # Reconstruct numpy array
    np_dtype = np.dtype(dtype_str)
    arr = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)

    return torch.from_numpy(arr.copy()).to(dtype=torch_dtype)


def extract_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract cloned CPU gradients from a model's named parameters.

    Args:
        model: PyTorch module whose gradients to extract.

    Returns:
        Dictionary mapping parameter name to gradient tensor.
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().cpu().clone()
    return gradients


def apply_gradients(model: nn.Module, gradients: Dict[str, torch.Tensor]) -> None:
    """Apply external gradient tensors to a model's parameters.

    Args:
        model: PyTorch module to update gradients on.
        gradients: Mapping of parameter name to gradient tensor.
    """
    for name, param in model.named_parameters():
        if name in gradients:
            grad = gradients[name].to(device=param.device, dtype=param.dtype)
            param.grad = grad.clone()


def serialize_gradients(
    source: Union[nn.Module, Dict[str, torch.Tensor]],
    compression: str = "none"
) -> Dict[str, Dict[str, Any]]:
    """Extract (if model) and serialize gradients for network transmission.

    Args:
        source: PyTorch model or dictionary of gradient tensors.
        compression: Compression mode ('none' or 'gzip').

    Returns:
        Dictionary mapping parameter names to serialized tensor payloads.
    """
    if isinstance(source, nn.Module):
        grads = extract_gradients(source)
    else:
        grads = source

    return {
        name: serialize_tensor(grad, compression=compression)
        for name, grad in grads.items()
    }


def deserialize_gradients(payload: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Deserialize a dictionary of serialized gradient payloads.

    Args:
        payload: Mapping of parameter names to serialized tensor payloads.

    Returns:
        Mapping of parameter names to PyTorch gradient tensors.
    """
    return {
        name: deserialize_tensor(data)
        for name, data in payload.items()
    }


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[torch.Size, int]]]:
    """Flatten a sequence of tensors into a single contiguous 1D tensor.

    Args:
        tensors: Sequence of tensors.

    Returns:
        Tuple of (flat_tensor, metadata) where metadata is a list of (shape, numel).
    """
    if not tensors:
        return torch.tensor([]), []

    metadata = [(t.shape, t.numel()) for t in tensors]
    flat = torch.cat([t.detach().reshape(-1) for t in tensors], dim=0)
    return flat, metadata


def unflatten_tensors(
    flat: torch.Tensor,
    metadata: Sequence[Tuple[torch.Size, int]]
) -> List[torch.Tensor]:
    """Unflatten a 1D contiguous tensor back into a list of tensors.

    Args:
        flat: 1D contiguous tensor.
        metadata: List of (shape, numel) tuples from flatten_tensors.

    Returns:
        List of unflattened tensors.
    """
    outputs = []
    offset = 0
    for shape, numel in metadata:
        slice_tensor = flat[offset : offset + numel].reshape(shape)
        outputs.append(slice_tensor)
        offset += numel
    return outputs
