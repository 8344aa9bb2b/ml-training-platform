"""Central checkpoint management, tracking, and pruning."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class CheckpointMetadata:
    """Metadata describing a saved training checkpoint.

    Attributes:
        step: Training step index.
        timestamp: Unix epoch timestamp when saved.
        path: Absolute path to the serialized checkpoint file.
        metrics: Associated training metrics (e.g. loss, accuracy).
        worker_id: Identifier of originating worker or 'global'.
    """

    step: int
    timestamp: float
    path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    worker_id: str = "global"


class CheckpointManager:
    """Manages creation, manifest indexing, pruning, and retrieval of checkpoints."""

    def __init__(self, checkpoint_dir: str, max_keep: int = 5) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory where checkpoints and manifest will be stored.
            max_keep: Maximum number of recent checkpoints to retain before pruning oldest.
        """
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.max_keep = max(1, max_keep)
        self.manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        self._lock = threading.Lock()

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._checkpoints: List[CheckpointMetadata] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load checkpoint manifest from disk if it exists."""
        if not os.path.exists(self.manifest_path):
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._checkpoints = [
                CheckpointMetadata(**item)
                for item in data.get("checkpoints", [])
                if os.path.exists(item.get("path", ""))
            ]
        except Exception:
            self._checkpoints = []

    def _save_manifest(self) -> None:
        """Atomically persist manifest to disk."""
        tmp_path = f"{self.manifest_path}.tmp"
        payload = {
            "updated_at": time.time(),
            "checkpoints": [asdict(c) for c in self._checkpoints],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.manifest_path)

    def save_checkpoint(
        self,
        step: int,
        state_dict: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        worker_id: str = "global",
    ) -> CheckpointMetadata:
        """Atomically save a checkpoint to disk and update manifest.

        Args:
            step: Training step index.
            state_dict: Model or optimizer state dictionary.
            metrics: Optional step metrics.
            worker_id: Originating worker or 'global'.

        Returns:
            CheckpointMetadata for the saved checkpoint.
        """
        with self._lock:
            filename = f"checkpoint_step_{step}.pt"
            final_path = os.path.join(self.checkpoint_dir, filename)
            tmp_path = f"{final_path}.tmp"

            payload = {
                "step": step,
                "timestamp": time.time(),
                "state_dict": state_dict,
                "metrics": metrics or {},
                "worker_id": worker_id,
            }

            # Atomic save
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)

            meta = CheckpointMetadata(
                step=step,
                timestamp=payload["timestamp"],
                path=final_path,
                metrics=metrics or {},
                worker_id=worker_id,
            )

            # Insert keeping ordered by step
            self._checkpoints = [c for c in self._checkpoints if c.step != step]
            self._checkpoints.append(meta)
            self._checkpoints.sort(key=lambda c: c.step)

            # Prune if exceeding max_keep
            while len(self._checkpoints) > self.max_keep:
                oldest = self._checkpoints.pop(0)
                if os.path.exists(oldest.path):
                    try:
                        os.remove(oldest.path)
                    except OSError:
                        pass

            self._save_manifest()
            return meta

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Return the most recent checkpoint metadata, or None if none exist."""
        with self._lock:
            if not self._checkpoints:
                return None
            return max(self._checkpoints, key=lambda c: c.step)

    def load_checkpoint(self, step: Optional[int] = None) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """Load checkpoint state dictionary and metadata.

        Args:
            step: Specific step to load. If None, loads the latest checkpoint.

        Returns:
            Tuple of (state_dict, metadata).

        Raises:
            FileNotFoundError: If no matching checkpoint exists.
        """
        with self._lock:
            if not self._checkpoints:
                raise FileNotFoundError("No checkpoints found")

            if step is None:
                meta = max(self._checkpoints, key=lambda c: c.step)
            else:
                matches = [c for c in self._checkpoints if c.step == step]
                if not matches:
                    raise FileNotFoundError(f"Checkpoint for step {step} not found")
                meta = matches[0]

            payload = torch.load(meta.path, weights_only=False)
            return payload["state_dict"], meta

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """Return list of all registered checkpoints."""
        with self._lock:
            return list(self._checkpoints)

    def cleanup_all(self) -> None:
        """Delete all checkpoints and clear manifest."""
        with self._lock:
            for c in self._checkpoints:
                if os.path.exists(c.path):
                    try:
                        os.remove(c.path)
                    except OSError:
                        pass
            self._checkpoints.clear()
            if os.path.exists(self.manifest_path):
                try:
                    os.remove(self.manifest_path)
                except OSError:
                    pass
