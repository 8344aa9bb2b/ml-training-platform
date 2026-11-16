"""Unit tests for CheckpointManager."""

import os
import pytest
import torch

from src.monitoring.checkpoint_manager import CheckpointManager, CheckpointMetadata


class TestCheckpointManager:
    def test_save_and_load_checkpoint(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), max_keep=3)

        state = {"weights": torch.tensor([1.0, 2.0, 3.0])}
        meta = mgr.save_checkpoint(step=10, state_dict=state, metrics={"loss": 0.5})

        assert meta.step == 10
        assert os.path.exists(meta.path)
        assert meta.metrics["loss"] == 0.5

        # Load back
        loaded_state, loaded_meta = mgr.load_checkpoint(step=10)
        assert loaded_meta.step == 10
        assert torch.equal(loaded_state["weights"], state["weights"])

    def test_get_latest_checkpoint(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), max_keep=5)

        mgr.save_checkpoint(step=1, state_dict={"w": torch.tensor([1.0])})
        mgr.save_checkpoint(step=5, state_dict={"w": torch.tensor([5.0])})
        mgr.save_checkpoint(step=3, state_dict={"w": torch.tensor([3.0])})

        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        assert latest.step == 5

        loaded_state, _ = mgr.load_checkpoint()  # Default loads latest
        assert torch.equal(loaded_state["w"], torch.tensor([5.0]))

    def test_pruning_max_keep(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), max_keep=2)

        m1 = mgr.save_checkpoint(step=1, state_dict={"w": torch.tensor([1.0])})
        m2 = mgr.save_checkpoint(step=2, state_dict={"w": torch.tensor([2.0])})
        assert len(mgr.list_checkpoints()) == 2
        assert os.path.exists(m1.path)

        # Third checkpoint triggers pruning of step 1
        m3 = mgr.save_checkpoint(step=3, state_dict={"w": torch.tensor([3.0])})
        assert len(mgr.list_checkpoints()) == 2
        assert not os.path.exists(m1.path)
        assert os.path.exists(m2.path)
        assert os.path.exists(m3.path)

    def test_manifest_persistence(self, tmp_path):
        dir_path = str(tmp_path / "ckpts")
        mgr1 = CheckpointManager(checkpoint_dir=dir_path, max_keep=3)
        mgr1.save_checkpoint(step=10, state_dict={"w": torch.tensor([10.0])})

        # New instance pointing to same directory loads manifest
        mgr2 = CheckpointManager(checkpoint_dir=dir_path, max_keep=3)
        assert len(mgr2.list_checkpoints()) == 1
        latest = mgr2.get_latest_checkpoint()
        assert latest.step == 10
