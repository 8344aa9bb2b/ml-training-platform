"""Unit tests for ExperimentTracker."""

import os
import pytest

from src.monitoring.experiment_tracker import ExperimentTracker, ExperimentRun


class TestExperimentTracker:
    def test_start_and_finish_run(self, tmp_path):
        tracker = ExperimentTracker(storage_dir=str(tmp_path))

        run = tracker.start_run(
            run_name="resnet-baseline",
            parameters={"lr": 0.01, "batch_size": 64, "workers": 4},
        )
        assert run.run_id is not None
        assert run.status == "running"

        # Log metrics across steps
        tracker.log_metric(run.run_id, "loss", 0.85, step=1)
        tracker.log_metric(run.run_id, "loss", 0.62, step=2)
        tracker.log_metric(run.run_id, "accuracy", 0.75, step=2)

        # Finish run
        finished = tracker.finish_run(run.run_id, status="completed")
        assert finished.status == "completed"
        assert len(finished.metrics["loss"]) == 2
        assert finished.metrics["loss"][1]["value"] == 0.62

        # Verify persisted run
        retrieved = tracker.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.run_name == "resnet-baseline"

    def test_list_runs(self, tmp_path):
        tracker = ExperimentTracker(storage_dir=str(tmp_path))
        r1 = tracker.start_run("run-1", {"lr": 0.1})
        r2 = tracker.start_run("run-2", {"lr": 0.01})

        runs = tracker.list_runs()
        assert len(runs) == 2
        names = [r.run_name for r in runs]
        assert "run-1" in names
        assert "run-2" in names
