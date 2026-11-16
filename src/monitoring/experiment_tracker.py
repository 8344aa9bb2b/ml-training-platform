"""Experiment tracking and run history manager."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRun:
    """Represents a single experiment execution run."""

    run_id: str
    run_name: str
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


class ExperimentTracker:
    """Manages experiment runs, parameter logging, and step-by-step metrics tracking."""

    def __init__(self, storage_dir: str = "./logs/experiments") -> None:
        """Initialize ExperimentTracker.

        Args:
            storage_dir: Directory where run metadata is stored.
        """
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self._runs: Dict[str, ExperimentRun] = {}
        self._lock = threading.Lock()
        self._load_runs()

    def _load_runs(self) -> None:
        """Load stored run files from disk."""
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    run = ExperimentRun(**data)
                    self._runs[run.run_id] = run
                except Exception:
                    pass

    def _persist_run(self, run: ExperimentRun) -> None:
        """Persist an experiment run JSON file."""
        filepath = os.path.join(self.storage_dir, f"{run.run_id}.json")
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(run), f, indent=2)
        os.replace(tmp_path, filepath)

    def start_run(
        self,
        run_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Start a new experiment run.

        Args:
            run_name: Human-readable run name.
            parameters: Job configurations or hyperparameters.
            tags: Arbitrary key-value metadata.

        Returns:
            The created ExperimentRun instance.
        """
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        run = ExperimentRun(
            run_id=run_id,
            run_name=run_name,
            parameters=parameters or {},
            tags=tags or {},
        )
        with self._lock:
            self._runs[run_id] = run
            self._persist_run(run)
        return run

    def log_metric(self, run_id: str, metric_name: str, value: float, step: int) -> None:
        """Log a metric value for a specific training step.

        Args:
            run_id: Identifier of the active run.
            metric_name: Name of metric (e.g. 'loss', 'accuracy').
            value: Float value.
            step: Training step index.
        """
        with self._lock:
            if run_id not in self._runs:
                return

            run = self._runs[run_id]
            if metric_name not in run.metrics:
                run.metrics[metric_name] = []

            run.metrics[metric_name].append({
                "step": step,
                "value": float(value),
                "timestamp": time.time(),
            })
            self._persist_run(run)

    def finish_run(self, run_id: str, status: str = "completed") -> Optional[ExperimentRun]:
        """Mark an experiment run as finished."""
        with self._lock:
            if run_id not in self._runs:
                return None

            run = self._runs[run_id]
            run.status = status
            run.finished_at = time.time()
            self._persist_run(run)
            return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run metadata by ID."""
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> List[ExperimentRun]:
        """Return list of all registered experiment runs."""
        with self._lock:
            return list(self._runs.values())
