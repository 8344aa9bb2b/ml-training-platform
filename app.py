"""Hugging Face Spaces Entrypoint (100% Free Tier - No Credit Card Required).

Launches:
1. Controller Orchestrator Daemon on tcp://127.0.0.1:8000
2. Worker 1 Daemon on tcp://127.0.0.1:8001
3. Worker 2 Daemon on tcp://127.0.0.1:8002
4. Serves the interactive Web Dashboard on port 7860 (Hugging Face default)
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controller.controller import Controller
from src.node.worker import Worker
from core.logger import get_logger
from dashboard import app


def start_background_cluster():
    """Start Controller and Workers in daemon background threads over real ZeroMQ TCP."""
    logger = get_logger("hf_spaces_supervisor")
    logger.info("Initializing multi-node distributed cluster...")

    # 1. Start Controller
    ctrl_cfg = {
        "controller": {
            "host": "127.0.0.1",
            "port": 8000,
            "heartbeat_interval": 2.0,
            "max_task_retries": 3,
        },
        "monitoring": {
            "prometheus_enabled": False,
            "checkpoint_dir": "./logs/checkpoints",
        },
        "logging": {"level": "INFO"},
    }
    controller = Controller(config=ctrl_cfg, logger=logger)
    ctrl_thread = threading.Thread(target=controller.serve, daemon=True)
    ctrl_thread.start()
    time.sleep(1.0)

    # 2. Start Worker 1
    w1_cfg = {
        "nodes": {
            "id": "worker-hf-1",
            "host": "127.0.0.1",
            "port": 8001,
            "health_check": {"interval": 2.0},
        }
    }
    worker1 = Worker(config=w1_cfg, logger=logger)
    worker1.connect_to_controller("tcp://127.0.0.1:8000")
    w1_thread = threading.Thread(target=worker1.serve, daemon=True)
    w1_thread.start()

    # 3. Start Worker 2
    w2_cfg = {
        "nodes": {
            "id": "worker-hf-2",
            "host": "127.0.0.1",
            "port": 8002,
            "health_check": {"interval": 2.0},
        }
    }
    worker2 = Worker(config=w2_cfg, logger=logger)
    worker2.connect_to_controller("tcp://127.0.0.1:8000")
    w2_thread = threading.Thread(target=worker2.serve, daemon=True)
    w2_thread.start()

    logger.info("Cluster supervisor: Controller + 2 Workers active over ZeroMQ TCP.")


if __name__ == "__main__":
    start_background_cluster()
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 RMLTP Live Dashboard serving on http://0.0.0.0:{port}...\n")
    app.run(host="0.0.0.0", port=port, debug=False)
