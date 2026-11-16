"""Main entry point for the node/worker component."""

import argparse
import sys
import os
import time
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.node.worker import Worker
from core.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed ML Training Platform - Worker Node")

    parser.add_argument("--config", type=str, default="config/nodes.yaml",
                        help="Path to node configuration file")
    parser.add_argument("--controller", type=str, default=None,
                        help="Controller address in format 'host:port' or 'tcp://host:port'")
    parser.add_argument("--id", type=str, default=None,
                        help="Worker node ID (e.g. worker-1)")
    parser.add_argument("--port", type=int, default=None,
                        help="Worker port number")
    parser.add_argument("--mode", type=str, choices=["standalone", "joined"],
                        default="joined",
                        help="Operation mode: standalone (no controller) or joined (connect to controller)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file or return defaults if missing."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {
        "nodes": {
            "id": "worker-default",
            "host": "127.0.0.1",
            "port": 8001,
            "controller_address": "127.0.0.1:8000",
            "health_check": {"interval": 2.0},
        },
        "logging": {"level": "INFO"},
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    node_config = config.setdefault("nodes", {})

    if args.id:
        node_config["id"] = args.id
    if args.port:
        node_config["port"] = args.port

    # Setup logging
    logger = get_logger(
        name=f"worker_{node_config.get('id', 'unknown')}",
        config=config.get("logging", {})
    )

    logger.info(f"Starting worker node: {node_config.get('id', 'worker-unknown')}")

    # Create and start worker
    worker = Worker(config=config, logger=logger)

    try:
        if args.mode == "standalone":
            logger.info("Worker running in standalone mode (no controller)")
            worker.serve_standalone()
        elif args.mode == "joined":
            controller_addr = args.controller or node_config.get("controller_address", "127.0.0.1:8000")
            logger.info(f"Worker connecting to controller at {controller_addr}")
            worker.connect_to_controller(controller_addr)
            worker.serve()
    except KeyboardInterrupt:
        logger.info("Worker shutting down")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
