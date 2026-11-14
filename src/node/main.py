"""Main entry point for the node/worker component."""

import argparse
import sys
import time
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.node.worker import Worker
from src.controller.controller import Controller
from core.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed ML Training Platform - Worker Node")

    parser.add_argument("--config", type=str, default="config/nodes.yaml",
                        help="Path to node configuration file")
    parser.add_argument("--controller", type=str, default=None,
                        help="Controller address in format 'host:port'")
    parser.add_argument("--mode", type=str, choices=["standalone", "joined"],
                        default="joined",
                        help="Operation mode: standalone (no controller) or joined (connect to controller)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        node_config = config.get("nodes", {})
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Setup logging
    logger = get_logger(
        name="worker",
        config=config.get("logging", {})
    )

    logger.info(f"Starting worker node: {node_config.get('id', 'worker-unknown')}")

    # Create and start worker
    worker = Worker(config=config, logger=logger)

    try:
        if args.mode == "standalone":
            # Run in standalone mode
            logger.info("Worker running in standalone mode (no controller)")
            worker.serve_standalone()
        elif args.mode == "joined":
            # Run with controller connection
            logger.info("Worker running in joined mode (connected to controller)")
            worker.connect_to_controller(args.controller or f"{node_config.get('controller_address', 'localhost')}")
            worker.serve()
    except KeyboardInterrupt:
        logger.info("Worker shutting down")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()