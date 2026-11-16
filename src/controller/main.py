"""Main entry point for the controller component."""

import argparse
import sys
import os
import time
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.controller.controller import Controller
from core.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed ML Training Platform - Controller")

    parser.add_argument("--config", type=str, default="config/controller.yaml",
                        help="Path to controller configuration file")
    parser.add_argument("--host", type=str, default=None,
                        help="Host to bind controller on")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to bind controller on")
    parser.add_argument("--mode", type=str, choices=["serve", "job"],
                        default="serve", help="Operation mode: serve scheduler or run single job")
    parser.add_argument("--job-config", type=str, default=None,
                        help="Path to job configuration file (required when mode=job)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file or provide fallback defaults."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {
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


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    ctrl_cfg = config.setdefault("controller", {})

    if args.host:
        ctrl_cfg["host"] = args.host
    if args.port:
        ctrl_cfg["port"] = args.port

    # Setup logging
    logger = get_logger(name="controller", config=config.get("logging", {}))

    logger.info(f"Starting controller in {args.mode} mode on {ctrl_cfg.get('host', '127.0.0.1')}:{ctrl_cfg.get('port', 8000)}")

    # Create and start controller
    controller = Controller(config=config, logger=logger)

    if args.mode == "serve":
        # Run as ongoing service
        logger.info("Controller running in service mode")
        try:
            controller.serve()
        except KeyboardInterrupt:
            logger.info("Controller shutting down")
        except Exception as e:
            logger.error(f"Controller error: {e}")
            sys.exit(1)
        finally:
            controller.stop()

    elif args.mode == "job":
        if not args.job_config:
            logger.error("Job configuration file required for job mode")
            sys.exit(1)

        try:
            job_config = load_config(args.job_config)
            logger.info(f"Starting training job: {job_config.get('job', {}).get('name', 'unspecified')}")
            controller.run_job(job_config)
        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            sys.exit(1)
        finally:
            controller.stop()


if __name__ == "__main__":
    main()
