"""Configuration file loader utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def merge_configs(*configs: dict) -> dict:
    """
    Merge multiple configuration dictionaries, with later configs overriding earlier ones.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        if not config:
            continue

        for key, value in config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value

    return merged


def validate_config(config: dict, required_keys: list) -> list[str]:
    """
    Validate that all required keys exist in configuration.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names

    Returns:
        List of missing keys (empty if all present)
    """
    missing_keys = [key for key in required_keys if key not in config]
    return missing_keys


__all__ = [
    "load_yaml_config",
    "merge_configs",
    "validate_config",
]