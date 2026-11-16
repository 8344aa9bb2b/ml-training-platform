"""Resource monitoring utilities."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ResourceUsage:
    """Resource usage statistics."""

    cpu_percent: float
    gpu_memory_gb: float
    gpu_utilization: float
    memory_gb: float
    disk_space_gb: float
    timestamp: float

    def __str__(self) -> str:
        return (f"CPU: {self.cpu_percent:.1f}% | "
                f"GPU RAM: {self.gpu_memory_gb:.1f}GB / {self.gpu_utilization:.1f}% | "
                f"SYS RAM: {self.memory_gb:.1f}GB")


class ResourceMonitor:
    """Monitor system and hardware resources."""

    def __init__(self, config: dict, logger):
        """
        Initialize resource monitor.

        Args:
            config: Monitor configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.monitoring_enabled = config.get("enabled", True)

    def get_resources(self) -> Optional[Dict[str, Any]]:
        """
        Get current resource usage.

        Returns:
            Dictionary of resource metrics, or None if disabled
        """
        if not self.monitoring_enabled:
            return None

        try:
            resources = {
                "timestamp": time.time(),
                "cpu": self._get_cpu_usage(),
                "gpu": self._get_gpu_usage(),
                "memory": self._get_memory_usage(),
                "disk": self._get_disk_usage(),
            }
            self.logger.debug(f"Resource usage: {resources.get('gpu')}")
            return resources
        except Exception as e:
            self.logger.warning(f"Failed to get resources: {e}")
            return None

    def _get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage statistics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            return {"percent": cpu_percent}
        except ImportError:
            return {"percent": 0.0}

    def _get_gpu_usage(self) -> Dict[str, float]:
        """Get GPU usage statistics (if available)."""
        try:
            import torch
            if torch.cuda.is_available():
                result = {}
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    result[i] = {
                        "memory_allocated_gb": memory_allocated,
                        "memory_reserved_gb": memory_reserved,
                    }
                return result
            return {"none": 0.0}
        except ImportError:
            return {"none": 0.0}

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss_gb": mem_info.rss / (1024 ** 3),
                "vms_gb": mem_info.vms / (1024 ** 3),
            }
        except ImportError:
            return {"rss_gb": 0.0, "vms_gb": 0.0}

    def _get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics."""
        try:
            import psutil
            disk = psutil.disk_usage("/")
            return {
                "total_gb": disk.total / (1024 ** 3),
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
            }
        except ImportError:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

    def check_thresholds(self, resources: Dict[str, Any]) -> list[str]:
        """
        Check if resources exceed configured thresholds.

        Args:
            resources: Resource usage dictionary

        Returns:
            List of resource warnings
        """
        warnings = []

        if not resources or "gpu" not in resources:
            return warnings

        gpu = resources["gpu"]
        if gpu != {"none": 0.0} and isinstance(gpu, dict):
            for device_id, usage in gpu.items():
                if isinstance(usage, dict):
                    memory_gb = usage.get("memory_allocated_gb", 0)
                    # Example threshold: 8 GB
                    if memory_gb > 8.0:
                        warnings.append(
                            f"GPU {device_id} memory: {memory_gb:.1f}GB exceeds threshold"
                        )

        return warnings


__all__ = ["ResourceUsage", "ResourceMonitor"]