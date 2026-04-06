"""
library.utils
==============

Shared utilities: logging setup and device detection.
"""

from .logging_setup import setup_logging
from .device import get_device, get_num_workers

__all__ = ["setup_logging", "get_device", "get_num_workers"]
