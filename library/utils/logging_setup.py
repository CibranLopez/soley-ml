"""library.utils.logging_setup — configure root logger for notebooks."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``library`` logger for notebook / script use.

    Idempotent — safe to call multiple times.

    Parameters
    ----------
    level : int
        e.g. ``logging.DEBUG``, ``logging.INFO`` (default).
    """
    logger = logging.getLogger("library")
    if logger.handlers:
        return                          # already configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
