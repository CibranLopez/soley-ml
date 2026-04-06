"""library.utils.device — torch device and DataLoader worker auto-detection."""

import logging
import os

log = logging.getLogger("library")


def get_device():
    """Return the best available torch.device (CUDA > MPS > CPU)."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: GPU — %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS")
    else:
        import torch
        n_cpus = os.cpu_count() or 4
        torch.set_num_threads(n_cpus)
        device = torch.device("cpu")
        log.info("Device: CPU  (%d cores, torch_threads=%d)",
                 n_cpus, torch.get_num_threads())
    return device


def get_num_workers(requested: int = 0) -> int:
    """Auto-detect DataLoader worker count.

    If ``requested == 0`` (default), uses half the available CPU cores.
    Minimum 2, maximum 8.
    """
    if requested > 0:
        return requested
    n_cpus = os.cpu_count() or 4
    return min(8, max(2, n_cpus // 2))
