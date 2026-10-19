"""Optional cap on this process's CUDA memory (PHOTO_MANAGER_CUDA_MEMORY_FRACTION)."""

from __future__ import annotations

import math
import os

CUDA_MEMORY_FRACTION_ENV: str = "PHOTO_MANAGER_CUDA_MEMORY_FRACTION"

_applied_fraction: float | None = None


def cuda_memory_fraction_from_env() -> float | None:
    """Return the configured fraction in (0, 1], None when unset; raise ValueError if invalid."""
    raw = os.environ.get(CUDA_MEMORY_FRACTION_ENV, "").strip()
    if not raw:
        return None
    try:
        fraction = float(raw)
    except ValueError:
        fraction = math.nan
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"{CUDA_MEMORY_FRACTION_ENV} must be a number greater than 0 and at most 1, got {raw!r}"
        )
    return fraction


def apply_cuda_memory_fraction(fraction: float | None) -> bool:
    """Cap this process's CUDA allocations when CUDA is the selected device.

    Call once before any model loads. Returns True when the cap was set
    (now or by an earlier call); torch is imported only when a fraction is set.
    """
    global _applied_fraction  # noqa: PLW0603
    if fraction is None:
        return False
    if _applied_fraction is not None:
        return True
    import torch  # noqa: PLC0415

    from cull.stage2.iqa import select_device  # noqa: PLC0415

    if select_device() != "cuda":
        return False
    torch.cuda.set_per_process_memory_fraction(fraction)
    _applied_fraction = fraction
    return True
