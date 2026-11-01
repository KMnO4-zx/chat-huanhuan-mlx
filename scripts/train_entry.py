"""mlx_lm.lora entrypoint with a wired ``clear_cache_threshold``.

mlx-lm 0.32.2 exposes ``--clear-cache-threshold`` but never forwards it to
``TrainingArgs``, whose default of 0 makes ``mx.clear_cache()`` run after
*every* optimizer step. That defeats the allocator's buffer reuse and causes
heavy Metal buffer/residency churn on Apple Silicon.

This shim patches ``mlx_lm.lora.TrainingArgs`` so the threshold defaults to
12GB: for a batch-4 run whose peak is ~7GB, the allocator cache never crosses
the threshold, so buffers are reused every step instead of being released and
re-allocated (which showed up as heavy ``MetalAllocator``/``ResidencySet``
churn). Total memory stays bounded by the run's own peak because the cache can
only hold buffers that were already allocated.

Usage (same CLI as ``mlx_lm.lora``):
    uv run python scripts/train_entry.py -c configs/base.yaml ...
"""

from __future__ import annotations

import mlx_lm.lora as _lora

_DEFAULT_THRESHOLD = 12_000_000_000  # 12 GB of cached memory before clearing


class _PatchedTrainingArgs(_lora.TrainingArgs):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("clear_cache_threshold", _DEFAULT_THRESHOLD)
        super().__init__(*args, **kwargs)


_lora.TrainingArgs = _PatchedTrainingArgs

if __name__ == "__main__":
    _lora.main()
