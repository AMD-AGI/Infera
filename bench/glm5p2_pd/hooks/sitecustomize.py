"""Install opt-in PyTorch allocator hooks before SGLang is imported."""

from __future__ import annotations

import os
import sys


_fraction_text = os.environ.get("INFERA_PYTORCH_MEMORY_FRACTION", "")
if _fraction_text:
    import torch

    _fraction = float(_fraction_text)
    if not 0.0 < _fraction <= 1.0:
        raise ValueError("INFERA_PYTORCH_MEMORY_FRACTION must be in (0, 1]")

    _original_set_device = torch.cuda.set_device
    _configured_devices: set[int] = set()

    def _set_device_and_allocator_fraction(device: object) -> None:
        _original_set_device(device)
        device_index = torch.cuda.current_device()
        if device_index not in _configured_devices:
            torch.cuda.set_per_process_memory_fraction(_fraction, device_index)
            _configured_devices.add(device_index)
            print(
                "[infera] PyTorch allocator memory fraction "
                f"set to {_fraction:g} on device {device_index}",
                file=sys.stderr,
                flush=True,
            )

    torch.cuda.set_device = _set_device_and_allocator_fraction
