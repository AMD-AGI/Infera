import functools

import torch


@functools.cache
def get_device_core_count(device_id: int = 0) -> int:
    return torch.cuda.get_device_properties(device_id).multi_processor_count


@functools.cache
def is_gfx95_supported() -> bool:
    return torch.version.hip is not None and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx95")
