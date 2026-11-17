import torch


def is_fp8_fnuz() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx94")
