from types import SimpleNamespace

import torch


def current_arch():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return SimpleNamespace(hip=None)
    gcn = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return SimpleNamespace(hip=SimpleNamespace(gcn_arch=gcn))
