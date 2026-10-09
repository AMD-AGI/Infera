#!/usr/bin/env python3
"""Initialize PyTorch before the pinned Mooncake probe installs HIP transport."""
import torch

torch.cuda.init()
from infera.tools.preflight.network import mooncakeperf
mooncakeperf._worker()
