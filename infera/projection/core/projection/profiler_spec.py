###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field
from typing import Union

from infera.projection.core.projection.base_module_profiler import BaseModuleProfiler
from infera.projection.core.projection.training_config import TrainingConfig


@dataclass
class ModuleProfilerSpec:
    profiler: type[BaseModuleProfiler]
    config: type[TrainingConfig]
    sub_profiler_specs: (
        dict[str, Union[type[BaseModuleProfiler], "ModuleProfilerSpec", None]] | None
    ) = field(default_factory=lambda: {})
