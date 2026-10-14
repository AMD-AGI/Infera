#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Prebuild the AITER varlen attention that cached-prefix chunked MLA uses.

The base prebuilds five `mha_varlen_fwd` variants but not the causal one with
LSE and no skip, which ATOM's chunked-MLA prefill calls once a prompt extends a
cached prefix. Without it, the first such prefill JIT-compiles it for ~30 s and
stalls the engine under load. This builds it the way AITER's runtime JIT would,
into the same directory, so the runtime finds it and builds nothing.

Needs no GPU, but GPU_ARCHS must list the target archs.
"""

import importlib.util
import os
import sys

MD_NAME = "mha_varlen_fwd_bf16_nlogits_nbias_mask_lse_ndropout_nskip_nqscale"

# Import AITER's jit package alone: `import aiter` probes the GPU.
sys.path.insert(0, importlib.util.find_spec("aiter").submodule_search_locations[0])
from jit import core  # noqa: E402
from jit.utils.mha_recipes import get_mha_varlen_prebuild_variants_by_names  # noqa: E402

args = core.get_args_of_build("module_mha_varlen_fwd")
args.update(get_mha_varlen_prebuild_variants_by_names([MD_NAME], core.CK_DIR)[0])
core.build_module(
    MD_NAME,
    args["srcs"],
    args["flags_extra_cc"],
    args["flags_extra_hip"],
    args["blob_gen_cmd"],
    args["extra_include"],
    args["extra_ldflags"],
    args["verbose"],
    args["is_python_module"],
    args["is_standalone"],
    args["torch_exclude"],
    args.get("third_party", []),
    args.get("hipify", False),
    flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
)
so = f"{core.get_user_jit_dir()}/{MD_NAME}.so"
if not os.path.isfile(so):
    sys.exit(f"AITER build left no {so}")
print(f"prebuilt {so}")
