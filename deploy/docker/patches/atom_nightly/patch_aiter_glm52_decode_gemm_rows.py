#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""FP8 GEMM rows for the decode row counts of GLM-5.2 above 128.

The per-token FP8 projections of the decode (qkv_a, q_b, o_proj, indexer ...)
look their FlyDSL kernel up in aiter's a8w8_bpreshuffle tuned table, which for
GLM-5.2 holds M = 64, 128, 256 (a few 96 / 192). The MTP verify runs
M = 4 x graph size, so every graph from 36 up (M 144 ...) takes a padded or
default kernel: the qkv_a GEMM costs 14.9 us at M 160 against 11.3 at 128.

These rows come from aiter's tuner (gemm_a8w8_bpreshuffle_tune.py, FlyDSL
candidates, MI355X) for M 144..240; only the ones at least 5% faster than the
production path at the same shape (the tuner's --run_config, measured on the
same node) are kept, since for M not a multiple of 32 the tuned kernel is often
slower than the padded one. They go in as one more model_configs table, which
aiter merges with the others at startup (none of these keys exist there).

Written against aiter 9205e1059 in rocm/atom-dev:nightly_202610131505.
"""

import importlib.util
import os
import sys

TAG = "aiter-patch glm52-decode-gemm-rows"
REL = "configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5.2_decode.csv"
CSV = """\
gfx,cu_num,M,N,K,q_dtype_w,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
gfx950,256,160,2048,2048,torch.float8_e4m3fn,flydsl,2371,0,5.6743,flydsl_bpreshuflle_32x32x512_F8_F8_B16_1x4x4x2_default,236.54,912.42,0.0
gfx950,256,192,2048,2048,torch.float8_e4m3fn,flydsl,686,0,5.8112,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x2x4x2_default,277.16,924.76,0.0
gfx950,256,224,2048,2048,torch.float8_e4m3fn,flydsl,914,0,5.8566,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x3x4x2_default,320.84,951.16,0.0
gfx950,256,144,2688,6144,torch.float8_e4m3fn,flydsl,557,0,13.4479,flydsl_bpreshuflle_16x64x512_F8_F8_B16_0x2x4x2_default,353.69,1351.43,0.0
gfx950,256,160,2688,6144,torch.float8_e4m3fn,flydsl,1117,0,11.5172,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x4x4x2_default,458.86,1593.99,0.0
gfx950,256,160,3072,6144,torch.float8_e4m3fn,flydsl,1117,0,11.7838,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x4x4x2_default,512.55,1768.57,0.0
gfx950,256,192,3072,6144,torch.float8_e4m3fn,flydsl,1117,0,14.2924,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x4x4x2_default,507.11,1485.66,0.0
gfx950,256,224,3072,6144,torch.float8_e4m3fn,flydsl,72,0,14.8527,flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x0x4x2_default,569.31,1456.09,0.0
gfx950,256,144,3584,512,torch.float8_e4m3fn,flydsl,1012,0,3.7819,flydsl_bpreshuflle_16x128x512_F8_F8_B16_0x4x4x2_default,139.74,777.63,0.0
gfx950,256,160,3584,512,torch.float8_e4m3fn,flydsl,860,0,4.1114,flydsl_bpreshuflle_32x128x256_F8_F8_B16_1x3x0x2_default,142.82,745.2,0.0
gfx950,256,176,3584,512,torch.float8_e4m3fn,flydsl,2165,0,4.3114,flydsl_bpreshuflle_16x64x256_F8_F8_B16_0x4x4x1_default,149.82,739.13,0.0
gfx950,256,192,3584,512,torch.float8_e4m3fn,flydsl,758,0,3.9518,flydsl_bpreshuflle_64x64x256_F8_F8_B16_0x3x0x2_default,178.31,837.48,0.0
gfx950,256,208,3584,512,torch.float8_e4m3fn,flydsl,1009,0,4.5154,flydsl_bpreshuflle_16x64x256_F8_F8_B16_0x4x4x2_default,169.06,760.16,0.0
gfx950,256,224,3584,512,torch.float8_e4m3fn,flydsl,443,0,4.3384,flydsl_bpreshuflle_32x64x256_F8_F8_B16_1x1x4x2_default,189.49,819.5,0.0
gfx950,256,240,3584,512,torch.float8_e4m3fn,flydsl,29,0,4.3525,flydsl_bpreshuflle_80x64x256_F8_F8_B16_0x0x0x2_default,202.37,845.08,0.0
gfx950,256,144,4096,2048,torch.float8_e4m3fn,flydsl,1075,0,7.6923,flydsl_bpreshuflle_48x64x256_F8_F8_B16_1x4x0x2_default,314.07,1282.21,0.0
gfx950,256,160,4096,2048,torch.float8_e4m3fn,flydsl,686,0,7.2955,flydsl_bpreshuflle_32x64x512_F8_F8_B16_1x2x4x2_default,367.95,1374.41,0.0
gfx950,256,192,4096,2048,torch.float8_e4m3fn,flydsl,566,0,7.4497,flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x2x4x2_default,432.4,1389.95,0.0
gfx950,256,224,4096,2048,torch.float8_e4m3fn,flydsl,802,0,8.0847,flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x3x4x2_default,464.84,1321.31,0.0
gfx950,256,144,6144,4096,torch.float8_e4m3fn,flydsl,451,0,13.1698,flydsl_bpreshuflle_48x64x256_F8_F8_B16_1x1x4x2_default,550.33,2090.02,0.0
gfx950,256,160,6144,4096,torch.float8_e4m3fn,flydsl,802,0,13.7254,flydsl_bpreshuflle_32x64x512_F8_F8_B16_0x3x4x2_default,586.73,2024.51,0.0
gfx950,256,192,6144,4096,torch.float8_e4m3fn,flydsl,452,0,14.0465,flydsl_bpreshuflle_48x128x256_F8_F8_B16_1x1x4x2_default,687.98,2015.56,0.0
gfx950,256,144,7168,512,torch.float8_e4m3fn,flydsl,514,0,4.7557,flydsl_bpreshuflle_48x128x256_F8_F8_B16_0x2x0x2_default,222.25,1221.3,0.0
gfx950,256,160,7168,512,torch.float8_e4m3fn,flydsl,29,0,4.7538,flydsl_bpreshuflle_80x64x256_F8_F8_B16_0x0x0x2_default,247.05,1271.76,0.0
gfx950,256,192,7168,512,torch.float8_e4m3fn,flydsl,1438,0,4.7498,flydsl_bpreshuflle_96x64x256_F8_F8_B16_0x1x0x1_default,296.7,1372.86,0.0
gfx950,256,224,7168,512,torch.float8_e4m3fn,flydsl,289,0,5.1789,flydsl_bpreshuflle_112x64x256_F8_F8_B16_0x1x0x2_default,317.47,1350.86,0.0
gfx950,256,240,7168,512,torch.float8_e4m3fn,flydsl,586,0,5.4215,flydsl_bpreshuflle_80x128x256_F8_F8_B16_0x2x4x2_default,324.93,1334.23,0.0
"""


def main() -> int:
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        print(f"[{TAG}] FAILED: package 'aiter' not found")
        return 1
    path = os.path.join(list(spec.submodule_search_locations)[0], REL)
    if not os.path.isdir(os.path.dirname(path)):
        print(f"[{TAG}] FAILED: {os.path.dirname(path)} not found")
        return 1
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            same = f.read() == CSV
        print(f"[{TAG}] {'already applied' if same else 'FAILED: a different ' + path + ' exists'}")
        return 0 if same else 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(CSV)
    print(f"[{TAG}] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
