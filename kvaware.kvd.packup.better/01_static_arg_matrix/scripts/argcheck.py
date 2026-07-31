#!/usr/bin/env python3
"""Drive sglang's own ServerArgs validation over the PD x DPA x hicache matrix.

Why: shows *why* kvd needs kvaware on a decode leg, without spending a single
GPU-minute. A decode leg sets disable_radix_cache=True by itself, and sglang
forbids enable-hierarchical-cache alongside it — so hicache (i.e. kvd) is
illegal on a decode leg until --disaggregation-decode-enable-radix-cache flips
it back, and infera only auto-appends that flag when kv-events are enabled.

Run inside a container that has the model path mounted (from_cli_args touches
the tokenizer). See REPRODUCE.md section B.
"""

import argparse

from sglang.srt.server_args import ServerArgs

M = "/mnt/vast/xiaobo/models/GLM-5.2-MXFP4"


def chk(name, extra):
    p = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(p)
    base = [
        "--model-path", M, "--tp-size", "8", "--trust-remote-code",
        "--kv-cache-dtype", "fp8_e4m3", "--context-length", "32768",
        "--chunked-prefill-size", "8192",
    ]
    try:
        ns = p.parse_args(base + extra)
        sa = ServerArgs.from_cli_args(ns)
        print(f"[OK  ] {name}")
        print(
            f"        hier={sa.enable_hierarchical_cache} "
            f"backend={sa.hicache_storage_backend} "
            f"ratio={getattr(sa, 'hicache_ratio', None)} "
            f"page={sa.page_size} disable_radix={sa.disable_radix_cache}"
        )
    except SystemExit as e:
        print(f"[EXIT] {name}: SystemExit {e}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")


DPA = ["--dp-size", "8", "--enable-dp-attention", "--ep-size", "8"]
HIC = [
    "--enable-hierarchical-cache",
    "--hicache-storage-backend", "dynamic",
    "--hicache-storage-backend-extra-config",
    '{"backend_name":"infera-kvd",'
    '"module_path":"infera.engine.sglang.kvd_adapter",'
    '"class_name":"InferaKvdBackend","prefetch_threshold":64}',
]
PRE = ["--disaggregation-mode", "prefill",
       "--disaggregation-transfer-backend", "mooncake",
       "--disaggregation-bootstrap-port", "8998"]
DEC = ["--disaggregation-mode", "decode",
       "--disaggregation-transfer-backend", "mooncake"]

chk("1 baseline mix", [])
chk("2 mix + hicache", HIC)
chk("3 mix + DPA + hicache", DPA + HIC)
chk("4 PD-prefill + DPA + hicache", PRE + DPA + HIC)
chk("5 PD-decode  + DPA + hicache", DEC + DPA + HIC)
chk("6 PD-decode + DPA + hicache + decode-radix(auto-append)",
    DEC + DPA + HIC + ["--disaggregation-decode-enable-radix-cache"])
chk("7 PD-prefill + DPA + hicache + disable-radix (CONFLICT probe)",
    PRE + DPA + HIC + ["--disable-radix-cache"])
chk("8 PD-decode + DPA + hicache + offload-kvcache",
    DEC + DPA + HIC + ["--disaggregation-decode-enable-offload-kvcache"])
