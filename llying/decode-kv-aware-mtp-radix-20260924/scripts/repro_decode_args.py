#!/usr/bin/env python3
"""CPU-only reproduction of the decode-leg KV-aware / radix-cache decision.

Runs Infera's own argv parser (``infera.engine.sglang.args.parse_sglang_args``)
and SGLang's ServerArgs resolution on the GLM-5.2 P/D decode command line. It
does not start an engine, touch a GPU, or load weights; only config.json is
read.

Usage (inside the infera-sglang image, model mounted read-only):
    python3 repro_decode_args.py CASE
CASE:
    current   decode + MTP + --enable-kv-events, as Infera ships today
    forced    same, plus an explicit --disaggregation-decode-enable-radix-cache
    patched   same as current, with the SGLang/Infera patches applied and
              SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1
"""

from __future__ import annotations

import json
import logging
import os
import sys

MODEL = os.environ.get("MODEL", "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4")
FLAG = "--disaggregation-decode-enable-radix-cache"

DECODE_ARGV = [
    "--model-path", MODEL, "--served-model-name", "glm5.2-mxfp4",
    "--host", "127.0.0.1", "--port", "29002", "--advertise-host", "127.0.0.1",
    "--etcd-endpoint", "127.0.0.1:22379", "--discovery-backend", "etcd",
    "--request-transport", "http", "--kv-event-transport", "zmq",
    "--tp-size", "8", "--ep-size", "1", "--trust-remote-code",
    "--dsa-prefill-backend", "triton", "--dsa-decode-backend", "triton",
    "--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.85",
    "--chunked-prefill-size", "4096", "--cuda-graph-max-bs-decode", "128",
    "--max-running-requests", "128",
    "--reasoning-parser", "glm45", "--tool-call-parser", "glm47",
    "--enable-cache-report", "--enable-metrics",
    "--disaggregation-mode", "decode", "--disaggregation-transfer-backend", "mooncake",
    "--dp-size", "8", "--enable-dp-attention",
    "--json-model-override-args", '{"index_share_for_mtp_iteration":false}',
    "--speculative-algorithm", "EAGLE", "--speculative-num-steps", "5",
    "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "6",
    "--enable-kv-events", "--kv-events", "on",
    "--kv-events-bind", "tcp://0.0.0.0:25558", "--kv-snapshot-port", "28802",
]

FIELDS = (
    "disaggregation_mode", "speculative_algorithm", "speculative_eagle_topk",
    "enable_dp_attention", "dp_size", "disaggregation_transfer_backend",
    "disaggregation_decode_enable_radix_cache", "disable_radix_cache", "page_size",
)


def resolved(sa) -> dict:
    return {k: getattr(sa, k, None) for k in FIELDS}


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else "current"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from infera.engine.sglang.args import parse_sglang_args

    argv = list(DECODE_ARGV)
    if case == "forced":
        argv.append(FLAG)
    elif case == "patched":
        os.environ["SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC"] = "1"
    elif case != "current":
        print(f"unknown case {case!r}", file=sys.stderr)
        return 2

    out: dict = {"case": case}
    try:
        args = parse_sglang_args(argv)
    except Exception as e:  # noqa: BLE001 - the error is the result
        out["infera_parse"] = f"{type(e).__name__}: {e}"
        print("RESULT " + json.dumps(out, default=str))
        return 0

    out["flag_forwarded_to_sglang"] = FLAG in args.sglang_argv
    out["infera_view"] = resolved(args.server_args)

    # Re-resolve exactly what launch_server would parse. Construction leaves
    # the record raw in this SGLang; resolve_once() runs the hook pipeline,
    # including pd_disaggregation_hook, the way the launcher does.
    import argparse

    from sglang.srt.server_args import ServerArgs

    p = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(p)
    try:
        sa = ServerArgs.from_cli_args(p.parse_args(args.sglang_argv))
        sa.resolve_once()
        out["launch_server_view"] = resolved(sa)
    except Exception as e:  # noqa: BLE001
        out["launch_server_view"] = f"{type(e).__name__}: {e}"

    # Full resolution needs an accelerator. The PD hook alone does not, so run
    # it by itself on a fresh record: it is the check that decides between
    # ChunkCache and a decode radix cache.
    from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation

    sa = ServerArgs.from_cli_args(p.parse_args(args.sglang_argv))
    try:
        handle_pd_disaggregation(sa)
        out["pd_hook"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["pd_hook"] = f"{type(e).__name__}: {e}"
    out["pd_hook_view"] = resolved(sa)
    print("RESULT " + json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
