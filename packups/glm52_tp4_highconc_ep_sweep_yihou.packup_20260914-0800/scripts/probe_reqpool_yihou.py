"""CPU-only: show how the pinned SGLang resolves max_running_requests for our CLI."""
import argparse
from sglang.srt.server_args import ServerArgs

CLI = ["--model-path", "/shared_nfs/models/GLM-5.2-MXFP4",
       "--tp-size", "4", "--ep-size", "1", "--dp-size", "4",
       "--moe-a2a-backend", "none", "--enable-dp-attention",
       "--speculative-algorithm", "EAGLE", "--speculative-num-steps", "5",
       "--speculative-num-draft-tokens", "6", "--speculative-eagle-topk", "1",
       "--kv-cache-dtype", "fp8_e4m3",
       "--dsa-decode-backend", "flydsl", "--dsa-prefill-backend", "flydsl",
       "--dsa-topk-backend", "aiter",
       "--cuda-graph-bs-decode", "40", "--cuda-graph-max-bs-decode", "40",
       "--random-seed", "1234", "--disable-radix-cache", "--skip-tokenizer-init",
       "--disable-overlap-schedule", "--trust-remote-code",
       "--mem-fraction-static", "0.98"]

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
sa = ServerArgs.from_cli_args(parser.parse_args(CLI))
print("max_running_requests  =", sa.max_running_requests)
print("decode cuda graph max_bs =", getattr(getattr(sa, "cuda_graph_config", None), "decode", None)
      and sa.cuda_graph_config.decode.max_bs)
print("disaggregation_mode   =", sa.disaggregation_mode)
print("attn dp size (dp_size)=", sa.dp_size)
