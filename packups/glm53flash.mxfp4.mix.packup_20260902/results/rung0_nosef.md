# Rung 0 + --disable-shared-experts-fusion — PASS  (2026-09-01)

image : lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822 (vendor)
source: sglang PR#36607 head c821c425, bind-mounted
model : /apps/data/models/GLM-5.3-Flash-MXFP4 (unmodified)
launch: bare sglang.launch_server, TP4, GPUs 0-3, no infera
delta vs vendor card: + --disable-shared-experts-fusion

## Evidence
[2026-09-01 12:56:35] The server is fired up and ready to roll!
[2026-09-01 12:54:50 TP0] max_total_num_tokens=7650496, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=32, context_len=65536, available_gpu_mem=55.47 GB
[2026-09-01 12:54:48 TP0] Mamba Cache is allocated. max_mamba_cache_size: 2371, conv_state size: 2.77GB, ssm_state size: 78.76GB 
AITER mHC lines: 8 (expect 2 per rank x 4 = 8)
fault scan: 0 'memory access fault', 0 'HIP error'.
  8 'Traceback' hits, ALL from torch/_dynamo/metrics_context.py (2 per rank) --
  torch.compile telemetry logging, not the model path. Benign-LOOKING; not
  investigated. Recorded rather than waved through.
chat 17*23 -> 391, reasoning_content separated: PASS
