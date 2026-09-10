# GLM-5.2 scheduler-free internal decode benchmark

A small Python wrapper around the optimized SGLang target + EAGLE workers. It does not create a Scheduler, HTTP server, or PD deployment. Real model weights and MoE routing are used; prefix KV/index contents and acceptance are synthetic. See `results/report.md` for measured results and limitations.

## Verified runtime
-8 MI355X,TP8/EP1/DP1; expected accept length3.61, including bonus.
- Image `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`.
- SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`; AITER `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Weights `/shared_nfs/models/GLM-5.2-MXFP4`, mounted read-only.

## Run on the existing authorized allocation
The launch scripts deliberately pin existing job126175 to node055 and reject any unexpected owner/state/node. They never request or cancel allocations. Do not use nodes234 or036. Once this hold expires, choose another already-authorized allocation and update the guarded job/node explicitly; do not reuse a stale ID.

```bash
ROOT=/shared_nfs/yihou/playground/glm52_decode_internal_yihou_20260909_1057
# One-time creation only when our named container does not already exist:
bash "$ROOT/scripts/create_container_yihou.sh"
# If the existing owned container was stopped after the experiment, start it instead:
# spur exec 126175 docker start yihou-glm52-internal-20260909

# Select a NEW iteration name; existing directories are never overwritten.
bash "$ROOT/scripts/run_decode.sh" my_full_run_yihou \
  --batch-size 16 --input-len 70000 --output-len 10000 \
  --accept-length 3.61 --tp-size 8 --warmup-steps 10 \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
  --mem-fraction-static 0.85
```

`run_decode.sh --dry-run NAME ...` prints the command without contacting the cluster. Inside a correctly configured container, the underlying entry is simply:

```bash
python3 "$ROOT/bench/profile_decode.py" \
  --model-path /shared_nfs/models/GLM-5.2-MXFP4 \
  --batch-size 16 --input-len 70000 --output-len 10000 \
  --accept-length 3.61 --tp-size 8 --warmup-steps 10 \
  --result-dir "$ROOT/iterations/my_direct_run_yihou" \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
  --mem-fraction-static 0.85
```

Use **plain Python, not torchrun**: the driver spawns its own8 TP ranks. `--help` works without Torch/SGLang. Additional options pass through to the pinned ServerArgs parser. Supported topology is single-node TP,DP1/EP1/PP1 and EAGLE5/6/topk1. Defaults select FP8 KV,FlyDSL DSA,aiter topk and graph batch size equal to concurrency. Optimization env vars are in the container-creation script. No global Python environment is modified.

For a fast path check add `--max-steps 16`; such output explicitly has `complete=false` until the requested OSL is reached. `--disable-cuda-graph` is available for eager diagnosis. Model loading/capture remain startup costs; cached kernels do not eliminate real weight loading. Keep the owned container for repeated invocations to retain its image-keyed AITER cache.

## Output and evidence
Each iteration contains exact command/config, live `runtime.log`, mirrored `console.log`, per-rank JSON, `result_yihou.json`, and per-step `steps_yihou.jsonl`. Canonical full run: `iterations/005_full_target_yihou/`. Short rerun: `iterations/006_repeat_segment_yihou/`. `working_process.md` indexes all experiments.

`complete=true` means the wrapper's requested useful-output count was reached, not that generated text was correct. Throughput counts only useful output tokens; input tokens and terminal over-computation are not counted. Runtime length progression is checked against worker-returned lengths before final clipping.

## Tests
```bash
python3 -m unittest discover -s "$ROOT/tests" -v
python3 "$ROOT/scripts/test_run_decode_yihou.py"
```

## Scope and preservation
No original SGLang source or packup was edited. The packup was read while evolving, then all7993 manifest entries were verified; its execution inputs remained unchanged before the full run. No model weights/image archive are copied into this workspace. External image archive and verification details are recorded in `iterations/001_environment/`. Keep original work and other users' containers untouched. No allocation cancellation or broad cleanup is part of these scripts.
