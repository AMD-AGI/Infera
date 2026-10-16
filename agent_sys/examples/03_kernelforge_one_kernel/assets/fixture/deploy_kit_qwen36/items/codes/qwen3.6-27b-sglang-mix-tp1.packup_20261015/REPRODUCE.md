# Reproduce

Run on the GPU node itself; in this run that was `smc300x-ccs-aus-a16-19` with
`transport: local`. Prerequisites:

- docker, with the images `infera:sglang-gfx942-glm52` (or the digest in
  `../environment.yaml`) and `quay.io/coreos/etcd:v3.5.14`;
- the weights at `/data/yihou_hf_cache/models/Qwen3.6-27B`, or set
  `E2E_KIT_MODEL_PATH`;
- one free MI300X;
- `rocm-smi`, `curl`, `ss` and `python3` on the host.

Every identifier is a parameter (`scripts/env.sh`). Set a run tag, a port band,
a work root on local disk, and the card you were given. The defaults are the
producing run's values: tag `r16aea3`, band 18101, card 6.

## 1. Check the card and the port band are free

```bash
cd items/codes/qwen3.6-27b-sglang-mix-tp1.packup_20261015
bash scripts/gpu_used.sh 6
ss -ltn | grep -E ':181(0[1-6]) ' || echo "band 18101-18106 free"
```

## 2. Bring it up (blocks until the router serves the model; about 5 minutes)

```bash
export E2E_KIT_RUN_TAG=repro1 E2E_KIT_PORT_BASE=18101 E2E_KIT_GPU_DEVICES=6
export E2E_KIT_WORK_ROOT=/var/tmp/$USER/kit-work/repro1
bash scripts/deploy.sh
cat "$E2E_KIT_WORK_ROOT/deployment.json"
```

## 3. Read the deployment back and prove it answers through the router

```bash
bash scripts/wait_ready.sh
curl -s http://127.0.0.1:18101/v1/workers
curl -s http://127.0.0.1:18101/v1/models
bash scripts/verify.sh "$E2E_KIT_WORK_ROOT/verify"
```

## 4. Tear it down

```bash
bash scripts/teardown.sh
docker ps -a --filter "label=infera_e2e_run=$E2E_KIT_RUN_TAG" --format '{{.Names}}'
```

## Expected output

The criterion for a successful reproduction is every line below. Timings may
differ; everything else must match.

- `gpu_used.sh 6` prints `6:` followed by a number under 4096 (an idle card read
  `6:284` here). Over 4096 means a co-tenant holds the card, and `deploy.sh`
  will refuse rather than take another.
- `deploy.sh` ends with
  `[deploy HH:MM:SS] READY in Ns. endpoint=http://127.0.0.1:18101 handshake=…/deployment.json`
  and exits 0. N was **276** for both bring-ups here, with weights on local
  NVMe. Before that, `worker /health 200 after ~271s` appears. During the wait
  the worker log repeats `GET /health … 503` for about 4 minutes. That is weight
  load, pool allocation, graph capture and AITER JIT, not a hang.
- `deployment.json` has `"endpoint": "http://127.0.0.1:18101"`,
  `"engine_endpoint": "http://127.0.0.1:18102"`,
  `"container": "yihou_l2a_repro1"`, `"run_tag": "repro1"`,
  `"containerized": true` and `"gpu_devices": [6]`.
- `wait_ready.sh` prints `[wait_ready] READY after …s: http://127.0.0.1:18101 serves Qwen3.6-27B`.
- `/v1/workers` returns exactly one worker with `"status":"active"` and
  `"disagg_mode":"mixed"`.
- `/v1/models` returns exactly one id, `"Qwen3.6-27B"`, which is not a path.
- `verify.sh` prints a JSON with `"all_pass": true`, `"finish_reason": "stop"`,
  a `content` that is non-empty and names Paris (`"\n\nParis"` here), and a
  `reasoning_chars` above zero. It also prints
  `"worker_ready_line_mode": "disagg=DisaggMode.MIXED"`,
  `"engine_cuda_graph_max_bs_decode": 32` and
  `"engine_reasoning_parser": "qwen3"`. It exits 0.
- `teardown.sh` prints a JSON listing `yihou_l2a_repro1` and
  `yihou_l2a_repro1_etcd` under `removed`. It reports
  `"containers_left_with_label": 0`, `"ports_still_listening": []` and
  `"drained": true`, and exits 0. The final `docker ps` prints nothing.

A reproduction **fails** if any of these happen:

- `verify.sh` exits non-zero;
- `content` is empty, or contains the chain of thought (the reasoning parser is
  missing);
- `finish_reason` is `"length"` (the request was too small);
- a model id starts with `/` (the served-name flag is missing);
- `disagg_mode` is not `mixed`.
