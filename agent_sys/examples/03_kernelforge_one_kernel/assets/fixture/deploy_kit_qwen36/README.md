# deploy_kit: Qwen3.6-27B on SGLang behind the infera router, MIX, TP1, MI300X

## Purpose

This handoff is a runnable deployment kit for **Qwen3.6-27B**, served by SGLang
0.5.16 through infera in MIX (aggregated) mode at TP1 on one AMD Instinct MI300X
(gfx942). The infera kv-aware router fronts it and etcd handles discovery. It
carries the scripts that bring the service up, wait for it and tear it down. It
also carries the environment record of the run that produced it, and the
evidence that the run served real completions through the router.

Later stages deploy from this handoff. They do not need a serve step of their
own: `scripts/deploy.sh` is the one entrypoint a program calls.

## Interface

A consumer gets three scripts, eight parameters and one handshake file, plus
the environment record. Everything below was exercised in this run (see
`items/codes/qwen3.6-27b-sglang-mix-tp1.packup_20261015/results/`).

- **Environment record:** `items/codes/environment.yaml`, validated against
  `environment.schema.json`. It holds `fixed.*` (node, image and image id, model,
  served name `Qwen3.6-27B`, `deploy_mode: mix`, `tp_size: 1`,
  `context_length: 32768`, `gpu_arch: gfx942`, `gpu_count: 8`,
  `gpu_devices: [6]`) and `runtime.*` (the container, ports and endpoint of the
  producing run, which has since been torn down).
- **Entrypoints** are in `items/codes/qwen3.6-27b-sglang-mix-tp1.packup_20261015/scripts/`
  and run on the node (`transport: local` here):
  - `deploy.sh` brings up the engine container, etcd, the worker and the
    router. It blocks until the router serves the model, then writes the
    handshake. It exits 0 only when the router answers.
  - `wait_ready.sh` blocks until the router's `/health` returns 200, an
    `active` worker is listed and `/v1/models` lists the served name. It exits
    1 on timeout and 2 if the engine process died.
  - `teardown.sh` removes only the containers labelled
    `infera_e2e_run=$E2E_KIT_RUN_TAG`. It hands back files the engine wrote as
    root, waits for the cards to drain, and prints a JSON report.
- **Parameters**, all in `${X:=default}` form in `scripts/env.sh`:
  `E2E_KIT_RUN_TAG`, `E2E_KIT_PORT_BASE` (a band of six ports: router, worker,
  etcd, etcd peer, kv-events, kv-snapshot), `E2E_KIT_WORK_ROOT`,
  `E2E_KIT_ENGINE_EXTRA_ARGS` (appended last to the worker argv),
  `E2E_KIT_ENGINE_EXTRA_ENV` (exported into the worker process only),
  `E2E_KIT_ROUTER_EXTRA_ARGS` (appended last to the router argv),
  `E2E_KIT_GPU_DEVICES` (default `6`, obeyed exactly) and
  `E2E_KIT_CUDA_GRAPH_MAX_BS` (default `32`, which covers stage 2's
  closed-loop replay capped at concurrency 32 as well as concurrency-16 loads).
- **Handshake:** `$E2E_KIT_WORK_ROOT/deployment.json` with `endpoint` (the
  router, i.e. the product endpoint), `engine_endpoint`, `container`, `run_tag`,
  `work_root_in_container` and `work_root_on_host` (the same path, mounted
  read-write), `measurement_roots`, `containerized: true`, `ports`,
  `gpu_devices` and `cuda_graph_max_bs`.
- **API** is OpenAI-compatible at the endpoint: `/v1/chat/completions`,
  `/v1/models`, `/v1/workers`, `/health` and `/metrics`. Use model id
  `Qwen3.6-27B`. Reasoning text is returned in `reasoning_content`, separate
  from `content`.

## Boundary

The kit is proven only for the shape and host it ran on. Single node, TP1, MIX
mode, one MI300X card, the image `infera:sglang-gfx942-glm52` at digest
`sha256:c05d2beb…0818`, and weights on local NVMe at the path in the record.
Nothing here was tested for PD disaggregation, TP above 1, another GPU
architecture, or weights read over a network filesystem. Cold start was 276 s
from local NVMe; budget much more for NFS.

- **Network exposure.** By default every service binds and advertises
  `127.0.0.1`, as the operator supplied. etcd is unauthenticated, so do not
  re-point `E2E_KIT_NODE_IP` at a routable address on a shared host without
  putting something in front of it. Serving another host through
  `E2E_KIT_NODE_IP` was not tested. One port sits outside the band: sglang's own
  scheduler opens a kv-event publisher on an OS-chosen port, which the router
  registers (`notes.md`).
- **GPU isolation is by `HIP_VISIBLE_DEVICES`, not by device mounts.** The
  container mounts `/dev/dri` whole, so a stage that `docker exec`s in and
  starts its own GPU process must set `HIP_VISIBLE_DEVICES` itself.
- **Mounts a consumer inherits.** The engine container mounts the run root it
  was deployed from, the scratch root `/var/tmp/yihou/demoladder/l2a/scratch`
  and the work root read-write at the same path, plus the weights read-only. A
  consumer that needs other paths must set `E2E_KIT_MEASURE_ROOTS` before
  `deploy.sh`, because it cannot add mounts afterwards. The mount derivation
  accepts the cluster's measured forms (`/shared_nfs`, `/home/USER`,
  `/mnt/m2m_nobackup/...`). It allows other same-path binds only when the
  docker daemon has no authorization plugin, which is the case on this host.
- **What the evidence does not claim.** The load in `results/` is a
  liveness/usability smoke at concurrency 16, not a benchmark. The completion
  check proves the router returned a separated, non-empty answer ("Paris"), not
  answer quality in general.
- **The CUDA graph ceiling** is 32. A consumer that drives more than 32
  concurrent decodes must raise `E2E_KIT_CUDA_GRAPH_MAX_BS`, or decode above 32
  runs eager.
