# Patches this kit depends on

Everything below is on branch `llying/example/vllm_1p1d_dsv41_flash`, on top of
`main` at `962845a65` (2026-11-18). Each entry says what breaks without it, how
it was found, and when it can be dropped. The engine image applies the vLLM
patches at build time and **fails the build** if one no longer applies, so a
base-image bump cannot ship an unpatched connector silently.

| # | where | change | status |
|---|---|---|---|
| 1 | infera | parse `vllm serve` args with vLLM ≥ 0.34 | cherry-pick of [Infera#251](https://github.com/AMD-AGI/Infera/pull/251) (open) |
| 2 | infera | `INFERA_VLLM_DP_ENGINE_LB=1`: a DP engine registers as one worker | cherry-pick of [Infera#250](https://github.com/AMD-AGI/Infera/pull/250) (open) |
| 3 | infera (Rust router) | drop idle upstream connections before the engine's keep-alive does | cherry-pick of [Infera#247](https://github.com/AMD-AGI/Infera/pull/247) (open) |
| 4 | infera (server) | raise `RLIMIT_NOFILE` at start-up | cherry-pick of [Infera#248](https://github.com/AMD-AGI/Infera/pull/248) (open) |
| 5 | infera | `infera.engine.vllm.gem_mooncake` + `deploy/docker/native/hip_gem_alloc.c` | ported from `dev/pd_opt/vllm_glm52_agentx` |
| 6 | vLLM (image) | Mooncake producer sends the KV of a prefill that finishes `STOPPED` | [llying-001/vllm#3](https://github.com/llying-001/vllm/pull/3), not upstream |
| 7 | vLLM (image) | Mooncake connector raises libzmq `MAX_SOCKETS` | not upstream |

## 1. vLLM ≥ 0.34 argument parser

**Symptom.** `python -m infera.engine.vllm` dies at import on any vLLM nightly
since 0.34: `make_arg_parser` moved from `vllm.entrypoints.openai.cli_args` to
`vllm.entrypoints.common.cli_args` and now expects vLLM's
`FlexibleArgumentParser`. **Fix.** Try the new location first, fall back to the
old one. **Drop when** #251 merges.

## 2. DP engine load balancing on the decode leg

**Why.** A DP engine normally registers its ranks and the router pins each
request to one with a rank header. On the decode leg there is no prefix cache to
aim at, and the router's input-token booking leaves ranks unevenly full; with
`INFERA_VLLM_DP_ENGINE_LB=1` vLLM's own DP balancer picks the rank from live
queue depth. The prefill leg does **not** set it: there the router keeps each
session on the rank that holds its prefix. **Drop when** #250 merges.

## 3, 4. Router stability under AgentX load

Long agentic sessions reuse pooled upstream connections that uvicorn has
already closed (5 s keep-alive vs the router's 90 s idle pool), which surfaced
as sporadic 502s; and the router holds client + prefill + decode connections
per request, past the default 1024 open files at high concurrency. Both were hit
in the DeepSeek-V4.1 P/D runs of 2026-11-19. The kit additionally sets
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` on the engines. **Drop when** #247 and #248
merge.

## 5. GEM KV allocator (amdgpu ≥ 6.16 without usable peer memory)

**Symptom.** On amdgpu 6.19 (AUS MI355X) the KV cache cannot be registered for
RDMA at a normal utilization target: KFD charges a VRAM buffer pinned through
dma-buf twice, and on 6.19 the peer-memory path is charged twice as well
(fixed upstream in torvalds/linux `9131cc85d0de`). **Fix.** The decode and
prefill legs allocate the KV cache from amdgpu GEM BOs (charged once), imported
through HIP VMM so batch copies run on the owning device, and Mooncake registers
them by dma-buf. `gem_mooncake.MooncakeConnector` is the upstream connector plus
`get_mem_pool_context`; vLLM's `MultiConnector` delegates that hook, so the
prefill leg's Mooncake + UMBP stack also gets it. Selected by
`MR_MODE=gem` (cluster/cluster.gem.sh). **Drop when** the fleet runs a driver
with the KFD fix, or a peer-memory module that charges once.

## 6. Mooncake: send the KV of a prefill that finishes STOPPED

**Symptom.** Under load, a fraction of requests hang until
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`; the decode log shows
`Timeout waiting for P side ready` while the prefill is idle. **Root cause.** A
prefill leg runs with `max_tokens=1`. DeepSeek-V4.1's SWA bounded replay drops
the last prompt token, so the one sampled token is the model's guess at it —
often a stop token — and `check_stop` finishes the request `FINISHED_STOPPED`
before `max_tokens`. `MooncakeConnectorScheduler.request_finished` only sends
KV for `FINISHED_LENGTH_CAPPED`, so the decoder's pull is never answered. On
DeepSeek-V4.1-Flash 1P1D (TP4 + TP4, 16k prompts and GSM8K), 29 of 128 prefill
legs finished STOPPED and exactly those 29 hung. **Fix.** Treat
`FINISHED_STOPPED` like `FINISHED_LENGTH_CAPPED`, as the NIXL connector does
(`deploy/docker/patches/vllm-nightly/patch_mooncake_send_kv_on_stopped.py`).
**Drop when** the base carries llying-001/vllm#3 or an equivalent fix.

## 7. Mooncake: libzmq socket cap

**Symptom.** At AgentX concurrency 768 the decode leg logs
`Too many open files` on every new pull. **Root cause.** The decode connector
holds one DEALER socket per pending pull and producer rank until the prefill
answers; hundreds of queued requests exceed libzmq's default of 1023 sockets
per context. **Fix.** Raise `zmq.MAX_SOCKETS` on the connector's context
(`VLLM_MOONCAKE_ZMQ_MAX_SOCKETS`, default 65536;
`patch_mooncake_zmq_max_sockets.py`). The engine container also raises
`nofile` to 1048576. **Drop when** upstream bounds or pools these sockets.

## Not patched, configured

These looked like bugs and are not: each is a setting the kit makes, with the
reason next to it in `engine/leg.sh`.

- **All KV on one NIC.** Infera's ROCm default pins every GPU to one NIC when
  the NICs span several subnets (one /24 per rail on AUS), so all KV crossed
  `ionic_0`. The kit writes a per-GPU topology (`RDMA_GPU_NICS`).
- **Same-node P/D over HIP IPC.** Mooncake prefers HIP IPC between legs on one
  node. `MC_DISABLE_HIP=1` keeps KV on RDMA, as it would be across nodes.
- **`prefill-schedule-interval` on the prefill leg.** It admits prefills only
  every N steps; on a prefill-only engine that idles N−1 of every N steps. The
  c384 value (11) stays on the decode leg only.
