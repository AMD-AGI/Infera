# Patches this kit depends on

Everything below is on branch `llying/example/vllm_1p1d_dsv41_flash`, on top of
`main` at `962845a65` (2026-11-18). Each entry says what breaks without it, how
it was found, and when it can be dropped. The engine image applies the vLLM
patches at build time and **fails the build** if one no longer applies, so a
base-image bump cannot ship an unpatched connector silently.

| # | where | change | commit | status |
|---|---|---|---|---|
| 1 | infera | parse `vllm serve` args with vLLM ≥ 0.34 | `1756103ed` | cherry-pick of [Infera#251](https://github.com/AMD-AGI/Infera/pull/251) (open) |
| 2 | infera (Rust router) | drop idle upstream connections before the engine's keep-alive does | `bb47e540e` | cherry-pick of [Infera#247](https://github.com/AMD-AGI/Infera/pull/247) (open) |
| 3 | infera (server) | raise `RLIMIT_NOFILE` at start-up | `028839c3a` | cherry-pick of [Infera#248](https://github.com/AMD-AGI/Infera/pull/248) (open) |
| 4 | infera | `INFERA_VLLM_DP_ENGINE_LB=1`: a DP engine registers as one worker | `43b1d8420` | cherry-pick of [Infera#250](https://github.com/AMD-AGI/Infera/pull/250) (open) |
| 5 | infera | GEM KV allocator + `gem_mooncake.MooncakeConnector` | `6f0ad33b9` | ported from `dev/pd_opt/vllm_glm52_agentx` |
| 6 | vLLM (image) | Mooncake producer sends the KV of a prefill that finishes `STOPPED` | `9285d2c88` | [llying-001/vllm#3](https://github.com/llying-001/vllm/pull/3), not upstream |
| 7 | vLLM (image) | Mooncake connector raises libzmq `MAX_SOCKETS` | `9285d2c88` | not upstream |
| 8 | infera | opt-in canonical KV for the Mooncake connector (`"canonical_kv": true`) | `752b2990a` | new here |
| 9 | vLLM (image) | Mooncake accepts ROCm's 4 KiB-padded KV backing as block-major | `c8b9acdd8` | new here, not upstream |
| 10 | infera (GEM allocator) | HIP IPC for GEM KV buffers (`pidfd_getfd` + HIP VMM) | `0007cd808` | new here |
| 11 | infera (Rust router) | serial vLLM Mooncake dispatch (`INFERA_VLLM_MOONCAKE_SERIAL=1`) | `91c930286` | cherry-pick of [Infera#246](https://github.com/AMD-AGI/Infera/pull/246) (open) |
| 12 | infera (Rust router) | render DeepSeek-V4.1 chat with the V4.1 encoding for kv-aware routing | `48d70ea88` | new here; unit tests pass, live render parity pending |

## 1. vLLM ≥ 0.34 argument parser

**Symptom.** `python -m infera.engine.vllm` dies at import on any vLLM nightly
since 0.34: `make_arg_parser` moved from `vllm.entrypoints.openai.cli_args` to
`vllm.entrypoints.common.cli_args` and now expects vLLM's
`FlexibleArgumentParser`. **Fix.** Try the new location first, fall back to the
old one. **Drop when** #251 merges.

## 2, 3. Router stability under AgentX load

Long agentic sessions reuse pooled upstream connections that uvicorn has
already closed (5 s keep-alive vs the router's 90 s idle pool), which surfaced
as sporadic 502s; and the router holds client + prefill + decode connections
per request, past the default 1024 open files at high concurrency. Both were hit
in the DeepSeek-V4.1 P/D runs of 2026-11-19. The kit additionally sets
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` on the engines. **Drop when** #247 and #248
merge.

## 4. DP engine load balancing on the decode leg

A DP engine normally registers its ranks and the router pins each request to
one with a rank header. On the decode leg there is no prefix cache to aim at,
and the router's input-token booking leaves ranks unevenly full; with
`INFERA_VLLM_DP_ENGINE_LB=1` vLLM's own DP balancer picks the rank from live
queue depth. The prefill leg does **not** set it: there the router keeps each
session on the rank that holds its prefix. **Drop when** #250 merges.

## 5. GEM KV allocator (amdgpu ≥ 6.16 without usable peer memory)

**Symptom.** On amdgpu 6.19 (AUS MI355X) the KV cache cannot be registered for
RDMA at a normal utilization target: KFD charges a VRAM buffer pinned through
dma-buf twice, and on 6.19 the peer-memory path is charged twice as well
(fixed upstream in torvalds/linux `9131cc85d0de`). **Fix.** The legs allocate the
KV cache from amdgpu GEM BOs (charged once), imported through HIP VMM so batch
copies run on the owning device, and Mooncake registers them by dma-buf.
`gem_mooncake.MooncakeConnector` is the upstream connector plus
`get_mem_pool_context`; vLLM's `MultiConnector` delegates that hook, so the
prefill's Mooncake + UMBP stack also gets it. Selected by `MR_MODE=gem`
(cluster/cluster.gem.sh). **Drop when** the fleet runs a driver with the KFD
fix, or a peer-memory module that charges once.

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

## 8. Canonical KV for the Mooncake connector (keeps `fp8_e4m3`)

**Symptom.** With a Mooncake P/D connector both DeepSeek-V4.1 legs refuse to
start: `--kv-cache-dtype fp8_e4m3 needs gfx950 and the aiter sparse MLA kernel
(VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1), which is off with a KV connector`.
**Root cause.** vLLM's ROCm DeepSeek-V4/V4.1 attention keeps the AITER sparse
MLA kernel only when the connector's `only_restores_canonical_kv()` is true —
"every block it loads is a whole block vLLM's own cache writer produced for an
identical layout" — because that kernel does not scrub NaNs from unwritten
slots. Upstream Mooncake inherits `False`; UMBP returns `True`. Without the
kernel the per-tensor `fp8_e4m3` rows of InferenceX's c384 point cannot be
read, and the fallback is `fp8_ds_mla` + Triton sparse attention.
**Fix.** `gem_mooncake.MooncakeConnector.only_restores_canonical_kv` returns the
connector config's `canonical_kv` (default `False`); the kit sets it on both
legs, which run one model, one KV dtype and one block size, so every block
Mooncake moves is a whole block the prefill's writer produced. Mooncake checks
layer and group layout at handshake and fails a transfer on a mismatch rather
than writing it (#9 is how one surfaced). Validated by the accuracy gate on
every bring-up (GSM8K 0.97–0.975, needles at 16k/64k/128k 32/32). **Drop when**
upstream Mooncake claims canonical KV for homogeneous P/D itself.

## 9. Mooncake: ROCm's padded KV backing is block-major

**Symptom.** Every request fails at the decode's pull with `Mooncake registered
group index mismatch for language_model.model.layers.0.attn.swa_cache:
producer=0, consumer=-1`. **Root cause.** On ROCm vLLM pads the shared KV
backing to a 4 KiB multiple (`v1/worker/utils.py _allocate_kv_cache_buffer`,
whose comment says readers of `storage.nbytes()` must tolerate the tail). The
connector decides whether a layer is a packed block-major region with
`tensor_blocks * block_stride == storage.nbytes()`, so the answer depends on
whether the engine's KV size happens to be 4 KiB-aligned — on its block count.
A prefill at `--gpu-memory-utilization 0.82` and a decode at 0.9 registered the
same layers differently. GLM-5.2 never hit it: with a single KV group no layers
share a view, so packed or not the group index is 0. **Fix.** The flat-buffer
rule: block-major when the storage holds the blocks plus a tail shorter than
one block (`patch_mooncake_packed_rocm_tail.py`). **Drop when** upstream fixes
the check.

## 10. HIP IPC for GEM KV buffers (UMBP standalone on a GEM prefill)

**Symptom.** A GEM prefill with UMBP's standalone server dies at KV
registration: `GPU RegisterMemory failed: hipIpcOpenMemHandle failed: invalid
argument`. **Root cause.** The UMBP server is a separate process that maps the
engine's KV cache through HIP IPC, and HIP has no IPC handle for imported
(GEM/VMM) memory. The interim workaround — a KFD-memory prefill at
`--gpu-memory-utilization 0.82` to fit the doubled KFD charge — cost 38% of the
prefill's GPU prefix cache (18.6M instead of 29.9M tokens per rank).
**Fix.** The GEM allocator answers `hipIpcGetMemHandle` for its buffers with
the exporter's pid and the BO's dma-buf fd; an opener that preloads it copies
the fd with `pidfd_getfd` (same user + `CAP_SYS_PTRACE`, Linux ≥ 5.6) and maps it
through HIP VMM (`hipIpcOpenMemHandle` / `hipIpcCloseMemHandle`). A dma-buf
import is not charged in `kfd.vram_used`, so the BO stays charged once. The kit
preloads the allocator into the UMBP server when the prefill is on GEM
(`UMBP_GEM_IPC=1`, default; `0` restores the KFD/0.82 fallback). Validated: the
server registers both ranks' 62 GB KV regions; a 34k-token prompt re-sent after
`reset_prefix_cache` loads 34,176 of 34,179 tokens from UMBP and answers
correctly. **Drop when** MORI's standalone UMBP registers VMM / dma-buf memory
itself (its `fd_socket` already passes fds), or the driver fix (#5) lands.

## 11. Serial vLLM Mooncake dispatch

**Symptom.** At AgentX concurrency 768 a formal run aborts in warmup: a root
request fails after exactly `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` (1800 s), and
the decode then dies "permanently unhealthy". **Root cause.** The router posts
the prefill and decode legs concurrently; vLLM's decode reserves the whole
prompt's KV on arrival and starts its pull timer. The warmup sends ~800 cold
128k-token prompts at once and one DP2 prefill needs ~24 min to work through
them: decode sat at 97% KV with 0 running requests, and requests still queued
on the prefill near 30 min timed out. GLM-5.2 hit the same structure (its U4).
**Fix.** `INFERA_VLLM_MOONCAKE_SERIAL=1` posts the decode leg after the prefill
reply; the kit sets it by default (`MOONCAKE_SERIAL`). Cost: the decode's
request preprocessing moves onto the TTFT path (estimated 0.1–0.5 s, not yet
measured A/B). **Drop when** #246 merges and the kit can rely on it.

## 12. Router: DeepSeek-V4.1 chat encoding

**Symptom.** kv-aware routing never finds a cached prefix on the prefill: in a
c768 run the router applied 68,967 KV-event batches yet 19,526 prefill picks
all logged `cache_hits=0`. **Root cause.** `detect_dsv4` matched any
architecture containing `DeepseekV4`, so `DeepseekV41ForCausalLM` was rendered
with the V4 encoder. vLLM's `deepseek_v41` tokenizer mode renders in thinking
mode by default and opens with BOS + `<｜System｜>` + `Reasoning Effort: 75 …`;
the router produced BOS + system content, diverging at token 1 on every
request. **Fix.** `encoding_dsv41` ports the plain-chat path of vLLM's renderer
(template-kwargs merge, V4.1 defaults and normalization, `encode_messages`),
tokenized without special tokens; tools, tool calls, images, tasks and
continued final messages are refused rather than approximated. Tests are
byte-exact against 24 pairs rendered by vLLM's own encoder. **Status.** cargo
fmt / clippy `-D warnings` / test pass (rust 1.100; all router tests including
the golden pairs). Not yet checked against a live engine: the next bring-up
checks the router's `infera_router_render_parity` and requires `cache_hits > 0`
on two-turn probes before any AgentX run. The Python router has the same V4
mismatch and is not fixed here.

## Not patched, configured

These looked like bugs and are not: each is a setting the kit makes, with the
reason next to it in `engine/leg.sh` or `common.sh`.

- **All KV on one NIC.** Infera's ROCm default pins every GPU to one NIC when
  the NICs span several subnets (one /24 per rail on AUS), so all KV crossed
  `ionic_0`. The kit writes a per-GPU topology (`RDMA_GPU_NICS`).
- **Same-node P/D over HIP IPC.** Mooncake prefers HIP IPC between legs on one
  node. `MC_DISABLE_HIP=1` keeps KV on RDMA, as it would be across nodes.
- **`prefill-schedule-interval` on the prefill leg.** It admits prefills only
  every N steps; on a prefill-only engine that idles N−1 of every N steps. The
  c384 value (11) stays on the decode leg only.
- **Mooncake notify QPs.** `MC_RDMA_NOTIFY_ENABLED=0`: the vLLM connector
  signals completion over its own ZMQ channel; the notify QPs failed with
  `-302` at GLM-5.2's P8D8 scale.
- **UMBP mode.** `UMBP_MODE=standalone` (default) keeps InferenceX's single
  449 GB pool shared by both prefill DP ranks; `embedded` gives each rank its
  own pool in-process and needs no IPC (validated: smoke + accuracy pass on a GEM
  prefill; a performance candidate, not yet measured).
