# Description

Moves `Dockerfile.sglang` from the SGLang v0.5.18 base to `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018` and keeps infera's SGLang integration working on it. Adds two runnable GLM-5.2-MXFP4 kits for MI355X, tuned for the InferenceX AgentX replay, and a router option the PD kit uses:

- `examples/sglang_mix_glm5.2`: aggregated TP4 + DP4 attention on one node.
- `examples/sglang_2p1d_glm5.2`: PD across two nodes, **TP1 × PP4 prefills** (one or two) and a TP + DP-attention decode on 8 or 4 GPUs. One recipe; `env.sh` picks the shape for each stretch of the Pareto line, and the README gives the first MI355X baseline for it.
- Router `--decode-request-weight` (folded in from #213): decode input demand also charges the requests a target already holds, so DP ranks are balanced by request count as well as by tokens.

Rebased onto main `adbb9dc5` (10-23); every commit is signed off.

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [x] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] Infra/Build change
- [ ] Code refactoring

## Changes

**Image**

- `Dockerfile.sglang`:
  - takes the v0.5.22 base;
  - `deploy/docker/patch.upstream.status.md` records, for every patch, whether it is baked, already upstream, or not needed on this base (each re-verified in a throwaway container of the base).
- New v0.5.22 patches:
  - `sglang_dsa/v0522/` (page-table rows re-cut, apply script);
  - `sglang_disagg/v0522/patch_prefill_nonfinal_chunk_wait_event.py`;
  - `sglang_responses/v0522/patch_responses_unstreamed_tool_args.py`;
  - `sglang_rocm/v0522/patch_mla_absorb_bmm_default_config.py`;
  - `sglang_kv_events/patch_kv_metrics_send_noblock.py`;
  - `sglang_pp/v0522/patch_deepseek_v2_pp_embed_dim.py`: `DeepseekV2Model` sizes its gemm-output allocator from `embed_tokens`, a `PPMissingLayer` on middle pipeline stages, so `--pp-size` > 2 failed to load; it now reads `config.hidden_size`.
- Updated patches:
  - DSA indexer-rows and HiCache staged write-back patches tolerate the new layout;
  - `patch_decode_radix_spec.py` is a no-op on this base.
- `build_mooncake_sglang.sh`: installs yalantinglibs from Mooncake's pinned submodule, which the v0.5.22 base no longer ships.

**infera**

- `infera/engine/sglang/args.py`:
  - v0.5.22 split `--cuda-graph-max-bs` into `--cuda-graph-max-bs-decode` / `-prefill`, and argparse then rejects the old spelling as an ambiguous prefix (e2e-mixed failed on it). When the running SGLang has no exact `--cuda-graph-max-bs`, it is mapped onto `--cuda-graph-max-bs-decode`; older images are unchanged.
  - The decode radix-cache guard builds the model config with `model_config_of` where `ServerArgs.get_model_config` is gone.
- `infera/router/kv_event/responses_input.py`: the Responses shim sets `supports_inline_system = False`, which v0.5.22 reads in `_construct_input_messages`; when a body's system messages render differently with and without inline system support, it returns None and the body routes on load.
- Tests: `tests/engine/sglang/test_cuda_graph_max_bs_args.py` (new), `tests/engine/sglang/test_decode_radix_cache_guard.py`, `tests/unit/router/test_responses_input.py`.

**Router** (`--decode-request-weight W`, `INFERA_DECODE_REQUEST_WEIGHT`)

- Decode input demand picks the target with the least `booked input tokens + this request's tokens + W × requests it holds` (booked at dispatch, released when the decode stream ends), the shape of Dynamo's `router-decode-active-request-weight`. Under DP attention a step waits for the busiest rank, whose cost grows with its request count, so token-only demand can leave one rank with many short requests. `W = 0` (default) is today's token-only demand; `W > 0` requires `--decode-input-demand on` or `shadow`.
- New metric `infera_router_decode_demand_requests{target}`; Python launcher flag; `manual/reference/cli.md`, `manual/features/kv_aware_routing.md`.

**Examples** (same layout as the ATOM kits)

- **Shared**: one site-specific `env.sh`; `engine/up.sh`, `smoke.sh`, `eval.sh`, `down.sh`; no automatic gates. The clients and `down.sh` read the run's own settings (router URL, container names) back.
- **Matching InferenceX**: AgentX runner pinned to InferenceX `918524ff` with its client constraints; GSM8K through InferenceX's `run_eval` on the full 1,319 questions.
- **Common engine settings**: `--disable-custom-all-reduce` (with the AITER custom all-reduce on, about 30% of GSM8K answers at concurrency 64 come out garbled under DP attention plus decode CUDA graphs; mixed README, note 6); `SGLANG_OPT_USE_TOPK_V2=true`.
- **PD recipe** (`examples/sglang_2p1d_glm5.2`):
  - prefills: TP1 × PP4, layers 20 / 20 / 20 / 18 (the last stage also holds the LM head and the NextN layer), overlap scheduler off, 8k chunks, shortest-prefill-first, write-back HiCache at 250 GB per stage; they carry the decode's EAGLE flags so the transfer includes draft-layer KV (note 10);
  - decode: TP + DP attention over `DECODE_GPUS` (8, or 4 for the 4+4 shape), EAGLE MTP 5/1/6;
  - KV: Mooncake over the eight ionic rails, destination-local rail, `MC_MAX_MR_SIZE=1 GiB`;
  - router: kv-aware with prefill session affinity and decode input demand; `DECODE_REQUEST_WEIGHT` (default 0) sets the request weight;
  - `agentx.sh` records the prefill as TP1 / PP4 and the decode TP from `DECODE_GPUS`.

## First Pareto baseline (MI355X, AgentX, InferenceX `918524ff`)

Each point on a freshly started deployment, 3,600 s measured, 10 warm-up requests per lane, forced MTP acceptance 3.61; throughput per GPU over all of the deployment's GPUs. Full table and the command for each shape in the README.

| shape | GPUs | decode request weight | point | tok/s/GPU | interactivity p90 |
|---|---|---|---|---|---|
| 2P1D | 16 | 0 | C176 | 38,392 | 43.2 |
| 1P, 4-GPU decode | 8 | 0 | C80 | 37,310 | 44.3 |
| 1P, 4-GPU decode | 8 | 0 | C64 | 32,532 | 48.6 |
| 1P1D | 12 | 1,000,000 | C88 | 31,088 | 71.7 |
| 1P1D | 12 | 1,000,000 | C64 | 28,981 | 86.0 |
| 1P1D | 12 | 1,000,000 | C48 | 22,331 | 103.9 |
| 1P1D | 12 | 1,000,000 | C32 | 16,309 | 111.7 |

![Pareto](https://github.com/AMD-AGI/Infera/blob/dev/pd_opt/glm_5.2_agentx/llying/2p1d-v0522-20261020/figures/pareto-p90-baseline-20261023.png?raw=true)

- Against the InferenceX MI355X lines (SGLang and ATOM, single node, up to C48) it is 1.4-2.2x higher from p90 43 to 107. Against ATOM's unlisted PD run (InferenceX PR #4045) it is 2-16% lower from p90 43 to 55 and 4-27% higher from 60 to 107; against the B300 single-node line it is up to 9% lower to p90 55, up to 26% higher from 60 to 100, and level at 107 (interpolated between each line's points).
- Request balancing on 1P1D (same deployment, weight 0 → 1,000,000): C88 29,071 → 31,088 tok/s/GPU at p90 50.7 → 71.7, C64 27,253 → 28,981 at 66.0 → 86.0; the busiest of the eight decode ranks runs 8 requests at p90 instead of 15 at C88, at the same step time per load. The KV-bound shapes (4-GPU decode, 2P1D) keep weight 0.

## Validation

- **GitHub CI** (on `3ad7527c`): dco, lint, unit, unit-torch-cpu, rust, agent_sys pass; engine and e2e were still queued.
- **Locally on the rebased branch**: `pre-commit run --all-files` (cargo-fmt / gofmt need toolchains the container lacks; `cargo fmt --check` passes separately), unit 2,864 passed, `cargo clippy -D warnings` clean, `cargo test` 369 + 46 + 4 + 14 passed.
- **In an image built from `3ad7527c`** (the head minus README and the weight default): `tests/unit` 2,656 passed / 91 skipped, `tests/engine/sglang` every file passes.
- **PD example on that image** (1P1D, real acceptance): smoke 6/6; GSM8K 0.9666 strict / 0.9674 flexible; real acceptance length median 4.62; AgentX 1P1D C64 / C48 / C32 above. The other baseline points ran from a copy of the kit with the same flags on the previous image (the PP patch bind-mounted, the router built from this source).
- **mixed** (C48, crsuse2-m2m-140, previous image; the kit is unchanged since): smoke 6/6; GSM8K at concurrency 64 0.980; needle 64/64; AgentX C48 34,662 tok/s/GPU at p90 22.7.

**Known issues** (documented in the READMEs):
- **Custom all-reduce garbling** (mixed note 6), worked around with `--disable-custom-all-reduce`.
- **ionic MR registration** (PD note 8): ionic rejects a 2 GiB dma-buf registration that starts 1 GiB-aligned; `MC_MAX_MR_SIZE=1 GiB` avoids it.
- **Prefill GPU memory access fault** (PD note 9, open): once with the previous TP4 prefills, none yet with PP4; it leaves the GPU unusable until an administrator resets it.
- **PP4 costs** (PD note 11): a single long cold prompt is slower (66k tokens 37.5 s against 14.2 s), and a saturated prefill keeps pushing long cold first turns back.
- **Slow VRAM release** after teardown with HiCache: 10–65 min.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
