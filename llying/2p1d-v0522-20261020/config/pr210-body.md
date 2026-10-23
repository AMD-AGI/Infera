> **Status (10-23).** Rebased onto main `adbb9dc5`; every commit is signed off, and the #187 commits main already carries were dropped (so was the `moe_mlp.py` ruff fix, now on main). Added since the description below was written, which will be rewritten with the new numbers when the runs finish:
> - the 2P1D example's prefills are TP1 × PP4 (new image patch `sglang_pp/v0522/patch_deepseek_v2_pp_embed_dim.py`), and the decode takes 8 or 4 GPUs;
> - #213 is folded in: router `--decode-request-weight` (decode input demand also charges the requests a target holds), on by default in the example (`DECODE_REQUEST_WEIGHT=1000000`).
>
> The example is being validated on an image built from this head (smoke, GSM8K, AgentX 1P1D C88 / C64 / C48 / C32).

# Description

Moves `Dockerfile.sglang` from the SGLang v0.5.18 base to `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018` and keeps infera's SGLang integration working on it. Adds two runnable GLM-5.2-MXFP4 kits for MI355X, tuned for the InferenceX AgentX replay:

- `examples/sglang_mix_glm5.2`: aggregated TP4 + DP4 attention on one node.
- `examples/sglang_2p1d_glm5.2`: PD-disaggregated 2P(TP4)1D(TP8) across two nodes; `PREFILL_WORKERS=1` gives 1P1D.

Also fixes the ruff-format failure on `moe_mlp.py` that breaks `pre-commit run --all-files` on main.

The ssh transport for the e2e-disag harness (`feat/e2e-disag-ssh-transport`), which was used to run e2e-disag on two nodes without SLURM, is not part of this PR; CI keeps using SLURM.

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
  - `sglang_kv_events/patch_kv_metrics_send_noblock.py`.
- Updated patches:
  - DSA indexer-rows and HiCache staged write-back patches tolerate the new layout;
  - `patch_decode_radix_spec.py` is a no-op on this base.
- `build_mooncake_sglang.sh`: installs yalantinglibs from Mooncake's pinned submodule, which the v0.5.22 base no longer ships.

**infera**

- `infera/engine/sglang/args.py`:
  - v0.5.22 split `--cuda-graph-max-bs` into `--cuda-graph-max-bs-decode` / `-prefill`, and argparse then rejects the old spelling as an ambiguous prefix (e2e-mixed failed on it). When the running SGLang has no exact `--cuda-graph-max-bs`, it is mapped onto `--cuda-graph-max-bs-decode`; older images are unchanged.
  - The decode radix-cache guard builds the model config with `model_config_of` where `ServerArgs.get_model_config` is gone.
- `infera/router/kv_event/responses_input.py`: the Responses shim sets `supports_inline_system = False`, which v0.5.22 reads in `_construct_input_messages`.
- Tests:
  - `tests/engine/sglang/test_cuda_graph_max_bs_args.py` (new);
  - `tests/engine/sglang/test_decode_radix_cache_guard.py` (updated).

**Examples** (same layout as the ATOM kits)

- **Shared**: one site-specific `env.sh`; `engine/up.sh`, `smoke.sh`, `eval.sh`, `down.sh`; no automatic gates.
- **Matching InferenceX**: AgentX runner pinned to InferenceX `918524ff` with its client constraints; GSM8K through InferenceX's `run_eval` on the full 1,319 questions.
- **Common engine settings**:
  - `--disable-custom-all-reduce`: with the AITER custom all-reduce on, about 30% of GSM8K answers at concurrency 64 come out garbled under DP attention plus decode CUDA graphs (mixed README, note 6).
  - `SGLANG_OPT_USE_TOPK_V2=true`.
- **2P1D recipe**:
  - prefills: TP4 + DP4 attention, shortest-prefill-first, write-back HiCache at 250 GB per rank;
  - decode: TP8 + DP8 attention with EAGLE MTP 5/1/6;
  - the prefills carry the same EAGLE flags (note 10), as InferenceX's AMD multi-node recipe does: without them the transfer carries no draft-layer KV and the decode's draft layer reads uninitialised KV;
  - KV: Mooncake over the eight ionic rails, destination-local rail, `MC_MAX_MR_SIZE=1 GiB`;
  - router: kv-aware with prefill session affinity.

**Lint**: `infera/projection/core/projection/module_profilers/moe_mlp.py` ruff format only.

## Validation

**CI jobs reproduced on crsuse2-m2m-138 / 140** (main merged, plus the `args.py` fix):

| job | result |
|---|---|
| lint | passed |
| unit | 2,570 passed |
| unit in the v0.5.22 image | 2,359 passed |
| engine (SGLang half) | 8 files passed |
| e2e-mixed | gpt-oss-120b-tp2-ep, Kimi-K2.6-MXFP4-tp4-ep-dpattn, GLM-5.1-FP8-tp4 passed; 2 skipped by the matrix |
| e2e-disag (via the ssh transport above) | gpt-oss-120b-tp2 passed; 2 skipped by the matrix on gfx950 |

Since then only examples, docs and formatting changed. A full `pre-commit run --all-files` passes on `aab4cc36` (`cargo-fmt` and `gofmt` skipped: the branch changes no Rust or Go); the one later commit changes only `examples/sglang_2p1d_glm5.2/README.md` and `engine/worker.sh`, and passes `git diff --check` and `bash -n`.

**mixed** (C48, crsuse2-m2m-140):
- smoke 6/6;
- GSM8K at concurrency 64: 0.980;
- long context: needle 64/64;
- AgentX C48: 34,662 tok/s/GPU, interactivity p90 22.7.

**2P1D**:
- smoke 6/6;
- GSM8K (InferenceX `run_eval`, 1,319, c64): 0.9697 strict;
- AgentX C176: 35,637 tok/s/GPU, interactivity p90 35.7.

**2P1D head commit** (prefills with the EAGLE flags, run from this branch with `PREFILL_WORKERS=1`):
- smoke 6/6;
- GSM8K: 0.9704 strict / 0.9697 flexible;
- real acceptance length on GSM8K: median 4.60, against 3.86 before the change.

The reference points in the README and below were measured before the prefills carried the EAGLE flags (README note 10).

Concurrency sweep (each point on a freshly started deployment, 3,600 s, forced acceptance 3.61; full table in the README):

| shape | point | GPUs | tok/s/GPU | interactivity p90 |
|---|---|---|---|---|
| 2P1D | C176 | 16 | 35,637 | 35.7 |
| 2P1D | C128 | 16 | 26,592 | 36.7 |
| 2P1D | C96 | 16 | 23,515 | 40.8 |
| 2P1D | C64 | 16 | 18,202 | 49.4 |
| 1P1D | C88 | 12 | 27,453 | 42.8 |
| 1P1D | C64 | 12 | 23,241 | 51.4 |
| 1P1D | C32 | 12 | 14,153 | 73.2 |

**Known issues** (documented in the READMEs):
- **Custom all-reduce garbling** (mixed note 6), worked around with `--disable-custom-all-reduce`.
- **ionic MR registration** (2P1D note 8): ionic rejects a 2 GiB dma-buf registration that starts 1 GiB-aligned; `MC_MAX_MR_SIZE=1 GiB` avoids it.
- **Prefill GPU memory access fault** (2P1D note 9, open): one rare fault so far, which leaves the GPU unusable until an administrator resets it.
- **Slow VRAM release** after teardown: 10–65 min.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes

