This PR has three parts:

1. **ATOM / aiter patches** (7): shorter decode steps, and no more GC stalls in ATOM's API process.
2. **Router** (Python and Rust): the ATOM PD decode takes the prompt token ids the prefill returned; `/metrics` no longer scrapes ATOM workers.
3. **GLM-5.2 4P4D kit**: defaults follow ATOM's recipe and InferenceX PR #4045, with a Mooncake Store CPU tier.

Result: with the new kit defaults, C56–C112 run 5.0%–23.8% above InferenceX PR #4045, and the peak moves from 42.5K at C96 to 48.2K tok/s/GPU at C112.

## 1. ATOM / aiter patches (`deploy/docker/patches/atom_nightly/`)

These apply only with `ATOM_NIGHTLY=1` (`rocm/atom-dev:nightly_202610131505`: ATOM 661ef766, aiter 9205e1059). Like the existing scripts there, they are strict: a drifted anchor fails the build. `patch_aiter_custom_alltoall.py` adds both aiter kernels below and rebuilds `module_custom_all_reduce` once, after clearing its JIT build directory. `check_dcp_merge.py` checks both kernels against RCCL on GPUs (bit-exact all-to-all, merge up to FP8 rounding, eager and inside a CUDA graph); run it on 4 or 8 GPUs after moving the base image.

| patch                                                                                       | change                                                                                                                                                                                                                                | effect (MI355X, GLM-5.2 decode TP4 + DCP4)                                                                                                                |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patch_atom_api_gc_stalls.py`                                                             | ATOM API process: when serving online, finished sequences are no longer queued on `CoreManager.outputs_queue` (nothing reads it online), and the per-request context gauges are opt-in (`ATOM_METRICS_PER_REQUEST_CONTEXT=1`)     | Both API processes spent ~45% of the wall clock in gen-2 GC passes that grew with every completed request. C64 throughput +1.9%, mean TTFT 2.85 → 2.04 s |
| `patch_atom_dcp_topk_custom_ag.py`                                                        | The sparse indexer's DCP top-k exchange goes through `dcp_all_gather` (aiter's IPC all-gather instead of pynccl)                                                                                                                    | 1.3 MB/rank 54 → 31 µs, 24 per step                                                                                                                     |
| `patch_aiter_mla_bmm_configs.py`                                                          | Per-M Triton configs for the MLA absorbed bmm (q-up N512/K192, v-up N256/K512), for M ≤ 512, swept on MI355X with outputs checked. The stock configs take a masked slow path whenever M is not a multiple of 64                      | Graph 40: q-up 19.1 → 8.4 µs, v-up 15.2 → 8.1 µs, about −1.4 ms per step                                                                             |
| `patch_aiter_custom_alltoall.py` (first half) + `patch_atom_dcp_custom_a2a.py`          | New aiter `alltoall_vec` on the custom all-reduce's registered buffers and CUDA-graph path; the DCP merge uses it when it applies and RCCL otherwise                                                                                | Bit-identical to RCCL; 5.3 MB/rank 56 → 34 µs; −1.0…−1.6 ms per step                                                                                 |
| `patch_aiter_glm52_decode_gemm_rows.py`                                                   | 27 tuned GLM-5.2 FP8 GEMM rows for M 144–240, only those at least 5% faster than the current path, as one more `model_configs` table; the build fails if aiter's tables already hold one of its keys                               | qkv_a at M 160 14.9 → 11.5 µs; −0.9 ms per step at B33/B36                                                                                             |
| `patch_aiter_custom_alltoall.py` (second half) + `patch_atom_dcp_custom_a2a_combine.py` | New aiter kernel `alltoall_combine_quant`: reads this rank's heads straight from the peers' registered buffers, merges them with the LSE weights and quantizes per token to FP8, so the received copy and the unpack launch go away | Matches the RCCL path up to FP8 rounding (4.6e-6 of codes differ); 160 rows 44.7 → 38.1 µs per layer; −0.56 ms per step on average                     |

Both Python entries of the all-to-all accept world sizes 2, 4 and 8 only, the sizes the kernels are instantiated for.

In total, at B40 with ~101K context, the patches in the first five rows take a decode step from 42.77 ms to about 40 ms (−6…−7%), and the one-kernel merge takes off another 0.2–0.6 ms.

## 2. Router (`infera/`, `rust/router/`)

- **The ATOM PD decode reuses the prefill's prompt token ids** (Python and Rust):
  - The prefill leg asks for `return_token_ids`, and the hand-off puts the returned ids in the decode's `kv_transfer_params['prompt_token_ids']`.
  - ATOM's decode takes them as the prompt instead of rendering the chat template and tokenizing again (~0.25 s per request at 130K tokens).
  - A prefill that returns no ids, an empty list or negative ids, such as an older ATOM without the field, leaves the decode tokenizing as before.
  - Effect: C64 throughput +0.5%, TTFT p50 −0.09 s.
- **The frontend `/metrics` no longer scrapes ATOM workers** (Python and Rust): the frontend reads nothing from ATOM's exposition, so each scrape only cost ATOM's API process one more `/metrics` render. sglang and vllm are unchanged.

## 3. GLM-5.2 4P4D kit (`examples/atom_4p4d_glm5.2/`)

- **Decode memory**: mem util 0.94, `--max-num-batched-tokens 2048` (`DECODE_MAX_BATCHED_TOKENS`) and `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`.
  - KV blocks go from about 160K (previous settings) to about 198.5K (~12.7M tokens over the four DCP ranks).
  - The AgentX decode-prefix cliff moves from about C85 to about C100.
- **Mooncake**: `MC_NUM_QP_PER_EP=1` on every Mooncake peer (P, D and the Store), and ATOM's default 16 send workers (`PREFILL_SEND_WORKERS` is empty by default). With two QPs, 16 workers used to hang an RDMA write; with one QP the hang did not recur in 10 deployments.
- **CPU tier**: `PREFILL_CPU_TIER` is `store` (default), `lmcache` or `off`.
  - With `store`, the new `engine/store.sh` starts a master and two owners, one per NUMA node with `STORE_OWNER_GB=768` GiB each, backed by 2 MiB pages through glibc's THP malloc (no root needed).
  - The four PP stages share that pool, each through its own GPU's NIC, with one load worker.
  - `store.sh` first checks THP and the free memory of both NUMA nodes, and `up.sh` checks that the decode's batch budget covers (K+1) × 2 × CONC tokens.
  - The DCP prefill (`PREFILL_PP=1`) cannot use the Store: ATOM's Store offload refuses DCP, so `up.sh` stops that combination and the DCP prefill runs with `PREFILL_CPU_TIER=lmcache`.
- **Other**:
  - `TOPK_FORCE_PATH=one` and `--timeout-keep-alive 900` on both legs. The Rust router reuses idle connections, and ATOM's 5 s default closes them under it.
  - `ROUTER_BACKEND` now defaults to `rust`.
  - The compile-cache volume is keyed on max-num-batched-tokens too (`<CTR>-cache-<role>-<image-id>-mbt<N>`), because ATOM's compile cache is not.
  - `agentx.sh` records the CPU tier in the AgentX metadata: `KV_OFFLOAD_BACKEND=mooncake` for the Store, `lmcache` for LMCache, no offload for `off`.
- **New knobs**: `PREFILL_EXTRA_ENV`, `DECODE_EXTRA_ENV`, `DECODE_MAX_BATCHED_TOKENS`, `PREFILL_CPU_TIER`, `STORE_OWNER_GB`, `STORE_OWNER_DEVICES`, and the Store ports.
- **Not taken from the recipe / PR #4045**:
  - Decode graph sizes: here every 4 up to 2 × CONC, there every 8 up to 256.
  - max-num-seqs: here 2 × CONC, there 512.
  - `OFFLOAD_PROFILE=1` (logging only).

## 4. Behavior changes / compatibility

| change                                                                             | impact                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The `ATOM_NIGHTLY=1` image carries 7 more patches                                | Applies to every model on that image. The GC fix applies to all of them; the DCP communication changes only touch MLA + DCP decode; the bmm configs and GEMM rows only match GLM-5.x shapes        |
| `atom:{prefill,decode}_request_context_tokens` are no longer recorded by default | Set `ATOM_METRICS_PER_REQUEST_CONTEXT=1` when you need them (for example, for profiling)                                                                                                         |
| ATOM PD request bodies                                                             | The prefill gets `return_token_ids: true`; the decode's `kv_transfer_params` gets `prompt_token_ids`. ATOM's request models ignore unknown fields, so an older ATOM falls back to tokenizing |
| Router `/metrics`                                                                | ATOM workers are no longer scraped (nothing was read from them)                                                                                                                                    |
| Kit defaults                                                                       | The Store needs 768 GiB free per NUMA node and THP `madvise` or `always`; the DCP prefill needs `PREFILL_CPU_TIER=lmcache`. Decode 0.94 / 2048, one QP + 16 send workers, Rust router        |
| Hosts where Store reads into HBM do not land                                       | On some MI355X hosts, Store reads into HBM do not land and the prefill's startup probe fails ("the chunk read back differs"). Use `PREFILL_CPU_TIER=lmcache` on such hosts (README gotcha 14)    |

## 5. Validation

**Unit tests and checks**:

- CI has not run the tests yet: its lint job fails on main's `infera/projection/core/projection/module_profilers/moe_mlp.py` (unformatted since #205), and the test jobs wait on lint. Run locally on the PR head with CI's commands: `pytest -m "not slow and not integration"` 2,791 passed, 148 skipped; `cargo fmt --check`, `cargo clippy --all-targets -D warnings` and `cargo test` pass; pre-commit passes on every file this PR touches; shellcheck on the kit is clean.
- The 9 nightly patches apply to the pinned upstream sources in the Dockerfile's order and in reverse, a second run is a no-op, and `module_custom_all_reduce` rebuilds.
- `check_dcp_merge.py` passes 24/24 on 4 and on 8 MI300X GPUs, and fails with the tolerance set to 0.
- The image of `41bd647a` builds on four different 8 × MI355X machines, and the results below come from it (`96a3af86` after it changes only kit scripts, `check_dcp_merge.py`, docs and comments).

**Correctness** (real MTP acceptance):

| configuration        | smoke                        | GSM8K (200 questions) | MTP acceptance |
| -------------------- | ---------------------------- | --------------------: | -------------: |
| New defaults (Store) | 6/6 (with long-prefix reuse) |                 0.975 |          0.799 |

**Performance**: AgentX with the InferenceX CI client, a cold start, warm-up and a 3600 s measured window per point, and forced MTP acceptance (K3, AL 2.99). GLM-5.2-MXFP4 on one 8 × MI355X node. Throughput is per GPU over all 8 GPUs and counts cache-hit input tokens. Every point had 0 errors and no JIT compile inside the measured window.

The new defaults (Store, Rust router; the four points split over three machines), against InferenceX PR #4045's 1P1D results (from that PR's own CI run, computed the same way from its published AgentX JSON; PR #4045 ran on other machines):

| CONC |  input tok/s/GPU | PR #4045 |            vs PR | output tok/s/GPU | P90 interactivity (this PR / #4045) | TTFT p50 s (this PR / #4045) | decode prefix hit | requests |
| ---: | ---------------: | -------: | ---------------: | ---------------: | ----------------------------------- | ---------------------------- | ----------------: | -------: |
|   56 |           32,925 |   30,834 |  **+6.8%** |            246.2 | 64.0 / 58.4                         | 0.84 / 1.45                  |             92.3% |    7,049 |
|   72 |           40,075 |   38,180 |  **+5.0%** |            321.9 | 52.1 / 48.9                         | 1.05 / 1.93                  |             84.1% |    9,477 |
|   96 |           46,234 |   42,523 |  **+8.7%** |            387.3 | 45.6 / 42.1                         | 1.66 / 3.73                  |             52.9% |   11,291 |
|  112 | **48,154** |   38,904 | **+23.8%** |            406.0 | 42.1 / 43.0                         | 3.64 / 14.78                 |             25.1% |   12,027 |

- C56 and C72 sit before the cliff; most of their gain comes from the decode patches of section 1 (a shorter ITL).
- C96 and C112 add the larger decode KV pool and the Store, whose asynchronous RDMA loads do not block the prefill. The peak therefore moves from PR #4045's C96 to beyond C112, and throughput is still rising from C96 to C112.
- For reference, main's kit README reports 30,116 tok/s/GPU at C96 (LMCache, Python router, measured on another machine).

To reproduce a point (cold start; `agentx.sh` runs InferenceX `918524ff`'s client, not the CI client used above): `CONC=96 MTP_AL=2.99 SESSION_AFFINITY=off ROUTER_EXTRA_ARGS="--breaker-failure-threshold 0" PREFILL_EXTRA_ARGS="--spec-decode-acceptance-length 2.99" bash engine/up.sh && bash engine/agentx/agentx.sh && bash engine/down.sh`.

## 6. Known limitations / follow-ups

- **The Store does not work on some hosts**: on some MI355X hosts every Store get into HBM leaves zeros, on one node or across nodes, with peer memory or DMA-BUF; other hosts are fine, and P→D is unaffected. Such hosts need LMCache, whose synchronous loads take 47–52% of each PP stage's time, so C96 / C112 drop to 30.7K / 23.3K and the peak is C72 (40.3K).
- **The patches are pinned to nightly_202610131505**: when the base moves, the strict anchors fail the build and each script has to be re-checked. Neither the aiter side (custom all-to-all, merge kernel, two tuned tables) nor the ATOM side has been filed upstream yet; that has to happen before this configuration can go to InferenceX.
- **Real MTP acceptance** varies between 0.59 and 0.82 across machines and configurations, cause unknown. The performance numbers above all use forced AL 2.99.
- **DCP prefill** (`PREFILL_PP=1`) cannot use the Store, since ATOM's Store offload refuses DCP, so it runs with LMCache. It was validated with LMCache and the previous decode settings and has not been re-run with the new decode settings.
- **Session headers**: the Rust router does not forward `x-session-id` / `x-correlation-id` to the engines (the Python router does).
- **One decode abort, not reproduced**: in one C96 run the decode's rank 0 aborted with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` while it computed the last 1,390 tokens of a 568K-token request itself. The rerun on another machine went through a 615K-token request of the same kind without errors; cause unknown.

## 7. Merging

Let CI run once main's lint passes again (the `moe_mlp.py` formatting fix goes to main separately). The 17 commits are self-contained and all carry `Signed-off-by`, so the history can be kept (merge or rebase). Before merging, fix the message of `14fb5592`: it cites the pre-rebase hashes `f7e46219` and `91910e57`, which are now `949a8f5d` and `8dd1f8b7`. If the PR is squashed instead, the squashed message needs `Signed-off-by` too.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
