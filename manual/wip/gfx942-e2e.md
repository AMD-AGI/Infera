# gfx942 (MI300X) E2E Matrix and Usage

This is the single WIP document for the gfx942 E2E suite. It records the current
matrix, validated results, commands, log locations, and known limitations.
Historical proposals and bring-up journals have been consolidated and removed.
The configuration source of truth remains
`tests/e2e/pd_{mixed,disag}/<engine>/matrix.py`.

Last updated: 2026-09-02.

## 1. Scope and acceptance criteria

The target hardware is an eight-GPU MI300X node (gfx942, approximately 192 GiB
per GPU).

| Dimension | PD-mixed | PD-disaggregated |
|---|---|---|
| Nodes | 1 | 2 |
| Worker roles | One worker group handles prefill and decode | Prefill and decode run on separate nodes |
| KV movement | Local | Mooncake transfers KV cache between nodes |
| RDMA requirement | Not applicable | Both observable worker transports must report `rdma` |
| Engines | SGLang, vLLM, ATOM | SGLang, vLLM, ATOM |

| Target model | Checkpoint family |
|---|---|
| GPT-OSS-120B | `openai/gpt-oss-120b` |
| GLM-5.2-FP8 | `zai-org/GLM-5.2-FP8` |
| DeepSeek-V4-Flash | MXFP4 and block-FP8 variants |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` |

A case is considered passing only when:

| Check | Passing requirement | False-green protection |
|---|---|---|
| Correctness | Counting/capital liveness and long-context retrieval pass | HTTP 200 alone is not accepted as correctness; model-generated code is never executed |
| PD workers | Both prefill and decode register successfully | A missing or dead worker fails the case |
| KV transport | Every observable Mooncake transport is `rdma` | TCP fallback cannot silently pass |
| MTP/EAGLE | Cases that request speculation produce non-zero counters | Ordinary decode cannot silently pass as MTP |
| Allocated environment | Explicit nodes, architecture, and model paths are valid | Runner-selected environment errors fail instead of skipping |

ATOM currently exposes no compatible Prometheus speculative counters. The
harness can verify its correctness and launch configuration, but cannot prove
from metrics that its draft head produced tokens.

## 2. Current validated results

| Model | Tier | Engine | Checkpoint / parallelism | Features | Result |
|---|---|---|---|---|---|
| GPT-OSS-120B | mixed | SGLang | MXFP4, TP2 | EP | **PASS** — 3 probes |
| GPT-OSS-120B | mixed | vLLM | MXFP4, TP2 | EP | **PASS** — 3 probes |
| GPT-OSS-120B | mixed | ATOM | MXFP4, TP2 | EP | **SKIP** — current gfx942 MXFP4 MoE path fails before first token |
| GPT-OSS-120B | disaggregated | SGLang | MXFP4, TP2 per leg | — | **PASS** — 3 probes + dual-leg RDMA |
| GPT-OSS-120B | disaggregated | vLLM | MXFP4, TP2 per leg | — | **PASS** — 3 probes + dual-leg RDMA |
| GPT-OSS-120B | disaggregated | ATOM | MXFP4, TP2 per leg | — | **SKIP** — same MXFP4 MoE failure |
| GLM-5.2-FP8 | mixed | SGLang | FP8, TP8/DP8 | DP attention, EAGLE | **PASS** — correctness + active MTP |
| GLM-5.2-FP8 | mixed | vLLM | FP8, TP8 | MTP3 | **PASS** — correctness + active MTP |
| GLM-5.2-FP8 | mixed | ATOM | FP8, TP8 | MTP3 | **PASS** — correctness; MTP counters unavailable |
| GLM-5.2-FP8 | disaggregated | SGLang | FP8, TP8/DP8 per leg | DP attention, EAGLE | **PASS** — correctness + dual-leg RDMA + active MTP |
| GLM-5.2-FP8 | disaggregated | vLLM | FP8, TP8 per leg | MTP off | **PASS** — correctness + dual-leg RDMA |
| GLM-5.2-FP8 | disaggregated | ATOM | FP8, TP8 per leg | MTP off | **PASS** — correctness + dual-leg RDMA |
| DeepSeek-V4-Flash | mixed | SGLang | block-FP8, TP4/DP4 | DP attention, EAGLE | **PASS** — correctness + active MTP |
| DeepSeek-V4-Flash | mixed | vLLM | MXFP4, TP4 | MTP | **PASS** — correctness + active MTP |
| DeepSeek-V4-Flash | disaggregated | SGLang | block-FP8, TP4/DP4 per leg | DP attention, EAGLE | **PASS** — correctness + dual-leg RDMA + active MTP |
| DeepSeek-V4-Flash | disaggregated | vLLM | MXFP4, TP4 per leg | MTP | **PASS** — correctness + dual-leg RDMA + active MTP |
| DeepSeek-V4-Pro | mixed | vLLM | MXFP4, TP8 | MTP | **PASS** — correctness + active MTP |
| DeepSeek-V4-Pro | disaggregated | vLLM | MXFP4, TP8 per leg | MTP | **PASS** — correctness + dual-leg RDMA + active MTP |

### Configuration details that affect correctness

| Case | Required setting | Why it is required |
|---|---|---|
| GLM-5.2 / ATOM, both tiers | `use_index_cache` and `GLM_5_2_INDEXER_PATTERN` | Prevents shared indexers with no checkpoint weights from remaining randomly initialized |
| GLM-5.2 / SGLang, disaggregated | `index_share_for_mtp_iteration=false` | Keeps EAGLE usable across the PD boundary |
| GLM-5.2 / vLLM, disaggregated | MTP disabled | MTP3 corrupted long-context output even with `disable_padded_drafter_batch=true` |
| GLM-5.2 / ATOM, disaggregated | MTP disabled | The same case hung while serving correctness probes with MTP enabled |
| DeepSeek-V4-Flash / SGLang | DP-attention variant only | The duplicate base variant was removed after DP4 validation |
| DeepSeek-V4-Pro / vLLM | `triton_unfused` packed MXFP4 path | Keeps experts packed and converts in-kernel, allowing the model to fit |

### Removed ATOM Flash cases

| Tier | Measured failure | Current action |
|---|---|---|
| mixed | The block-FP8 checkpoint has no chat template | Case removed from collection |
| disaggregated | The prefill worker exited while compiling the drafter | Case removed from collection |

### Result summary

| Category | Count | Notes |
|---|---:|---|
| Retained PASS cases | 16 | All retained target configurations have passed their required checks |
| Retained static skips | 2 | ATOM GPT-OSS mixed and disaggregated |
| Removed failed cases | 2 | ATOM DeepSeek-V4-Flash mixed and disaggregated |
| Legacy unmeasured skips | Not included | Kimi-K2.6 and GLM-5.1 remain outside this bring-up result |

SGLang uses `sgl-project/DeepSeek-V4-Flash-FP8`; vLLM uses
`deepseek-ai/DeepSeek-V4-Flash`. The removed ATOM Flash cases also used the
block-FP8 checkpoint. SGLang and ATOM do not run
DeepSeek-V4-Pro in the current target matrix: expanding its experts to FP8 needs
approximately 195.8 GiB per GPU, including runtime overhead, against
approximately 191.98 GiB usable.

## 3. Current matrix case IDs

`INFERA_E2E_K` matches the following pytest parameter IDs.

| Tier | Engine | Active target case IDs |
|---|---|---|
| mixed | SGLang | `gpt-oss-120b-tp2-ep`<br>`DeepSeek-V4-Flash-FP8-tp4-dpattn`<br>`GLM-5.2-FP8-tp8-dpattn` |
| mixed | vLLM | `gpt-oss-120b-tp2-ep`<br>`DeepSeek-V4-Pro-tp8`<br>`DeepSeek-V4-Flash-tp4`<br>`GLM-5.2-FP8-tp8` |
| mixed | ATOM | `GLM-5.2-FP8-tp8` |
| disaggregated | SGLang | `gpt-oss-120b-tp2`<br>`GLM-5.2-FP8-tp8-dpattn`<br>`DeepSeek-V4-Flash-FP8-tp4-dpattn` |
| disaggregated | vLLM | `gpt-oss-120b-tp2`<br>`GLM-5.2-FP8-tp8`<br>`DeepSeek-V4-Pro-tp8`<br>`DeepSeek-V4-Flash-tp4` |
| disaggregated | ATOM | `GLM-5.2-FP8-tp8` |

## 4. Environment preparation

Run commands from the repository root. The `mi300x-rccl` site profile supplies:

| Setting | Profile value | Purpose |
|---|---|---|
| `INFERA_E2E_MODEL_DIR` | `/iad-rccl-shared-3/liyingli/models` | Shared, read-only checkpoint root |
| `INFERA_E2E_SLURM_PARTITION` | `amd-rccl` | MI300X scheduler partition |
| `INFERA_E2E_GFX_ARCH` | `gfx942` | Selects the correct overlays and SGLang image |
| `INFERA_E2E_BUILD_ARGS` | Site apt mirrors | Makes image builds work on nodes that cannot reach the default Ubuntu mirror |
| `INFERA_E2E_SLURM_TIME` | `04:00:00` | Covers large-model cold starts and image builds |
| `INFERA_E2E_WORKER_ENV` | `MC_TE_FILTERS=mlx5_0` | Pins both PD legs to the same RDMA rail |

The model tree must follow Hugging Face repository IDs and must not contain
container-visible dangling symlinks:

| Model | Required path below `<model-dir>` | Used by |
|---|---|---|
| GPT-OSS-120B | `openai/gpt-oss-120b` | SGLang, vLLM |
| GLM-5.2-FP8 | `zai-org/GLM-5.2-FP8` | SGLang, vLLM, ATOM |
| DeepSeek-V4-Flash MXFP4 | `deepseek-ai/DeepSeek-V4-Flash` | vLLM |
| DeepSeek-V4-Flash block-FP8 | `sgl-project/DeepSeek-V4-Flash-FP8` | SGLang |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | vLLM |

GLM-5.2 uses TP8. One mixed case occupies a complete node, while a
disaggregated case occupies all eight GPUs on each of two nodes. A cold node
may spend about 40 minutes building Mooncake; the first large-model start also
includes weight loading, JIT compilation, and graph capture.

## 5. Running the suite

### Complete tiers

```bash
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e all mixed
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e all disag
```

The mixed tier needs one node. The disaggregated tier waits for and holds two
nodes.

### One engine

```bash
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e sglang mixed
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e vllm disag
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e atom disag
```

### One model

`INFERA_E2E_K` is a pytest `-k` expression. Prefer it during large-model
bring-up to avoid paying every tier's cold-start cost:

```bash
INFERA_E2E_K='gpt-oss-120b' \
  INFERA_E2E_SITE=mi300x-rccl \
  tests/run_tests.sh e2e all mixed

INFERA_E2E_K='GLM-5.2' \
  INFERA_E2E_SITE=mi300x-rccl \
  tests/run_tests.sh e2e all disag

INFERA_E2E_K='DeepSeek-V4-Flash' \
  INFERA_E2E_SITE=mi300x-rccl \
  tests/run_tests.sh e2e sglang disag

INFERA_E2E_K='DeepSeek-V4-Pro' \
  INFERA_E2E_SITE=mi300x-rccl \
  tests/run_tests.sh e2e vllm mixed
```

### Reusing an existing SLURM allocation

Pass the job ID of an existing two-node allocation. The runner attaches its
steps to that allocation, does not submit another hold job, and does not cancel
the user-owned allocation when it exits:

```bash
SLURM_JOB_ID=<jobid> \
  INFERA_E2E_SITE=mi300x-rccl \
  INFERA_E2E_K='GLM-5.2' \
  tests/run_tests.sh e2e vllm disag
```

Use only a RUNNING job owned by the current user that actually contains the
required nodes.

## 6. Logs and result inspection

Persistent worker logs for non-CI runs are written under:

```text
/tmp/infera-e2e-logs
```

| Tier | Filename pattern | Persistence behavior |
|---|---|---|
| mixed | `infera-e2e-<engine>-<case-id>-<timestamp>.log` | Written directly to the mounted host log directory |
| disaggregated | Names contain `infera-e2e-disagg-` and the router/prefill/decode role | Teardown collects remote container logs for both passing and failing runs |

Useful checks:

```bash
ls -lt /tmp/infera-e2e-logs
rg -n 'protocol: rdma|transport.*rdma|prefill|decode' /tmp/infera-e2e-logs
rg -n 'spec.*draft|spec.*accept|Loading drafter|Capture draft' /tmp/infera-e2e-logs
rg -n 'ERROR|Traceback|OutOfMemory|ReadTimeout' /tmp/infera-e2e-logs
```

The pytest output directly reports:

| Output marker | Meaning |
|---|---|
| `[e2e disagg transport] prefill @ <node>: rdma` | Prefill used RDMA |
| `[e2e disagg transport] decode @ <node>: rdma` | Decode used RDMA |
| `[e2e mtp] ... active ...` | The speculative path produced tokens |
| Correctness probe PASS/FAIL | The generated answer passed or failed semantic validation |

Do not treat HTTP 200 as correctness evidence. A model can return incorrect
text with HTTP 200; the probe assertions are authoritative.

## 7. Known limitations and upgrade checks

| Area | Current limitation | Upgrade or regression action |
|---|---|---|
| ATOM GPT-OSS | Current ATOM images have an upstream MXFP4 MoE defect; this is not a general gfx942 ATOM limitation | After an ATOM/aiter upgrade, retest mixed before restoring disaggregated coverage |
| ATOM MTP metrics | No compatible speculative Prometheus counters | Treat drafter logs as supporting evidence, not an activity or acceptance assertion |
| vLLM GLM-5.2 PD + MTP | MTP3 corrupts long-context output even with the unpadded drafter path | Keep MTP off; rerun the long-context probe after vLLM upgrades |
| SGLang image | gfx942 requires `Dockerfile.sglang.gfx942` | Do not substitute the default gfx950 SGLang image |
| RDMA fabric | Both PD legs must use the same rail | Keep `MC_TE_FILTERS=mlx5_0` for this cluster |
| Matrix evolution | MTP depth and individual cases may change with engine images | Unit-test generic expansion, overlays, and guards instead of freezing temporary values or exact case counts |

## Related documentation

`manual/wip/mi325-deepseek-v4.md` is intentionally retained. It documents the
broader DeepSeek-V4 runtime support contract on gfx942 and is not replaced by
this E2E operations guide.
