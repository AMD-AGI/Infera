# GLM-5.2 "degenerate output under concurrency" — FALSIFIED

**Date:** 2026-07-29
**Cluster:** crsuse spur (`amd-spur` / `amd-burst-qos`)
**Verdict:** **Not an engine bug.** The symptom was produced by testing the
model wrong — raw base-LM completion with no chat template, at a forced
`temperature=0` the model was never meant to run at.

---

## 1. What was being investigated

While validating the PD + DP-attention + MTP fixes (see the sibling kit
`glm52.mxfp4.spur.mooncake.packup_20260728/`), ~1–3 % of responses under
`concurrency=128` came back as a repeating loop:

```
1.1.1.1.1.1.1.1.1.1.1.1. ...
1.2.3.4.5.6.7.8.9.10.11. ...
2.3.3.3.3.3.3.3.3.3.3.3. ...
```

HTTP 200, correct token count, no exception, no `KVTransferError`. This was
treated as a concurrency-induced corruption bug and chased for several hours
across MTP / PD / custom-all-reduce / DP-attention / quantization.

**The question that ended it** (asked by the user): *is a repeating loop not
simply what low-temperature decoding does?* — plus *does the chat template
really not matter?* and *does `temperature` even take effect?*

---

## 2. The decisive result

Three runs, 128 concurrent requests, 512 max tokens each, same prompts:

| # | model | endpoint / template | sampling | looping |
|---|---|---|---|---|
| 1 | **MXFP4** | `/v1/chat/completions` (template applied) | `t=1.0, p=0.95` | **0 / 128** |
| 2 | **FP8** | `/v1/chat/completions` (template applied) | `t=1.0, p=0.95` | **0 / 128** |
| 3 | **FP8** | `/generate` raw text (no template) | `t=0` (greedy) | **1 / 128 (0.8 %)** |

Raw data: `results/RESULT{1,2,3}_*.jsonl` (full text of every response).

Both models, at the model's own recommended sampling with the chat template
applied, produce **zero** degenerate outputs at concurrency 128. The official
FP8 build reproduces the failure under the *wrong* configuration exactly as
MXFP4 did — so quantization was never involved either.

### The two mistakes

**(a) The chat template was skipped.**
GLM-5.2 is an instruct model. Both checkpoints ship `chat_template.jinja`,
which begins:

```
[gMASK]<sop>
... <|system|>Reasoning Effort: {{ effective_reasoning_effort }}
```

All prior testing posted a raw `text` field to `/generate`, i.e. base-LM
completion with none of that. The prompt was:

```
"Explain quantum computing in detail, part 31."
```

A base LM continuing a string that ends in `part 31.` by emitting `1.2.3.4...`
and sticking there is ordinary behaviour, not a defect.

With the template, the same prompt returns structured reasoning:

```
'1.  **Understand the Goal:** The user wants a detailed explanation of quantum
computing, specifically...'
```

**(b) `temperature=0` was forced, overriding the model's own recommendation.**
Both `generation_config.json` files say:

```json
{ "temperature": 1.0, "top_p": 0.95 }
```

and sglang's default `--sampling-defaults model` already honours that. The test
harness passed `temperature: 0` explicitly and overrode it. Degenerate
repetition under greedy decoding is a well-known property of neural LMs
(Holtzman et al. 2019, *The Curious Case of Neural Text Degeneration*), so the
harness was manufacturing the symptom it then reported.

### Sampling knobs verified working

Measured on the live FP8 server, 4 samples per setting, same prompt:

| setting | distinct outputs / 4 | first 50 chars |
|---|---|---|
| `t=0` | 4 | `' Quantum Fourier Transform\nQuantum Fourier Trans'` |
| `t=1.0` | 4 | `' Quantum Fourier transform\n\n2023-09-29 21:25:01 '` |
| `t=1.0, p=0.95` | 4 | `' Provide examples\nQuantum Error Correction: Shor'` |
| `t=2.0` | 4 | `' Class Hours strong traveling熠.notes miserable_'` |
| `t=1.0, top_k=1` | 4 | `' Quantum algorithms\nQuantum algorithms are compu'` |

`t=2.0` degrades into noise while `t=1.0,p=0.95` stays fluent — the knobs are
plumbed through and take effect. **`temperature`, `top_p` and `top_k` all work.**

An earlier claim that `eagle_utils.py:620`'s `or _is_hip` silently discards
sampling parameters was over-generalised: that line is in the **spec-decode
verify** path, which a no-MTP server never executes. It says nothing about the
ordinary sampler.

### Non-determinism at `temperature=0` is expected, not a defect

`t=0` still gives 4 distinct outputs from 4 identical requests. That is
documented sglang behaviour, not corruption:

```
enable_deterministic_inference:
    "Enable deterministic inference mode with batch invariant ops."
    default: False
```

Without it, batch composition changes reduction order and can flip an
`argmax`. All servers here ran with it `False`.

---

## 3. Environment

Captured from the live containers by `scripts/envcap.sh`, not from notes.

### Hardware (per node)

| | |
|---|---|
| node | `crsuse2-m2m-084` (also used: `-029`, `-106`, `-208`, `-244`, `-215`, `-046`) |
| GPU | **AMD Instinct MI355X** ×8, model `0x75a3`, 288 GiB HBM each |
| CPU | 236 logical cores |
| RAM | 2751 GiB |
| kernel | `6.8.0-107-generic` |
| KV NIC | `mlx5_0`, fw **28.43.3608**, board `MT_0000001045` |
| other NICs | `ionic_0..7`, fw **1.117.1-a-63** (no ODP — unusable for dmabuf) |

Scheduler is **Spur**, not stock Slurm: `spur exec <job> <cmd>`; ssh to compute
nodes is prohibited.

### Software

| | |
|---|---|
| image | `infera.yihou.sglang.1.0` |
| image id | `sha256:347bcd45da0dee1bc87f10c348e41f20ed56e11d23f9fead164cdef4e51dc970` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` (Ubuntu 22.04.5) |
| image tar | `/home/yihou/infera.yihou.sglang.1.0.tar` (NFS) |
| sglang | `v0.5.15.post1`, commit **`0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`**, editable at `/sgl-workspace/sglang` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| HIP | `7.2.26015-fc0010cf6a` |
| tilelang | `0.1.7.post3+cuda.gita55a8230` |
| sgl_kernel | `0.4.4` |
| triton | `3.6.0` |
| transformers | `5.12.1` |
| aiter | at `/sgl-workspace/aiter` (no `__version__`) |

**Source tree was clean upstream for all three results** — verified before the
runs, all patch markers 0:

```
_q_mqa  GLM52_BUG6  _glm52_match_page_table_rows  GLM52_BUG2  VARIANT_B   -> 0
```

### Models (absolute paths, NFS, not in git)

| path | size | quant | arch |
|---|---|---|---|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | 408 GB | quark / MXFP4 | `GlmMoeDsaForCausalLM` |
| `/shared_nfs/huggingface_models/zai-org/GLM-5.2-FP8` | 704 GB | fp8 e4m3, block 128×128 | `GlmMoeDsaForCausalLM` |

Both: 78 layers, 256 experts, hidden 6144, `index_topk=2048`,
`generation_config.json` = `temperature 1.0 / top_p 0.95`,
`chat_template.jinja` present.

Note: at 408 GB the MXFP4 weights **do not fit one 288 GiB GPU** — TP=1 is
impossible; TP8 was used throughout.

### Credentials

**None required.** Models and image tar are on NFS; no HF token, no registry
login, no API key. `DOCKER_CONFIG=/tmp/dockercfg` must be exported before any
docker call (docker 29 buildx plugin discovery fails on the default path) but
that directory holds no secrets.

---

## 4. Exact reproduction

See **[REPRODUCE.md](REPRODUCE.md)** for copy-pasteable commands.

---

## 5. Documents

| file | contents |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | exact commands, start to finish |
| [PITFALLS.md](PITFALLS.md) | the wrong turns, why each happened, and what invalidated it |
| [RETRACTIONS.md](RETRACTIONS.md) | every claim withdrawn, and what remains standing |
| `results/*.jsonl` | full response text for all 384 requests |
| `scripts/*` | every tool used, including the ones whose results were discarded |

---

## 6. What still stands

The **crash and deadlock** fixes from the 2026-07-28 kit are unaffected — those
were hard failures (`AssertionError`, `Expected lengths.size(0) == B`, all 8
schedulers dead, first-request deadlock) with stack traces, entirely
independent of sampling configuration:

- Bug 1 — HIP/aiter DP-padded row count in `dsa_indexer._get_topk_paged`
- Bug 2 — D2H sync on a rank-divergent branch (PD+MTP first-request deadlock)
- Bug 5 — `page_table` rows vs `topk_indices` rows in `dsa_backend.forward_decode`
- Bug 6 — the Bug-1 slice skipped DP-idle ranks (`0 < q_offset`)

PD + DPA + MTP went from "deadlocks on the first routed request" to 640/640
requests at concurrency 128. That result is measured and holds.

**What does not stand** is the entire "~2 % degenerate output" investigation
layered on top of it. See [RETRACTIONS.md](RETRACTIONS.md).
