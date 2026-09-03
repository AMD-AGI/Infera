# Notes — gotchas, wrong turns, and the things that cost time

The root cause has its own file: `results/root_cause.md`, with the engine-side
fix written up in `patches/patch_glm5next_shared_experts_fusion_quant_guard.py`.
This file is everything else a reproducer needs to know.

---

## 1. The method that found it: one variable per rung

**What.** After three bring-up rounds that each moved more than one variable and
each failed identically, the work stopped and wrote `PLAN.md`. Its §3 is a
ladder from the configuration most likely to work — the vendor's — back toward
the one we must ship, changing exactly one thing per rung.

| rung | image | engine source | launcher | isolates | result |
|---|---|---|---|---|---|
| 0 | vendor `rocm724` | bind-mounted `c821c425` | bare `launch_server` | is the vendor recipe reproducible here at all | **PASS** (+ the flag) |
| 1 | ours `rocm720` | bind-mounted, same | bare | **the base image / aiter** | **PASS**, 220 s |
| 2 | ours `glm53-c821c425` | baked overlay | bare | baked vs bind-mounted | **PASS**, 220 s |
| 3+4 | ours | baked | **infera** + etcd + kv-aware router | the infera wrapper and the MIX topology | **PASS**, worker 210 s |
| 5 | ours | baked | full MIX | decode CUDA graphs on | **PASS** — the shipped configuration |

Evidence: rungs 0-2 and 3+4 are written up in `results/ladder.md` and
`results/rung0_nosef.md`, with logs under `logs/n0433/`. **Rung 5 is not in
`ladder.md`** — it ran after that file was written. Its evidence is
`logs/n0433/mix_up_graphs.log.gz` (`worker serving after 240s`, `router
healthy`) and the entire n01-33 run, whose decode lines all carry
`cuda graph: True`.

**Why it matters.** Every rung passed *once the flag was added*, which means
**none** of these is a variable: rocm720 vs the vendor's rocm724, bind-mount vs
baked overlay, bare `launch_server` vs the infera wrapper, etcd, or the kv-aware
router. The single delta from the vendor's published recipe is one flag. Without
the ladder, "our image is different from the vendor's" would still be an open
suspicion today.

**How to reuse.** `scripts/rung.sh`, `RUNG=0|1|2`. Full evidence in
`results/ladder.md` and `results/rung0_nosef.md`.

---

## 2. Two corrections to record, because they cost time

### (a) The `config.json` exclude-list hypothesis was WRONG

**What was believed.** That `quantization_config.exclude` omitted the routed
experts of layers 3 and 5, while `mixed_precision_correction.json` records
`bf16_expert_layers: [3, 5, 6]` — a real metadata gap that would explain a BF16
tensor reaching a packed buffer.

**Why it was wrong.** The module-level entries `model.layers.{3,5,6}.mlp.experts`
**are present** in `exclude`. The grep that "proved" they were missing used a
regex with a trailing dot, which matched only the deeper per-expert entries
(`...experts.0.down_proj`) and missed the module-level ones.

**What it cost.** A round of config surgery that changed nothing — the same
error at the same shard. That null result is itself useful and is why the
framing moved on, but it was avoidable.

**How to avoid.** When a hypothesis rests on "X is absent from a list", print
the matching lines rather than the match count, and print the regex.

### (b) The 350 tok/s decode-probe reading was a client-side artifact

**What was measured.** `scripts/decode_probe.sh` at concurrency 32 reported
~350 tok/s.

**Why it was wrong.** The engine's own log showed **~2398 tok/s** at
`#running-req: 32, #queue-req: 0` over the same window — the queue was empty, so
the engine was not the limit. The probe fans out with `curl` subshells and
parses JSON serially; at 32-way that harness is the bottleneck, not the server.
`sglang.bench_serving` later measured **1795 tok/s** at 1k/1k.

**Do not publish the 350.** It is recorded here only so nobody rediscovers it
and believes it.

**How to avoid.** Cross-check any client-side throughput number against the
engine's own `gen throughput (token/s)` line and its `#queue-req`. If
`#queue-req: 0`, the client is the limit.

---

## 3. Environment traps that will bite a reproducer

### `/apps/data/models` is its own NFS mount

**What.** `/apps/data/models` is a symlink to `/perf_apps/data/models`, on a
separate mount. Binding the parent (`-v /apps:/apps`) gives the container a
*different*, empty-looking `/apps/data/models`, and every symlink into it is
dangling.

**How it surfaces.** Far downstream, as **`Unrecognized processing class`** —
because `config.json` happens to be the one real file left and the
tokenizer/processor files have vanished. Nothing in the message says "mount".

**Fix.** Bind `/apps/data/models` explicitly. `scripts/mix_up.sh` and
`scripts/rung.sh` both do; `mix_up.sh` additionally exposes `WEIGHTS_MOUNT` for
a second bind when the model dir is a symlink farm.

### Port 2379 was held by a foreign host etcd

**What.** On one node, `2379` was already bound by an `etcd` belonging to
somebody else, on the host, not in a container.

**Fix.** Ours moved to **12379** (`ETCD_PORT` default in `mix_up.sh`). A foreign
process is never killed to free a port.

### `reset_gpus.sh` — the gate, and why it must not kill blindly

**What.** `docker rm -f` returns when the container is gone, not when the kernel
has reclaimed the GPU allocations. Starting the next worker too early aborts
distributed bootstrap with

    The memory capacity is unbalanced. Some GPUs may be occupied by other
    processes. pre_model_load_memory=194.9 ...

which reads like a config or model problem and is really a stale process. That
cost a full debug round once, so the gate is real.

**Why it is careful.** The script it was adapted from killed **every** KFD
process on the node. These are shared hosts: n04-33 carried another user's
`torchtitan-job27029` (up for days) and the team's own big-model track on GPUs
4-7. So ownership is now resolved **per PID** from `/proc/<pid>/cgroup`: a PID
inside a container matching `$OWN_CTR_RE` is killed; anything else is never
touched, and if it still holds a GPU we need, the script **aborts and says whose
it is**. An abort there is the correct outcome — it means the node is busy, not
that the script failed.

### The first node was lost mid-project

n04-33 filled with other users' work partway through, so the whole result was
re-established from scratch on **n01-33** — fresh image build, fresh bring-up,
fresh sweep. That is why `logs/` is split by node and why `environment.md` lists
two different image ids for the same tag.

---

## 4. Flags and env vars that are load-bearing and do not look it

Each of these was verified, not assumed. Full rationale lives in
`scripts/mix_worker.sh`'s header.

| setting | what happens without it |
|---|---|
| `--disable-shared-experts-fusion` | **model does not load.** `results/root_cause.md`. |
| `SGLANG_USE_AITER=1` | server starts, answers correctly, and is **4.3-5.4× slower** with nothing in any log saying so. The health check is the presence of the two mHC lines, not the absence of errors. |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` | vendor-set for this checkpoint; not present in the FP8-Flash recipe. Not verified independently here — carried because the card sets it. |
| `--quantization quark` | the Flash card passes it explicitly (the big GLM-5.3-MXFP4 card says the loader auto-detects). |
| `--moe-runner-backend aiter` | native AITER FP4 MoE kernels (`torch.float4_e2m1fn_x2`). With `triton` the checkpoint is dequantised to BF16 GEMMs — still serves, much slower. |
| `--dsa-prefill-backend` / `--dsa-decode-backend` | the flags are `--dsa-*`. GLM-5.2 used `--nsa-*`; copying that recipe forward gets unknown-flag errors. |
| `--random-range-ratio 1.0` (sweep) | the default draws prompt lengths uniformly, so the percentiles mix request sizes. A fixed-length sweep wants a delta, not a distribution. |
| `--temperature 1.0 --top-p 0.95` (sweep) | the checkpoint's own `generation_config`, deliberately not greedy: at temperature 0 this reasoning model falls into repetition on a long prompt. |
| `PYTHONHASHSEED=0` | stable block hashes → stable kv-aware routing keys across restarts. |

**Deliberately off, and why:**

- **MTP / speculative decoding.** Not validated for this model on ROCm; the AMD
  lane disables it. Do not add `--speculative-*`. (The NextN draft layer also has
  the unguarded BF16-experts problem — `results/root_cause.md`, last section.)
- **hicache / kvd.** Only `patch_hicache_rocm_host_alloc.py` is in this image;
  the staged-write-back gate was deliberately excluded, so hicache is unverified
  here. Re-derive both gates before turning either on.
- **Prefill CUDA graphs.** Only decode graphs are captured. Prefill is where the
  DSA/KDA shape variance lives, and decode-only is what upstream validated on
  gfx950.

---

## 5. Two memory pools, not one

This model is **hybrid**: `linear_attention` (KDA) layers plus
`deepseek_sparse_attention` layers. It therefore keeps a **KDA state pool** in
addition to the paged KV pool, and both show up per decode line:

    Decode batch, #running-req: 24, #full token: 446208, full token usage: 0.06,
    mamba num: 96, mamba usage: 0.04, cuda graph: True,
    gen throughput (token/s): 2334.54, #queue-req: 0

A decode line with only `full token usage` means you are not looking at this
model. If `max_running_requests` is clamped at startup, **suspect the KDA pool
first**: raise `--mamba-full-memory-ratio` or pin `--max-mamba-cache-size`. Do
not override `linear_lower_bound` via `--json-model-override-args`.

Measured allocation here: `max_mamba_cache_size: 2371`, `conv_state 2.77 GB`,
`ssm_state 78.76 GB`, leaving `available_gpu_mem 54.35 GB` and
`max_total_num_tokens 7650368`.

---

## 6. On the GLM-5.2 comparison

`results/fixlen_vs_glm52.md` carries the numbers and the three caveats. The
short version, because the ratio will get quoted otherwise:

1. **Different models** — GLM-5.3-Flash is 320 B / 18 B active with hybrid KDA;
   GLM-5.2 is 744 B with uniform MLA+DSA. A smaller, sparser model being faster
   is expected, not a finding.
2. **Half the hardware** — ours is TP4 on 4 GPUs, the baseline TP8 on 8. Per GPU
   the gap is larger, which overstates it the other way.
3. **Ours is the less-optimised configuration** and still wins: the baseline runs
   DP-attention, EAGLE MTP and kvd; ours runs none of them.

Point 3 is the one comparison-independent statement worth keeping: **there is
headroom we have not touched.** The mission's fixlen-alignment requirement is
written against **GLM-5.3 (big)**, which is the same architecture as GLM-5.2 and
is therefore the real apples-to-apples comparison. That belongs to the big-model
track, not this packup.

`results/fixlen_p90.csv` (isl 15500 / osl 3300) completed all four arms — conc
1 / 8 / 16 / 24 — while this packup was being written; the file here is the
final one. **No p90 comparison against GLM-5.2 was made**; only the p50 arm has
a baseline to compare against.
