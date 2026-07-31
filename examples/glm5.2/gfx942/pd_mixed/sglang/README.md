# GLM-5.2-FP8 aggregated on MI325X — SGLang (runnable scripts)

One SGLang server holding the whole model on a single 8x MI325X (gfx942) node:
GLM-5.2-FP8 weights, fp8 KV cache, DSA sparse attention on the tilelang backends, and
optionally MTP (EAGLE against the checkpoint's own nextn layer) and DP-attention.

**These are in-container scripts.** The platform this was developed on is a Kubernetes
web console: you request GPU resources there, name the image in the request, and it hands
back a pod that is already running that image. There is no docker socket inside it, so
nothing here does `docker run` — you open a shell in the pod you were given and run the
scripts in it. That is the difference from
[`examples/sglang_1p2d_kimi2.6/`](../../../../sglang_1p2d_kimi2.6/), whose scripts start
their own containers. To reproduce this outside such a platform, start the container
yourself with the model mounted read-only and 8 GPUs visible, then exec in.

Everything is driven by environment variables — you should not need to edit any script.

## The platform

One pod per node, with all 8 GPUs. Only `eth0` is visible inside; the host's RDMA ports
show up in sysfs but this single-node request did **not** come with `/dev/infiniband`, so
userspace verbs enumerate nothing (`ibv_devinfo` reports "No IB devices found").

| | |
|---|---|
| GPU | 8x AMD Instinct MI325X (gfx942 / CDNA3), 304 CU, 256 GiB HBM3E each, SKU M3250101 |
| CPU / RAM | 2x AMD EPYC 9575F (64C each, 128 threads), ~3 TiB |
| ROCm | 7.2.0 |
| Fabric | 8x Broadcom RoCEv2 on the host (`bnxt_en` / `bnxt_re`, fw 231.2.63.0, `rdma0`–`rdma7`) |
| Storage | checkpoints on a shared WekaFS mount |

This recipe never leaves the node, so none of that fabric is used. It is recorded because
the PD-disaggregated recipes that will land next to this one move the KV cache over it,
and two things are already known: the platform injects `NCCL_IB_GID_INDEX=3` into the pod,
so Mooncake will want `MC_GID_INDEX=3` rather than the `1` the ionic-based MI355X clusters
use — and a PD-shaped resource request has to actually carry `/dev/infiniband`, or RDMA
silently degrades to TCP.

One thing not to be confused by: the image entrypoint `infera_inject_host_ionic.sh` is
inert here. It only does anything when `/host-libionic/libionic.so` is bind-mounted, which
is a fix for Pensando ionic ABI drift and irrelevant to a Broadcom fabric; the MI30x base
already ships its own `libbnxt_re` provider with the distro's inbox one disabled.

## Scripts

| Script | Purpose |
|---|---|
| `run_sglang.sh` | Launch the server. `MTP` and `DP_ATTENTION` select the configuration. |
| `verify_correctness.py` | Two checks: needle-in-a-haystack retrieval, and the first request after an idle period. |
| `bench.sh` | Concurrency sweep via `sglang.bench_serving`. |
| `stop.sh` | Stop the server and wait for the VRAM to come back. |

## The three configurations

`MTP` and `DP_ATTENTION` are independent switches:

| | `MTP` | `DP_ATTENTION` | What it is for |
|---|:---:|:---:|---|
| default | 1 | 1 | Highest aggregate throughput and KV capacity — each rank owns only its own requests' KV, so capacity scales with `DP_SIZE`. **Needs the sglang fix in §1.2.** |
| latency | 1 | 0 | Lowest single-user latency: plain TP shards the attention weights instead of replicating them per rank. |
| baseline | 0 | 0 | No speculation, about half the startup time, and what to fall back to when isolating a problem. |

Shared by all three: `--tp-size 8`, `--kv-cache-dtype fp8_e4m3`, tilelang for both DSA
backends, `--chunked-prefill-size 131072`, `--mem-fraction-static 0.85`, and env
`SGLANG_DSA_TRITON_PREFILL=1 HSA_NO_SCRATCH_RECLAIM=1 SGLANG_USE_AITER=1`. SGLang adjusts
some of these itself and says so in the log: DP-attention divides `chunked-prefill-size`
by `DP_SIZE`, and speculative decoding caps `max-running-requests`.

## 1. Prerequisites

### 1.1 Hardware / model / image

```text
Hardware: 1 node, 8x MI325X (gfx942), ROCm 7.2.0 — see "The platform" above
Model:    GLM-5.2-FP8 (~704 GiB, 78 layers + 1 nextn/MTP layer), on a local mount
Image:    built from deploy/docker/Dockerfile.sglang.gfx942, or the base it uses,
          lmsysorg/sglang:v0.5.16-rocm720-mi30x
```

`verify_correctness.py` only needs `requests`, which the image already has.

### 1.2 MTP with DP-attention needs an sglang fix that is not in this repo yet

With both switches on, the HIP DSA indexer dies as soon as one batch is spread unevenly
across the DP ranks:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

The aiter/HIP paged-MQA path sizes its `logits` output from the DP-padded row count while
`lengths` carries the real one; the CUDA path slices to the real count.

The timing is what misleads here, so it is worth being precise: this is not a startup
failure, and it does not need heavy load either. DP-attention pads every rank up to the group
maximum only when `sum(tokens) * 2 >= max(tokens) * DP_SIZE` (`dp_attention.py:99-106`);
below that each rank keeps its own length and no padded rows exist to mismatch. MTP
contributes 4 rows per request, so with `k` of the 8 ranks holding one request each the test
is `8k >= 32`, met at **k ≥ 4** — and the assert then fires as soon as any one rank is empty.
The window is 4 to 7 busy ranks.

That makes a serial client safe and almost anything else fatal. Aggregated warmup issues a
single `/generate`, so `global_num_tokens` is `[4,0,...,0]`, the threshold is not met, and the
server comes up clean; a strictly serial client keeps it that way, which is why an earlier
MTP + DP8 run here served 197 sequential requests green on an unpatched tree, and why
`verify_correctness.py` cannot hit this at all. `bash bench.sh` (§4) is the reproducer, and it
does not take much: `ISL=65536 OSL=512 CONC=8` died here in 27 seconds of serving, one
request per rank, the moment the first of eight differently-sized requests finished and left
its rank idle.

**This was found and fixed by Yinxing Hou (`yihou@amd.com`) on MI355X / gfx950**; the fix
lives on branch `worktree-dsa-hip-dp-rows-fix` (`deploy/docker/patches/sglang_dsa/`) and
has not merged to `main` yet. It is deliberately not duplicated here. The defect is in the
shared aiter/HIP path, so gfx942 needs no different logic — only note that his diff is cut
against sglang 0.5.15.post1 while this image is v0.5.16, so it lands with one line of
fuzz (use `patch -p1`, not `--fuzz=0` and not `git apply`). Until it merges, either:

```bash
# apply his diff into the running container
cd /sgl-workspace/sglang
patch -p1 < .../sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff
```

or run with `DP_ATTENTION=0` (or `MTP=0`), where the fix is inert and not needed.

## 2. Run

```bash
export MODEL=/path/to/GLM-5.2-FP8

bash run_sglang.sh                      # MTP + DP8   (default)
DP_ATTENTION=0 bash run_sglang.sh       # MTP, plain TP8
MTP=0 DP_ATTENTION=0 bash run_sglang.sh # no speculation

tail -f run_sglang.log
curl -s 127.0.0.1:30000/health
```

Cold start is **~20 minutes** with MTP and about half that without: 704 GiB of weights,
read a second time because MTP extracts the nextn layer as the EAGLE draft model. Don't
kill a slow launch — the watchdog timeout is already well above the default.

## 3. Verify

```bash
curl -s 127.0.0.1:30000/v1/models                     # is it up
./verify_correctness.py                               # both checks, 12 cases
./verify_correctness.py --checks idle                 # 3 conditions, ~1 min of sleeping
./verify_correctness.py --checks needle --lengths 65536 --depths 50
./verify_correctness.py --checks needle --flush-cache --repeat 12
```

Expect 12/12: 3 idle conditions and 9 needle cases, every needle case reported as `sparse`
rather than `dense-eq`. A `dense-eq` line means the prompt stayed under `index_topk` (2048)
and proves nothing about the DSA path.

`needle` in its plain form is enough for a wrong top-k, which is deterministic. It is not
enough for the corruption in note 1, which only appeared on *cold* prefills and only on ~8%
of them: these prompts are seeded, so every rerun is byte-identical and served warm from the
prefix cache. `--flush-cache` flushes before each request and `--repeat` supplies the sample
size — a hundred-odd clean cold requests is what it took to call that one fixed, hence the
last line above, which also warns if any request still came back with cached prompt tokens.

`idle` covers the first request after the run queue drains, an axis the needle sweep does not
reach. The failure it guards against was measured on vLLM with this model — garbled output on
one request with nothing in any log, the identical request right after it fine — so a
back-to-back benchmark passes it cleanly while a code-agent workload, whose requests are
separated by thinking time, sits on it. It is a regression guard here, not a known defect.
Its three conditions separate "idleness triggers it" from "it happens at random": only the
middle one failing means idleness, all three failing sporadically means something else.

Both checks send one blocking request at a time, so none of this says anything about
behaviour with several requests in flight — see §1.2 above for a defect only a concurrent
client can reach.

Needle measured 9/9 on both the baseline and the MTP + DP8 configurations, and accept length
~3.8/4 on real prompts. Two caveats: the MTP + DP8 numbers were taken with the **pre-7-29
revision** of the sglang fix in §1.2, whose idle-rank slice was later corrected, so that
configuration is worth re-confirming once the fix merges; and accept length came out
~1.9–2.4 on `--dataset-name random` sweeps versus ~3.8 on real prompts, so record the
workload next to any speculative-decoding number you quote.

## 4. Benchmark

```bash
MODEL=/path/to/GLM-5.2-FP8 bash bench.sh
MODEL=... ISL=65536 OSL=512 CONC="8 16 32" TAG=longctx bash bench.sh
```

Results land in `bench_<TAG>/` next to the script.

Any concurrency at or above 4 spreads a decode batch unevenly over 8 DP ranks, so on an
unpatched tree the default configuration dies here — not at startup — with the §1.2 assert.
This sweep is the only thing in this directory that exercises that shape; `CONC=8` at
`ISL=65536` is enough. Note that `RANGE` below 1.0 widens the window, since requests of
differing length drop out of the batch one at a time rather than together.

## 5. Stop

```bash
bash stop.sh
```

It waits for `rocm-smi` to report the VRAM back. Relaunching before that is how you get an
OOM that reads like a bad `--mem-fraction-static`.

## Notes & gotchas

1. **Do not pass `--enable-aiter-allreduce-fusion`.** On this model it produced all-NaN
   `next_token_logits` on ~8% of long *cold* prefills (3/36 with the flag, 120/120 clean
   without), which sampling turns into token id 0 — decoded as `!`, with nothing in any
   log. Upstream has commented out its own auto-enable for this model family, so the
   `Enable Aiter AllReduce Fusion...` line in the log does **not** mean it is on; read
   `enable_aiter_allreduce_fusion=` in `server_args` instead. The MI355X cookbook passes
   it; don't copy that here.
2. **Short prompts prove nothing.** `index_topk` is 2048, so anything below that never
   leaves the dense-equivalent regime.
3. **MTP stops at 3 steps.** `--speculative-num-steps > 3` fails to build the draft
   kernels on gfx942; `run_sglang.sh` rejects it up front.
4. **`ROCM_QUICK_REDUCE_QUANTIZATION` does nothing here.** Inherited from the MI355X
   cookbook, inert under `AiterCustomAllreduce`, deliberately not set.
5. **`--enable-dp-lm-head` is an untried knob.** It makes the vocab projection parallel
   across the attention TP group so the hidden states need no all-gather across DP groups,
   and GLM-5.2 wires it into both the LM head and the nextn/MTP layer — so it applies
   exactly to the default `MTP=1 DP_ATTENTION=1` configuration and may be worth
   throughput. It requires `--enable-dp-attention`. Every run behind the numbers above had
   it off, so `run_sglang.sh` deliberately does not expose it: measure it before adopting
   it, and add it via `EXTRA_ARGS` to try.
6. **GLM-5.2 is a DSA model but not DeepSeek-V4.** Both carry `index_topk`. The log should
   read `Use dsa attention backend for DeepSeek with DSA` and say nothing about dsv4; if it
   mentions dsv4, the engine has misidentified the checkpoint and is applying the wrong
   gfx942 policy.
7. **gfx942 is not a variation on gfx950.** The DSA path is tilelang's non-gfx950 (304 CU)
   branch, fp8 KV is `e4m3fnuz` rather than `e4m3fn`, and the SGLang base image is
   MI30x-specific — `Dockerfile.sglang.gfx942` and `Dockerfile.sglang` cannot be merged.
