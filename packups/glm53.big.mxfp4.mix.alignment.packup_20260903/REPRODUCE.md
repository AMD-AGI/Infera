# Reproduce — GLM-5.3-MXFP4 MIX alignment

Cold-start reproduction on one 8×MI355X node. ~15 min bring-up + ~10 min per
sweep arm. Everything below is copy-pasteable; substitute only `MY_IP`.

## 0. Prerequisites

- One 8×MI355X (gfx950) node, all 8 GPUs free.
- Image `infera/engine-sglang:v0518-glm53` present (see `environment.md` for the
  digest and how it was built).
- Weights, **absolute path, resolved**: `/perf_apps/data/models/GLM-5.3-MXFP4`
  (408 GB, 282 shards). On these hosts `/apps/data/models` is a **symlink onto a
  different NFS mount**; bind the realpath, not the symlink parent — see
  `notes.md` §5.
- Scripts from `scripts/` staged somewhere on the node.

**Secrets needed** (values not in this packup): SSH access to the node; a docker
login for the registry only if the image must be pulled rather than built.

## 1. Check the node is actually free

```bash
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used Memory'   # want ~0.3 GB each
docker ps                                                          # want empty
ss -lnt | grep -E ':(22379|30010|8110|5567|8811) '                 # want empty
```

**Do not use `rocm-smi --showmemuse`'s `VRAM%`** — it does not fall when memory
is released and read 76 % on empty cards. `--showmeminfo vram` is the check.

## 2. Bring up the arm

Three arms were run. Pick one.

```bash
cd <scripts-dir>
MY_IP=<node-ip>                      # the fenic/management interface, e.g. 10.235.192.136
IMG=infera/engine-sglang:v0518-glm53
MODEL=/perf_apps/data/models/GLM-5.3-MXFP4

# (a) TP4 control — DPA off, MTP off, kvd off  [4 GPUs]
MY_IP=$MY_IP IMAGE=$IMG VARIANT=mxfp4 MODEL=$MODEL \
  MODEL_MOUNT=/perf_apps/data/models bash big_mix_up.sh

# (b) MIX TP8 features-off isolator            [8 GPUs]
MY_IP=$MY_IP IMAGE=$IMG VARIANT=mxfp4 MODEL=$MODEL \
  MODEL_MOUNT=/perf_apps/data/models \
  TP=8 GPUS=0,1,2,3,4,5,6,7 DPA=0 MTP=0 CTR=glm53_big_mix8 bash big_mix_up.sh

# (c) matched TP8 — DPA dp8 + EAGLE MTP 3/1/4  [8 GPUs]
#     add: TP=8 GPUS=0,1,2,3,4,5,6,7 DPA=1 MTP=1 MAX_RUNNING=256 CUDA_GRAPH_BS=256
```

Cold start is ~9 min for mxfp4 (282 shards) and ~15 min for fp8 (141 shards,
704 GB). **Silence is not a hang.**

## 3. Verify — and do NOT accept a 200 as evidence

```bash
MY_IP=$MY_IP VARIANT=mxfp4 MODEL=$MODEL CTR=<ctr> ROUTER_PORT=<port> bash big_smoke.sh
```

What must be true, block by block:

- **block 3** — a coherent completion with `reasoning_content` separated.
  Garbage here is **not** a sampling problem; it means the DSA-on-ROCm env block
  did not take effect. `VERDICT: PASS` required.
- **block 5** — resolved args must show `dsa_prefill_backend='tilelang'`,
  `dsa_decode_backend='tilelang'`, `kv_cache_dtype='fp8_e4m3'`,
  `moe_runner_backend='aiter'`, `quantization='quark'`, and the DPA/MTP settings
  you intended. **Read them off the engine, never off the wrapper.**
- **block 6b** — the mxfp4 silent-dequant guard. Must print
  `VERDICT: PASS -- AITER native FP4 MoE, not dequantised to BF16`, backed by
  `float4_e2m1fn_x2` and `QuantType.per_1x32` lines. Without it the server still
  starts, still answers correctly, and is several times slower with nothing said.
- **block 7** — fault scan EMPTY.

**Per-rank reading trap:** at dp8 the startup line prints
`max_running_requests=32` and `chunked_prefill_size=8192` while the globals are
256 and 65536. That is a **division by `dp_size`, not a clamp.** The GLM-5.2
baseline's own log shows both values too.

## 4. Run the sweep

```bash
CTR=<ctr> HOST=$MY_IP PORT=<router-port> SERVED=glm-5.3-mxfp4 MODEL=$MODEL \
  ARMS="p50 p90" CONCS="1 8 16 24" \
  OUTDIR=<out> CSV=<out>/fixlen.csv bash big_fixlen.sh
```

Three flags in `big_fixlen.sh` are load-bearing and are **not** defaults:
`--random-range-ratio 1.0` (pins every prompt to exactly ISL);
`--temperature 1.0 --top-p 0.95` (the checkpoint's own non-greedy
`generation_config`); `--num-prompts = 10 × conc` (the InferenceX convention).
It benches the **router**, not the engine port, to match the baseline harness.

## 5. Archive before teardown

```bash
docker cp <ctr>:/tmp/fixlen/. <dest>/jsonl/      # per-request ttfts/cached_tokens/generated_texts
docker cp <ctr>:/tmp/glm53_big_mix.log <dest>/
```

Do this **first**. Engine logs and per-request JSONL have been lost twice on this
project by tearing down before copying.

## 6. Tear down — your own containers, by exact name

```bash
docker rm -f <ctr> <ctr>_etcd
rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-7]\].*Used Memory'   # confirm drained
```

Never a pattern, never a variable. This is a shared node.

## Comparing against the GLM-5.2 baseline

Baseline packup:
`~/dev/git.16-19/infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/`

Read its **"what ISL means here"** section first: ISL there is the *fresh
remainder* of a prefix-cached prompt, not the full prompt, and that substitution
was a deliberate decision — do not "correct" it.

To re-derive the acceptance and cached-fraction comparisons in `notes.md`:

```bash
# acceptance, scoped to the p50+p90 window (NOT the whole log — see notes.md §2)
zcat logs/glm52_mix_base.log.gz | grep -a "accept len" \
  | awk '$2 >= "07:13:48]" && $2 <= "07:49:00]"' \
  | grep -aoE "accept len: [0-9.]+" | awk '{print $3}' | sort -n \
  | awk '{a[NR]=$1; if($1>=4) f++} END {printf "n=%d median=%s at4.00=%.1f%%\n", NR, a[int(NR*0.5)+1], 100*f/NR}'

# engine-side cached fraction over a window
zcat logs/glm52_mix_base.log.gz | grep -a "Prefill batch" \
  | awk '$2>="07:13:48]" && $2<="07:19:30]"' \
  | grep -oE "#new-token: [0-9]+, #cached-token: [0-9]+" \
  | awk -F'[ ,]' '{n+=$2; c+=$5} END {printf "cached_frac=%.4f\n", c/(n+c)}'
```

**The engine log has no arm delimiters**, so any window-scoped statistic needs an
external boundary (run durations, or the sweep's own output-file mtimes). Getting
this wrong once produced a cached fraction of 0.80 that had swallowed an
unrelated workload's traffic.
