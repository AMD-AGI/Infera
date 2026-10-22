# Reproduce — OffloaderV2 on the prefill leg (S0 = G8K1, sweep G4K1 / G3K1 / G2K1)

Wall time per node: image load + JIT seed ~5 min. Then the correctness round: bring-up 12-35 min (prefill `scheduler_e2e` 275-2064 s observed) plus tests 3-9 min, then stop.
The perf round follows: bring-up again, A1 ~84 min, flush, A2 ~83 min, then optional A3.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # harness (verbatim copy in scripts/harness/)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs         # round outputs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013             # session workspace: shim, offload patch, dryruns, correctness tests, metrics
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012            # image tar, metrics sampler, extracted sglang source
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010             # $WS/cache/<node-key>/{aiter,triton} JIT caches
```

Scripts have these paths baked in. To run from this kit instead, copy `scripts/harness/*` to a new directory and `sed` `H`/`WS`/`N` in the env files and configs.

## 0. Prerequisites

- spur cluster login (`ssh spur007` = `crsuse2-slog-007`), account/QoS `amd-oai` / `amd-oai-qos`.
- **Secrets:** none beyond the cluster login. Model and image tar are on NFS, and no tokens are in any env file.
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (unmodified), served as `glm5.2-mxfp4`.
- Image `infera-sglang:v0521-yihou-1010-mc4340`, tar `$CAP/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst`.
  After `docker load` the ID is `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`; verified on 089/252/266/288.
- OffloaderV2 is **already in the image** (`sglang/srt/utils/offloader.py`). Only the attribute patch is added, as a bind mount.

## 1. Node, image, JIT seed

```bash
ssh spur007 'sbatch --parsable -A amd-oai -p amd-spur -q amd-oai-qos -N1 --gres=gpu:8 -c 160 -t 24:00:00 \
   -J yihou-oai-hold --export=ALL ~/spur-hold/hold_node.sh'        # with HOLD_SECONDS=86400 in the env, else it sleeps 8 h
J=<jobid>; K=<node short id, e.g. 252>
# checks: 8x MI355X, ECC uncorrectable 0, no foreign containers, 0 KFD processes
spur exec $J bash -c 'cd /; rocm-smi --showproductname | grep -c "Card Series"; amd-smi metric --ecc | grep -E "TOTAL_(UN)?CORRECTABLE_COUNT" | sort | uniq -c; docker ps; rocm-smi --showpids'
# image + JIT seed (copy of the 102 caches); log $N/logs/prep.$K.yihou.log must end with rc=0 and SEED_OK
T=$CAP/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst; C=$WS/cache
I=9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2
spur exec $J bash -c "cd /; setsid nohup bash -c \"zstd -dc $T | docker load; echo rc=\\\$?; mkdir -p $C/$K/aiter/prefill $C/$K/aiter/decode && cp -a $C/102/aiter/prefill/$I $C/$K/aiter/prefill/ && cp -a $C/102/aiter/decode/$I $C/$K/aiter/decode/ && cp -a $C/102/triton $C/$K/ && echo SEED_OK\" > $N/logs/prep.$K.yihou.log 2>&1 < /dev/null &"
```

Each new node needs:
- its `<node>) EXPECTED_IMAGE_ID=…` in a config copy. `config.yihou.mc4340n5.sh` = n4 + 252/266/288; the diff is one comment and one `case` alternative.
- a `topology.yihou.<K>.tsv` in the harness. These exist for 089/136/252/266/288.

## 2. Env files (`scripts/harness/env.<round>-<node>.yihou.sh`)

All rounds use `CONFIG=$H/config.yihou.offload.sh` on top of `OFF_BASE_CONFIG=config.yihou.mc4340n{4,5}.sh` → `MC4340_BASE_CONFIG=config.yihou.E2-target.sh`.
The workload is pf90: prefill mem 0.90, decode mem 0.95, decode max_running 96, C48.

| round | env | OFF_G / OFF_K / OFF_STEP | DECODE_SIMULATE_ACC_LEN | purpose |
|---|---|---|---|---|
| ofs0c | `env.ofs0c-089` | 8 / 1 / 1 | empty (sim OFF) | correctness |
| ofs0 | `env.ofs0-089` | 8 / 1 / 1 | 3.61 | perf C48 A1 → A2 |
| ofg4c / ofg4 | `env.ofg4c-252` / `env.ofg4-252` | 4 / 1 / 1 | OFF / 3.61 | correctness → perf C48/C48/C56 |
| ofg3c / ofg3 | `env.ofg3c-266` / `env.ofg3-266` | 3 / 1 / 1 | OFF / 3.61 | same |
| ofg2c / ofg2 | `env.ofg2c-288` / `env.ofg2-288` | 2 / 1 / 1 | OFF / 3.61 | same |
| references | `env.pf90-089`, `env.cx90-089` | — | 3.61 / OFF | no offload |

The `*-136` variants of ofs0/ofs0c were prepared for node 136 (plain ssh, own image build `9872bbda…`). They never ran, because the node was taken by another user.

What `config.yihou.offload.sh` adds, **prefill leg only** (decode unchanged):
- `PREFILL_MOUNTS_ADD=$WS/harness/mounts.yihou.offload-attrs.tsv`. This bind-mounts `$N/offload/offloader.yihou-attrs.py` over `/sgl-workspace/sglang/python/sglang/srt/utils/offloader.py`.
- `--offload-group-size $OFF_G --offload-num-in-group $OFF_K --offload-prefetch-step $OFF_STEP --offload-mode cpu`

## 3. Dryrun + token diff (before every launch)

```bash
cd $H; ( source env.ofg2-288.yihou.sh; bash run.v3.yihou.sh dryrun > $N/dryrun.ofg2-288.yihou.txt 2>&1 )
bash tokdiff.yihou.sh $N/dryrun.ofs0-089.yihou.txt $N/dryrun.ofg2-288.yihou.txt
```

Expected differences (`results/dryruns/tokdiff.*`):
- ofs0 vs pf90 / ofs0c vs cx90: the offload mount and the four offload flags on the prefill leg.
- ofg* vs ofs0: only node paths/IP and `--offload-group-size` 8 → 4/3/2.

## 4. Launch (driver runs ON the node; local ssh shim `$N/shimbin/ssh`)

```bash
# S0 (as run on 089, after tw90 on the same node): scripts/harness/after_tw90_s0.yihou.sh
#   = wait tw90 -> stop -> chain_cx env.ofs0c-089 -> chain_cap2 env.ofs0-089 48 48
# sweep, one node per ratio:
spur exec $J bash -c "cd /; setsid nohup bash $H/ofs_seq.yihou.sh ofg2c ofg2 288 48 48 56 > $R/ofg2.seq.log 2>&1 < /dev/null &"
#   ofs_seq = chain_cx <corr env> (up, census, smoke, audit, $N/correctness_tests.yihou.py, stop) -> wait 0 KFD -> chain_cap2 <perf env> C1 C2 C3
```

## 5. Read-out

```bash
grep -a offloader $R/<round>/server-logs/prefill-0.log | grep -c "DP0 TP0"          # offloaded MoE modules per rank
grep -a "Load weight end" $R/<round>/server-logs/prefill-0.log | head -1             # weights after offload
grep -ao "max_total_num_tokens=[0-9]*" $R/<round>/server-logs/prefill-0.log | head -1
grep -a "Memory pool end" $R/<round>/server-logs/prefill-0.log | head -1             # avail mem
cat $N/runs/<round>c/correctness.log; zcat results/correctness/<round>/results.jsonl.gz
cd $R && python3 -B $N/tools_hits.yihou.py "ofs0 A2=ofs0/agentx-c48-A2/agentx_conc48.json"
python3 -B scripts/tools/results_csv_v15.yihou.py offload
```

Expected per rank: 9 / 19 / 26 / 39 modules for G = 8 / 4 / 3 / 2, and the KV values in the README table.
S0 single replicates: A1 ≈ 22.4k, A2 ≈ 22.8k tok/s/chip.

## 6. H2D microbenchmark (gate for the offload ratio)

`scripts/session/h2d_bench.yihou.py` was run inside the image container on 089 GPUs 0-3, with a pinned 1.21 GB buffer. Output: `results/h2d/h2d_bench.089.json`.
