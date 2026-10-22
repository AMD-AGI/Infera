# Reproduce — OffloaderV2 sweep at C48 / C56 / C64

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # verbatim copy in scripts/harness/
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
C=/shared_nfs/yihou_glm52/v0521dbg_20261010/cache            # JIT/client caches, key = last '-' field of the hostname
```

## 0. Prerequisites

- Same as `../glm52.v0521-offloaderv2.packup_20261015/REPRODUCE.md` §0: model, image tar `9d755dd6`, the YIHOU-OFFLOAD-ATTRS patch (copied in `patches/`). Secrets: none beyond the spur login.
- **Node pools:**
  - `amd-spur` oai (252/266/288; released).
  - `amd-spur-v2` burst (new partition, 28 × MI355X nodes `crsuse2-m2m-v3-*`, PreemptMode=CANCEL, burst MaxWall 24 h).
  - Plain-ssh node 136 (its own image build `9872bbda…`, node-local run dir `/mnt/m2m_nobackup/yihou_v0521dbg`).

## 1. Hold + prepare a node

```bash
HOLD_SECONDS=86400 sbatch --parsable --export=ALL -A amd-burst -p amd-spur-v2 -q amd-burst-qos -N1 --gres=gpu:8 -c 160 \
    -t 24:00:00 -J yihou-newpart-burst ~/spur-hold/hold_node.sh
# on the node (spur exec <job>): docker stop any non-system container holding GPUs (user-authorized 10-15), then
spur exec $J bash -c "cd /; setsid nohup bash $N/prep_node.yihou.sh <key> seed > /dev/null 2>&1 &"   # docker load + seed from cache/102
```

**Check before trusting a reused key.** v3-026 maps to key `026`, which an older node already used. Count the files and compare with cache/102:

```bash
ls $C/<key>/aiter/prefill/9d755dd6*/ | grep -c '\.so$'   # must be 16 (decode: 14), same set as cache/102
```

Client dirs (`agentx uv-python client-home client-pycache`) of a reused key may be broken; move them aside so the bench recreates them.

## 2. Config chain and env files (`scripts/harness/`)

`CONFIG=config.yihou.offload.sh` → `OFF_BASE_CONFIG` → for new rounds `config.yihou.tokw2.sh` (`TOKW2_BASE_CONFIG`) → `config.yihou.mc4340n{,5,6}.sh` → `config.yihou.E2-target.sh`.
Workload: pf90 (prefill mem 0.90, HiCache 450 GB write_through, decode mem 0.95, max_running 96, sim 3.61).

| round | env | node | G/K/step | tok/detok | chain |
|---|---|---|---|---|---|
| ofg4b | `env.ofg4b-136` | 136 | 4/1/1 | 1/1 | chain_cap2l 48 48 56 |
| ofg4h / ofg4h2 | `env.ofg4h-136` / `env.ofg4h2-136` | 136 | 4/1/1 | 4/4 | chain_cap2l 56 56 64 |
| ofg3b | `env.ofg3b-v3020` | v3-020 | 3/1/1 | 1/1 | chain_cap2 48 48 56 |
| ofg2h, ofg2h2 (failed), ofg2h3 | `env.ofg2h*-v3026` | v3-026 | 2/1/1 | 1/1, 1/1, 4/4 | chain_cap2 56 56 64; ofg2h3 continued via `cont2_v3026.yihou.sh` |
| pf90b (stopped in bring-up) | `env.pf90b-v3026` | v3-026 | — | 1/1 | — |
| ofg4 / ofg2 (aborted) | `env.ofg4-252` / `env.ofg2-288` | 252 / 288 | 4 / 2 | 1/1 | `ofs_seq.yihou.sh` |

Token diffs are in `results/dryruns/tokdiff.*`. Every pair differs only by node IP/paths, the group size, or the two tok/detok flags.

## 3. Launch

```bash
# burst node (driver ON the node):
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.ofg3b-v3020.yihou.sh 48 48 56 > $R/ofg3b.chain.log 2>&1 &"
# optional warm-up watcher (py-spy dumps if PD warm-up > 300 s):
spur exec $J bash -c "cd /; setsid nohup bash $H/warmup_pyspy.yihou.sh ofg2h3 026 <ISO-start> 300 > $R/ofg2h3.pyspy.log 2>&1 &"
# 136 (login host; VRAM-gated, capped stop): see run136_ofg4h2.yihou.sh / after_ofg4b136l2.yihou.sh
```

## 4. Read-out

```bash
cd $R && python3 -B $N/tools_hits.yihou.py "ofg3b A2=ofg3b/agentx-c48-A2/agentx_conc48.json"
python3 -B scripts/session/results_csv_v15b.yihou.py highconc
grep -a "Start of pd disaggregation warmup\|End of disaggregation warmup" <prefill-0.log>     # normal: 55-85 s
```
