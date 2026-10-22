# Reproduce — one-leg synthetic expert reduction (x8p / x8d2 / x8pc / x128p), AgentX C48 warm-stack protocol

Paths used below (all on the spur cluster, login host `crsuse2-slog-006`, reached as `ssh spur006`):

```bash
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010          # harness workspace (writable from oai compute nodes)
H=$WS/harness  R=$WS/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013          # session dir: mounts/, shimbin/, dryruns, metrics TSVs
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012          # image tar, metrics sampler, image source tree
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4  # config.json sha256 8b46225f7afd...a5c726
PK=<this packup>
```

`scripts/harness/` and `patches/` here are verbatim copies of `$H/...` and `$N/mounts/...` (2026-10-14). The scripts
have absolute paths baked in (`$H`, `$N`, `$CAP`); run them from the original locations, or copy the packup files back to
those paths if they are gone (the mounts TSVs name `$N/mounts/expert8/tree/...` and `$N/mounts/expertn/...`).

## 0. Hold a node (amd-oai, 8x MI355X)

```bash
ssh spur006 'sbatch -A amd-oai -p amd-spur -q amd-oai-qos -N 1 --gres=gpu:8 -c 160 -t 24:00:00 -J yihou-oai-x \
    <PK>/scripts/alloc/hold_inf.yihou.sh'           # may pend QOSGrpNodeLimit (QOS cap 8 nodes)
ssh spur006 'squeue -u yihou -o "%i %T %N"'        # -> JOB, NODE ; IP: getent hosts <node>
```
Do not use `amd-burst-qos` (PreemptMode cancel, jobs died at 20 min). Plain-ssh nodes (135-139) need user approval and use
`chain_cap2l.yihou.sh` from the login host (NFS read-only there, `NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg`).

## 1. Verify the node (GPUs, ECC, foreign containers)

```bash
ssh spur006 "spur exec $JOB bash -c 'rocm-smi --showuse --showmemuse; amd-smi metric --ecc | grep -iE \"GPU|correctable|deferred\"; docker ps -a'"
```
Require 8 idle GPUs, **0 correctable/deferred ECC on every GPU** (026 had 13.2 M on GPU3 and its run crashed — x8d),
and no foreign GPU containers (085/274/206 carried guanchen's `crusoe-sgl-q36`; it was `docker stop`ped — not removed —
only after the user approved).

## 2. Image + JIT cache

```bash
ssh spur006 "spur exec $JOB bash -c 'zstd -dc $CAP/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst | docker load;
  docker image inspect --format {{.Id}} infera-sglang:v0521-yihou-1010-mc4340'"
```
Expected ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2` (kept on 026/085/274/206/089). If
the node re-IDs it (137 → e8e2838f…), add `<node>) EXPECTED_IMAGE_ID=<id>` in a NEW copy of `config.yihou.mc4340n3.sh`
(`mc4340n3` knows 026/085/274/206; never edit a config a running chain re-sources). Seed the JIT caches (saves ~30 min):
```bash
ssh spur006 "spur exec $JOB bash -c 'mkdir -p $WS/cache/<key> && cp -a $WS/cache/102/aiter $WS/cache/102/triton $WS/cache/<key>/'"
```
(`<key>` = last 3 digits of the node; logs of the 2026-10-13 seeds: `logs/seed.*.yihou.log`, `logs/prep.206.yihou.log`.)

## 3. Env file for your node

Copy the round env from `scripts/harness/env.<round>.yihou.sh` to a new file and set `YIHOU_NODE / YIHOU_IP / YIHOU_JOB /
YIHOU_ROUND`; add a `topology.yihou.<key>.tsv` (copy of e.g. `topology.yihou.274.tsv` with the node name/IP). Rounds:

| round | CONFIG | key overrides (all also: `CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95`) |
|---|---|---|
| x8p | `config.yihou.expert8leg.sh` | `E8_LEG=prefill` (KV cap default 3670016) |
| x8pc | `config.yihou.expert8leg.sh` | `E8_LEG=prefill E8_MAX_TOTAL_TOKENS=2433088` |
| x8d2 (x8d) | `config.yihou.expert8leg.sh` | `E8_LEG=decode` |
| x128p | `config.yihou.expertnleg.sh` | `EN_LEG=prefill` (no KV cap) |
| q1 (reference) | `config.yihou.mc4340n.sh` | — (`env.q1-137.yihou.sh` C44 + `env.q1c48-137.yihou.sh` C48 as A3) |

Chain of configs: overlay → `E8_BASE_CONFIG`/`EN_BASE_CONFIG` (`config.yihou.mc4340n2.sh` or `n3`: image + regbase mounts)
→ `MC4340_BASE_CONFIG=config.yihou.E2-target.sh` → `config.yihou.v0521-base.sh` → `bench-harness/config.sh`.

## 4. Dryrun + token diff against the reference (must show only the intended deltas)

```bash
ssh spur006 "cd $H && source <new env> && bash run.v3.yihou.sh dryrun > /tmp/yihou/dry.txt && \
  bash tokdiff.yihou.sh $H/dryrun.q1-137.yihou.txt /tmp/yihou/dry.txt"
```
Allowed differences (see `results/tokdiff/*.txt` for the 2026-10-13 output): node IP/paths/cache dirs, the 137-only
`/mnt/m2m_nobackup` mount and `:ro` NFS; on the reduced leg only: the expert8 (5) / expertn (3) `-v` mounts,
`--ep-size 4` (expert8 only), `--disable-shared-experts-fusion`, `--max-total-tokens <cap>` (expert8 only).

## 5. Launch (driver runs ON the compute node, detached)

```bash
ssh spur006 "spur exec $JOB bash -c 'cd /; setsid nohup bash $H/chain_cap2.yihou.sh <new env> 48 48 > $R/<round>.chain.log 2>&1 < /dev/null &'"
```
`chain_cap2` = `pipeline.v3` (stop-check → ports → gpuidle → etcd → prefill up → decode up → router → smoke → audit v3)
→ A1 C48 (fresh, unscored) → `POST :29357/flush_cache` (decode) then `:29101/flush_cache` (prefill) → A2 C48 (scored)
[→ flush → A3 at the 3rd CONC arg]. It puts `$N/shimbin` first on PATH (the harness' `ssh <node> cmd` runs locally) and
starts a 5 s /metrics sampler per bench into `$N/runs/<round>/metrics.<tag>.tsv`. Bring-up 25-35 min, each bench ~85 min.
To queue a round behind another on the same node: `wait_then.yihou.sh <old-round> <old-env> <new-env> 48 48`
(→ `stop_then_chain.yihou.sh`: stop old stack, wait for 0 KFD processes, start chain_cap2).

Live checks: `status.tsv` lines up to `smoke ok`; `audit.txt` all PASS (v3 checks per role `EXPERT8` = fusion off and
`MAX_TOTAL_TOKENS`, and that the mounts are on the reduced leg only); prefill/decode log lines
`YIHOU_EXPERT8 role=target|draft layer=… routed=8 topk=8 ep=4 local=2` (304 lines prefill; 312 incl. 8 `role=draft` on
decode) or `YIHOU_EXPERTN keep=128 …`; /metrics `max_total_num_tokens` = 3,670,016 / 3,318,400 / 2,433,088 as in the README.

## 6. Read the score

```bash
ssh spur006 "python3 $N/tools_cmp.yihou.py '<round> A2 C48=<round>/agentx-c48-A2/agentx_conc48.json'"
# args are label=path-relative-to-$R; it always also prints its built-in rows (q1 A1..A4, x8d/x8p A1, tb56 A1)
ssh spur006 "python3 -B $N/tools_summ.yihou.py"     # rewrites the session table ($N/summary.v0521cap2.yihou.md)
```
Score = `request_metrics.throughput.per_gpu.total_tput_tps`. Reference values: `results/results.csv`.
Expected A2 (single runs, ± node effects unknown): x8p 25,905 · x128p 25,297 · x8d2 23,591 · x8pc 22,489 · q1 A3 22,306.

## 7. Rebuild the patched sources (only if `$N/mounts` is gone)

- expert8: `cd <copy of the image sglang tree root (python/sglang/...)> && patch -p1 < patches/expert8/yihou-expert8-nextn.patch`,
  or use `patches/expert8/tree/` directly; check `sha256sum` of the `.orig` files against `patches/sha256.image-vs-mounts.txt`
  (image tree: `$CAP/src/image-9d755dd6/sglang/srt`). Overlay config: `patches/expert8/config.expert8.json`.
- expertn: `python3 scripts/session/build_expertn.py $N $CAP/src/image-9d755dd6/sglang/srt` (reads `$N/yihou_expertn.py`
  = `patches/expertn/yihou_expertn.build-src.py` and `$N/loader_hunk.py`, writes `$N/mounts/expertn/`). Overlay config:
  `patches/expertn/config.expert128.json`.
