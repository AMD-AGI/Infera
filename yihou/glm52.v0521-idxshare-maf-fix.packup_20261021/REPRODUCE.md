# Reproduce — index_share decode MAF, its diagnosis, and the fix

All paths are on the spur cluster (`ssh spur006`, fallback spur007/spur005). Login hosts have 1 CPU and evict a user above ~4 GB:
run drivers ON the node via `spur exec <jobid> bash -c 'cd /; ...'` (runs as root at `/`). Never `docker rm`; stop with `docker stop`.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness       # copies: scripts/harness/, scripts/mounts-src/
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
W=/shared_nfs/yihou_glm52/g4tune_20261019                 # env.g4n*, launch_after_prep, stop_round (copies: scripts/session/)
D=/shared_nfs/yihou_glm52/idxshare_dbg_20261020           # env.dbge*/idxf*, dryruns, guards, debug-agent excerpts
```

## 0. Prerequisites

- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`; image `infera-sglang:v0521-yihou-1010-mc4340`
  (ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`, checked by `launch_after_prep` at every launch, `results/launch/`).
- A held 8-GPU MI355X node (OAI: `-A amd-oai -q amd-oai-qos -p amd-spur`, via the machine-pool manager), prepared with
  `spur exec $J bash -c "cd /; setsid nohup bash $N/prep_node.yihou.sh <key> seed > /dev/null 2>&1 &"` (image load + JIT cache seed;
  `launch_after_prep` waits for it and checks img / .so counts / locks / KFD).
- The node must be in the node table of the config chain (`config.yihou.mc4340n16.sh` 250, `n17` +097, `n18` +268 +054, with
  `topology.yihou.<key>.tsv`). For a new node add it the same way.
- Arm a stop-watcher so nothing outlives the job's 24 h MaxWall:
  `spur exec $J bash -c "cd /; setsid nohup bash $D/stop_before_maxwall.yihou.sh <key> <HH:MM UTC> >> $D/stop_before_maxwall.<key>.yihou.log 2>&1 &"`.

## 1. Reproduce the fault (no fix)

Config = the user's 10-20 rule: `config.yihou.g4new.sh` (PR #42553 GTT HiCache 260 GB/rank, `INDEX_SHARE_FOR_MTP=true`, decode custom AR on)
→ `config.yihou.xferfix2.sh` → `config.yihou.offload.sh` (G4K1 step 1) → `config.yihou.tokw2.sh` → `config.yihou.mc4340n1{6,7,8}.sh`
→ `config.yihou.E2-target.sh` → `config.yihou.v0521-base.sh`. Env template: `scripts/session/env.g4n1-097.yihou.sh`.

```bash
cp $W/env.g4n1-097.yihou.sh $D/env.<round>-<key>.yihou.sh   # edit YIHOU_NODE / YIHOU_IP / YIHOU_JOB / YIHOU_ROUND (+ TOKW2_BASE_CONFIG node table)
( cd $H; source $D/env.<round>-<key>.yihou.sh; bash $H/run.v3.yihou.sh dryrun > $D/dryrun.<round>-<key>.yihou.txt )
bash $H/tokdiff.yihou.sh $W/dryrun.g4n1-097.yihou.txt $D/dryrun.<round>-<key>.yihou.txt    # must show only node/IP/job/round
spur exec $J bash -c "cd /; setsid nohup bash $W/launch_after_prep.yihou.sh $D/env.<round>-<key>.yihou.sh <round> <key> 56 56 > /dev/null 2>&1 &"
grep -c "Memory access fault" $R/<round>/server-logs/decode-0.a1.log   # 8 idx-true no-fix rounds faulted 23-100 min into load; rounds stopped earlier (g4n1b, dbge2*, dbge3*) had not faulted yet
```

Control: `G4NEW_IDXSHARE=false` in `YIHOU_OVERRIDES` (= g4n3, `scripts/session/env.g4n3-097.yihou.sh`) ran A1+A2 clean twice.
Stop a round: `spur exec $J bash -c "cd /; bash $W/stop_round.yihou.sh <key> <round>"` (docker stop only).

## 2. Apply the fix (validation rounds idxf1a/b/c)

`CONFIG=$H/config.yihou.idlefix.sh` with `IDLEFIX_BASE_CONFIG=$H/config.yihou.g4new.sh` prepended to `YIHOU_OVERRIDES`. The overlay swaps
`mounts.yihou.v2-xferfix2-gtt.tsv` → `mounts.yihou.v2-xferfix2-gtt-idlefix.tsv`, which adds exactly one line:

```
$H/mounts-v0521/src/eager_runner.yihou-idlefix.py	/sgl-workspace/sglang/python/sglang/srt/model_executor/runner/eager_runner.py
```

Env: `scripts/session/env.idxf1a-097.yihou.sh` (097), `env.idxf1b-268.yihou.sh`, `env.idxf1c-054.yihou.sh`. tokdiff vs g4n1 must show only that
mount (+ node). Launch as in §1. Pass = A1+A2 complete, `grep -c "Memory access fault"` = 0 in `server-logs/decode-0.a1.log`.

Without the harness: apply `patches/0001-idlefix-vs-image-9f5d8712.diff` to `/sgl-workspace/sglang` inside the image (or
`patches/0002-...` to upstream main); it is pure Python, no rebuild.

## 3. Diagnosis toolchain (what found the bug)

All of it is decode-only and is selected by overlays on top of g4new:

| tool | how | output |
|---|---|---|
| ROCm debug agent (lean) | `config.yihou.dbgE1.sh` adds decode env `HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 HSA_ENABLE_DEBUG=1 ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals` (`DECODE_DEBUG_AGENT=1`, default) | at the fault the agent prints every stopped wave (`wave_N: pc=... <kernel name> (stopped, reason: MEMORY_VIOLATION)`, SGPR/VGPR dump, disassembly) into the decode log; excerpts `results/debug_agent/` |
| seed guard | `config.yihou.dbgE1.sh`, mounts `*-seedguard.tsv` (eagle_worker_v2 / eagle_disaggregation copies + counters), `YIHOU_DSA_SEED_GUARD=off\|count\|mask` | `YIHOU-SEEDGUARD` lines: bad_entries, pd_seedless_batches, draft_eager, draft_iters (`results/seedguard/`) |
| indexer top-k guard v1-v5 | `config.yihou.dbgE2.sh`, `YIHOU_TOPK_GUARD=count\|mask`, `YIHOU_TOPKGUARD_VER=''\|2\|3\|4\|4sk\|5sk2\|5sk3\|5sk3fix` selects `mounts.yihou.v2-xferfix2-gtt-seedguard-topkguard<VER>.tsv` | `YIHOU-TOPKGUARD` counters (blind spots: `notes.md`) |
| sgl_kernel call-site guard (SKGUARD v1-v3) | mounted `top_k.yihou-skguard{,2,3}.py` over `/opt/venv/lib/python3.10/site-packages/sgl_kernel/top_k.py` (VER `4sk`, `5sk2`, `5sk3*`), `YIHOU_SKGUARD=off\|count\|mask` | v2+: eager calls checked synchronously before launch, `YIHOU-SKGUARD-EAGER BAD` with site, shapes, Python stack; v3 adds the ForwardBatch + attention-metadata dump and `mask` clamps lengths into the row (`results/bad_dumps/`) |
| CAP_SYSLOG dmesg | `spur exec` root has no CAP_SYSLOG; a container with `--cap-add SYSLOG` can read the kernel log. The exact command used on 10-20 was not saved; equivalent: `docker run --rm --cap-add SYSLOG --network none infera-sglang:v0521-yihou-1010-mc4340 dmesg -T \| grep -iE 'amdgpu.*(fault\|UTCL2\|PERMISSION)'` (reconstructed, not run for this kit) | `results/maf/dmesg.from_prep_notes.txt` |
| disassembly | `results/debug_agent/coop.co.yihou` (code object extracted from the image's sgl_kernel `common_ops`) and `coop.dis.yihou.txt`; the extraction commands were not recorded | pc +700 = `s_waitcnt` after `global_load_dwordx4` in the round-0 histogram row scan; SGPR map in `notes.md` |

Unit tests for SKGUARD (run inside the image on one GPU): `scripts/diagnosis/test_skguard.yihou.py`, `test_skguard2.yihou.py`
(eager log, CUDA-graph replay counting 5/5, mask clamp).

Env of every diagnosis round: `scripts/session/env.dbge*.yihou.sh`; dryruns `results/dryruns/`.

## 4. Read-out

```bash
cd $R && python3 -B $N/tools_hits.yihou.py "idxf1a-A2=idxf1a/agentx-c56-A2/agentx_conc56.json"
grep -c "Memory access fault" $R/<round>/server-logs/decode-0.a1.log
grep "SKGUARD-EAGER BAD" $R/<round>/server-logs/decode-0.a1.log | cut -c1-600
grep "YIHOU-SEEDGUARD" $R/<round>/server-logs/decode-0.a1.log | tail -4
```
