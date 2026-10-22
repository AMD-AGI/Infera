# Reproduce — decode in-flight KV cap (runshare v1/v2), AgentX C40 warm-stack protocol

Prerequisites (allocation, image `infera-sglang:v0521-yihou-1010-mc4340`, harness location `WS=/shared_nfs/yihou_glm52/v0521dbg_20261010`):
see the baseline kit REPRODUCE §0-2 (`/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/REPRODUCE.md`).
`scripts/harness/` holds the round files; the full harness is in the capacity pack-up `scripts/harness/` (and on NFS).

## 1. Patch (decode leg only, runtime bind-mount)

- v2 (recommended next run): `patches/decode.yihou-runshare2.py` → `/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py`
  (`:ro`), env `YIHOU_DECODE_RUNNING_KV_SHARE=0.5 YIHOU_RUNSHARE_MAX_WAIT_S=5`. Base = image decode.py md5 `bd1e5fe7…`;
  check with `diff <image decode.py> patches/decode.yihou-runshare2.py` == `patches/decode.yihou-runshare2.diff`.
- Harness overlay: `CONFIG=$WS/harness/config.yihou.runshare2.sh` (`RUNSHARE_BASE_CONFIG` → `config.yihou.mc4340.sh` on 102
  or `config.yihou.mc4340n.sh` with your node's image ID). v1: `config.yihou.runshare.sh`.

## 2. Round (node 102 example; retarget YIHOU_NODE/IP/JOB for another node)

```bash
cd $WS/harness && source env.pc40d-102.yihou.sh && bash run.v2.yihou.sh dryrun > /tmp/d.txt
bash tokdiff.yihou.sh dryrun.e2g-102.yihou.txt /tmp/d.txt     # only the decode mount + 2 env vars may differ
```
`chain_pc40d.yihou.sh` = stop previous stack → up → census → smoke → audit → burst smoke → C40 A1 → flush → C40 A2.
Its first lines wait for the pc40c files; for a fresh run delete that wait (write a new file, do not edit in place) or
run the stages by hand:
```bash
bash pipeline.yihou.sh                                    # up .. bench (A1, fresh) -> runs/pc40d/agentx-c40
python3 <kit>/scripts/tools/longprompt_burst.yihou.py http://$YIHOU_IP:28100 64 200000 16   # cap must fire
curl -X POST http://$YIHOU_IP:29357/flush_cache; curl -X POST http://$YIHOU_IP:29101/flush_cache   # retry prefill on 400
BENCH_TAG=A2 bash run.v2.yihou.sh bench                   # scored -> runs/pc40d/agentx-c40-A2/agentx_conc40.json
```
Run drivers detached (`setsid nohup … < /dev/null &`) and **not on a login host that may be lost**.

## 3. Verify

- Cap: `strings runs/<r>/server-logs/decode-0.a1.log | grep -a 'YIHOU-RUNSHARE block' | tail` (every 100th block;
  fields nonrunning/cand/cap/running/xfer/waited, `skip`/`stop`); max of `pre-allocated usage:` must be ≤ 0.5 (+ slack).
- Score: `request_metrics.throughput.per_gpu.total_tput_tps` of A2 vs 22,114 (warm replicates on 102: 21,681-22,114).
- Telemetry: `scripts/tools/metrics_sampler.yihou.py` (5 s) — decode gauges refresh only on decode batches.
