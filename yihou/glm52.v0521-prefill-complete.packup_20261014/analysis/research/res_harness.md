I've read everything the question asks for. I changed nothing on the cluster. Paths are relative to `H=$WS/harness` unless they start with `/`, and line numbers come from `cat -n`.

**Answer up front:**
- **Retarget:** to move a round to a new node, write a new `env.<round>-<node>.yihou.sh` and add the node's image ID to the `case` in `config.yihou.mc4340n.sh`. Chains and the A3/A4 drivers name their env file, log files and container names directly, so each needs a fresh copy.
- **Per-leg settings:** the two legs can already take different mem fraction, max_running, EP, extra args, extra env, extra mounts and HiCache settings.
- **Model path:** the legs cannot take different `--model-path` today. There is one `MODEL`, and two checks plus the router all compare against it (section b).

## (a) Retargeting a round to a new node

**Per-round env file.** Template is `env.q1-137.yihou.sh:4-8`:
```
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=44 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
```
- **Required variables** (`config.yihou.v0521-base.sh:29-33`): `YIHOU_NODE`, `YIHOU_IP`, `YIHOU_JOB` and `YIHOU_ROUND`. `YIHOU_JOB` must be set; an empty value means plain ssh. Use a new `YIHOU_ROUND` for every fresh bring-up. `up` refuses if containers with the prefix already exist (`run.v2.yihou.sh:226-227`).
- **Image ID.** `config.yihou.mc4340n.sh:23-32` sets `IMAGE=infera-sglang:v0521-yihou-1010-mc4340` and picks `EXPECTED_IMAGE_ID` from a case statement:
  - 102, 174 and 029 → `sha256:9d755dd6…`
  - 136 → `sha256:9872bbda…`
  - 137 → `sha256:e8e2838f…`
  - any other node → `exit 2`

  For a new node, make a new copy of that config with a `crsuse2-m2m-NNN) EXPECTED_IMAGE_ID=<docker image inspect id>` line. `engine.sh:98-99` refuses a mismatch.
- **Image tar** (cap REPRODUCE.md:10-13): `/shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst`, loaded with `zstd -dc <file> | docker load`. The loaded ID can change; it did on 137.
- **NODE_RW_ROOT** (`v0521-base.sh:166-167`):
  ```
  if [[ -n "$YIHOU_JOB" ]]; then NODE_RW_ROOT="${NODE_RW_ROOT-}"
  else NODE_RW_ROOT="${NODE_RW_ROOT-/mnt/m2m_nobackup/yihou_v0521dbg}"; fi
  ```
  - A spur-job node writes to NFS `$WS`.
  - A plain-ssh node (where NFS is read-only) writes to node-local `/mnt/m2m_nobackup/yihou_v0521dbg`.
  - The path must be absolute and contain `yihou` (`:168`).
- **JIT cache.**
  - AITER: `$NODE_RW/cache/<nodekey>/aiter/<role>/<image-id-hex>` (`v0521-base.sh:176` + `engine.sh:105`).
  - triton: `$rw/cache/$node_key/triton/$role` (`engine.sh:127`).
  - Seeding, per cap REPRODUCE.md:14: "Seed JIT caches from `$WS/cache/102/{aiter,triton}` (cp -a) to save ~30 min."
  - `$WS/cache/102/aiter/prefill/` holds `3f917e9b…` and `9d755dd6…`. [inferred] If the new node's image ID differs (as on 137), the AITER subdirectory needs that ID's name.
  - `$WS/cache/137` holds only `agentx`.
- **Model path:** `v0521-base.sh:74-75` sets `SOURCE_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` and `MODEL="$SOURCE_MODEL"`. This is a hard assignment, so an override cannot change it.
- **Topology:** generated automatically as `$WS/harness/topology.yihou.<nodekey>.tsv` (`v0521-base.sh:57-62`).
- **NIC:** `DATA_NIC="${YIHOU_NIC-ens3}"`; set `YIHOU_NIC=` (empty) to resolve the NIC from the IP.
- **Spur shim** (`/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin/ssh`):
  - It is a drop-in `ssh` that runs `exec spur exec "$job" bash -c "$remote"` (`:119`).
  - The job comes from `SPUR_SSH_JOBS="node:jobid,…"` (`:16,45-49`).
  - stdin is replayed via base64, capped at 1 MiB. Binary stdout gets mangled, so copy-back uses base64 tar (`run.v2:95-104`).
  - `SPUR_SSH_ROCM_BYPASS_NODES` / `SPUR_SSH_ROCM_IMAGE` wrap `rocm-smi*` in a container for jobs that hold no GPU cgroup (`:82-90`).
  - The config adds the shim to PATH only when `YIHOU_JOB` is non-empty and the host is not the node; it strips the shim for plain ssh (`v0521-base.sh:39-49`).
- **Where things run:**
  - `run.v2.yihou.sh`, the chains, flush curls and `metrics_sampler` run on the login host.
  - `engine.sh` runs on the node via ssh (`run.v2:91`).
  - The AgentX client is a `docker run` on `CONTROL_NODE` (= `YIHOU_NODE`, `agentx_bench.sh:126`).
  - pc40d instead runs the sampler as a container on the node (`chain_pc40d:15-17`).
  - Kvcap REPRODUCE.md:30 says to run drivers with `setsid nohup … < /dev/null &` and "not on a login host that may be lost".

## (b) How the knobs are passed

- **Overrides.** `YIHOU_OVERRIDES` holds space-separated `KEY=VALUE` items. They are exported before `source $CONFIG` (`run.v2:23-29`) and forwarded through `COMMON` to `engine.sh`, `agentx_bench.sh` and `stop.sh` (`run.v2:48-50`). Every knob uses `:=`, so an override wins (`v0521-base.sh:17-19`).
- **Per-leg knobs that exist:**
  - `PREFILL_/DECODE_MEM_FRACTION`
  - `PREFILL_/DECODE_MAX_RUNNING` (`config.sh:50,66`, both default 64; q1 uses `DECODE_MAX_RUNNING=96`)
  - `PREFILL_/DECODE_CHUNK_SIZE`, `PREFILL_/DECODE_GRAPH_MAX_BS`
  - `PREFILL_/DECODE_TP`, `_EP`, `_DP`, `_DPA` (`v0521-base.sh:85-86`, EP1 default)
  - `PREFILL_/DECODE_CPUSET_*` (`engine.sh:143-155`)
  - `PREFILL_HICACHE_SIZE` (default 450), `PREFILL_HICACHE_RATIO`, `HICACHE_HOST_REGISTER_CHUNK_GB` (E2-target sets 512, prefill env only)
  - `DECODE_HICACHE=0` (`engine.sh:76-78` rejects decode HiCache together with MTP)
- **Extra args per leg:** `PREFILL_EXTRA_ARGS_ADD` and `DECODE_EXTRA_ARGS_ADD` (`v0521-base.sh:243-244`). The base already sets decode `--disable-custom-all-reduce` and prefill `--hicache-size N`.
- **Extra env per leg:** `PREFILL_EXTRA_ENV_BASE` and `DECODE_EXTRA_ENV_BASE` are `:=` knobs. Overlays append after the base is sourced, for example `config.yihou.runshare2.sh:10`: `DECODE_EXTRA_ENV+=" YIHOU_DECODE_RUNNING_KV_SHARE=… YIHOU_RUNSHARE_MAX_WAIT_S=…"`.
- **Extra mounts per leg:**
  - `YIHOU_MOUNTS` is one TSV shared by both legs; the mc4340 overlay hard-sets `mounts.yihou.v2-regbase.tsv`.
  - `PREFILL_MOUNTS_ADD` and `DECODE_MOUNTS_ADD` are per-role TSVs (`engine.sh:170-180`); runshare2 uses `DECODE_MOUNTS_ADD`.
- **CONC** applies to the client only. `bench` uses `$CONC` and `BENCH_TAG` → `$RD/agentx-c$CONC-$BENCH_TAG` (`run.v2:334-336`). The A3/A4 env files keep `YIHOU_ROUND=q1` and change only `CONC=48/52`.
- **Per-leg `--model-path`: not supported as built.**
  - `engine.sh:122` mounts `-v "$MODEL:$MODEL:ro"` and `:253` passes `--model-path "$MODEL"`.
  - `audit_live.v2.yihou.py:44` (`"model_path": env["MODEL"]`) and `agentx_env.py:216` (`if value["model_path"] != env["MODEL"]: raise`) check both legs against the one `MODEL`.
  - The router uses `--router-tokenizer-path "$MODEL"` (`run.v2:75`).
  - `engine.sh:15` sets `role` before sourcing CONFIG (`:32`), so [inferred] an overlay could branch `MODEL` on `$role` for `engine.sh`. Audit and `agentx_env` would then fail for one leg unless they are also changed. `engine.sh:165` mounts `SOURCE_MODEL` separately when it differs from `MODEL`.
  - The expert8 packup (`$PK/yihou-expert8-pd.packup_20260924`, CLAUDE.md:6) is "Actual8 routed experts, TP4/DP4/DPA/EP4 on 138+136, MTP/HiCache off". Its model is a synthetic checkpoint with its own scripts and config (`scripts/config.yihou.expert8.sh`). It is not a knob in this harness.

## (c) Warm-stack protocol and chain sequencing

- **A1 (fresh):** `pipeline.yihou.sh` runs up → census (STOP on fault, OOM or Traceback; requires `rank-PID overlap prefill/decode: none`) → smoke → audit → bench (`:43-55`), ending with `PIPELINE DONE`.
- **Flush** (`chain_q1.yihou.sh:24-30`): decode first, then prefill. Prefill returns 400 while requests are still pending, so it retries every 10 s, up to 60 tries:
  ```
  curl -s -o /dev/null -w '%{http_code}' -X POST "http://$YIHOU_IP:$port/flush_cache"   # 29357 then 29101
  ```
- **A2:** `BENCH_TAG=A2 bash $H/run.v2.yihou.sh bench` (`chain_q1:33`).
- **A3 / A4** (`a3_q1c48.yihou.sh`, `a4_q1c52.yihou.sh`):
  - Each loops on the previous log until it sees `q1 chain done` (or `A3 C48 DONE`).
  - It then sources an env file that differs only in `CONC`, flushes, starts the sampler, and runs `BENCH_TAG=A3|A4 … bench`. It does not stop the stack.
- **Next round on the same node** (`chain_q4.yihou.sh:20-24`): wait for the previous marker → `(source env.q1… && run.v2 stop)` → require that 0 containers matching `'^glm52-v0521-yihou-137-q1-'` remain → pipeline → flush → A2. `pipeline.yihou.sh` also accepts `PRE_STOP_ENV` / `WAIT_PREFIXES` / `GO_FILE` (`:5-9,23-42`).
- **Launch** (cap REPRODUCE.md:23-24):
  ```
  setsid nohup bash chain_q1.yihou.sh > ../runs/q1.chain.log 2>&1 < /dev/null &
  setsid nohup bash a3_q1c48.yihou.sh > ../runs/q1.A3c48.log 2>&1 < /dev/null &
  ```

## (d) Wall time and VRAM gating

- **Measured, `runs/q1/status.tsv`:**

  | stage | start → end (UTC) | duration |
  |---|---|---|
  | `up`: prefill healthy | 10:54 → 11:09 | 15 min (`ok:43x20s`) |
  | `up`: decode | → 11:21 | ~12 min |
  | A1 | 11:21 → 12:43 | 82 min |
  | A2 | 12:43 → 14:03 | 80 min |
  | A3 | 14:04 → 15:24 | 80 min |
  | A4 | 15:28 → 16:51 | 83 min |

  RUNBOOK:286 gives "≈ 20-25 min warmup + 3600 s + reporting".
- **Timeouts:**
  - Prefill health: 180 × 20 s, about 60 min (`run.v2:238`).
  - Then `DECODE_SETTLE_S` (default 120 s) before decode starts.
  - Decode: 120 × 15 s per attempt, attempts a1/a2/a3 (`:168,246`).
- **VRAM gate** (`gpu_idle_gate`, `run.v2:132-148`): it uses `rocm-smi --showmeminfo vram --json` and requires the maximum across GPUs to be under 400 MiB on 2 consecutive checks 60 s apart. It gives up after 80 iterations. `stop.sh` does `docker stop --time 300` and falls back to `docker kill` (`stop.sh:49-52`), never `rm`.

## (e) Hard-coded nodes, jobs and paths that must change

- **`config.yihou.mc4340.sh:23-28`** knows only 102 and 136. **`mc4340n.sh:24-31`** knows 102/136/174/029/137; any other node exits 2.
- **`chain_e2g.yihou.sh:13`:** `run_hostsample.yihou.sh crsuse2-m2m-102 207495 …`; log names are `e2g.*`.
- **Env files** pin node, IP and job:
  - `env.e2g/q1-102/pc40d-102`: `crsuse2-m2m-102 10.245.147.130 YIHOU_JOB=207495`
  - `env.q1/q1c48/q1c52/q4-137`: `crsuse2-m2m-137 10.245.153.247` (plain ssh)
- **Chains and drivers hard-code their own files:**
  - `source $H/env.q1-137.yihou.sh` (`chain_q1:9`), `env.q4-137` (`chain_q4:9,22`), `env.pc40d-102` / `env.pc40c-102` (`chain_pc40d:11,22`)
  - Logs: `$R/q1.pipeline.log`, `$R/q1.A2.log`, `$CAP/runs/q1/…`
  - Container greps: `'^glm52-v0521-yihou-137-q1-'` (`chain_q4:23`), `'^glm52-v0521-yihou-102-pc40c-'` (`chain_pc40d:23`)
  - `chain_pc40d:20` waits on pc40c files. Kvcap REPRODUCE.md:22 says to drop that wait in a new file, not edit in place.
- **Flush and sampler ports** 29357/29101 are literal in every chain and A-script. They follow `ENGINE_PORT_BASE=29101` / `STRIDE=256` (`v0521-base.sh:212-213`). RUNBOOK §3 still lists the old 39101/39357, which is stale.
- **`chain_pc40d:16`:** the sampler image is hard-coded as `infera-sglang:v0521-yihou-1010-mc4340`.
- **`pipeline.yihou.sh:12`** and every chain: `RUNS` / `R` = `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs` (absolute). `WS` is hard-set at `v0521-base.sh:24`.
- **RUNBOOK.v2** node tables (227/067/238/213/138/136, with job ids) are historical and should not be reused.

=====VERIFY=====
CONFIRMED / WRONG / UNVERIFIED (all reads were first-hand and read-only)

1. **`env.q1-137.yihou.sh:4-8` contents (node 137, IP 10.245.153.247, empty `YIHOU_JOB`, `CONFIG=mc4340n`, and the `YIHOU_OVERRIDES` string with CONC=44, DECODE_MAX_RUNNING=96, PREFILL_MEM_FRACTION=0.88, DECODE_MEM_FRACTION=0.95)**: CONFIRMED. The file's comment block at lines 1-3 also says "plain ssh, /shared_nfs ro there -> NODE_RW_ROOT default /mnt/m2m_nobackup/yihou_v0521dbg".

2. **Image-ID `case` in `config.yihou.mc4340n.sh:23-32` (102/174/029 → 9d755dd6, 136 → 9872bbda, 137 → e8e2838f, any other node exits 2)**: CONFIRMED.
   - The statement is `return 2 2>/dev/null || exit 2`.
   - The assignments are hard (not `:=`), per the comment at line 22.
   - The error text it prints is stale: it still says "built on 102 and 136 only".

3. **Model path is hard-set at `config.yihou.v0521-base.sh:74-75` (`SOURCE_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, `MODEL="$SOURCE_MODEL"`)**: CONFIRMED.
   - Nuance: only `YIHOU_OVERRIDES` cannot change it.
   - Two harness configs assign `MODEL=` themselves: `config.yihou.v0521-base.sh` and `config.yihou.102-v0521.sh`. [inferred] An overlay sourced after the base could reassign `MODEL`.

4. **One `MODEL` for both legs (`engine.sh:122`, `:165`, `:253`; `agentx_env.py:216`; `audit_live.v2.yihou.py:44`; router `run.v2.yihou.sh:75`)**: CONFIRMED on content. WRONG on paths.
   - `engine.sh` is at `$H/bench-harness/engine.sh`, not `$H/engine.sh`. The latter does not exist; only `engine.orig.yihou.sh` sits in `$H`.
   - `agentx_env.py` is at `$H/bench-harness/tools/agentx_env.py`.
   - `role` is set at `engine.sh:15` and `source "$CONFIG"` is at `:32`. Both CONFIRMED.
   - `run.v2.yihou.sh:66` also mounts `-v "$MODEL:$MODEL:ro"` into the router container.

5. **`engine.sh` rejects decode HiCache together with MTP**: CONFIRMED, but the line numbers are wrong. The check is at `$H/bench-harness/engine.sh:75-78` ("Decode MTP and HiCache cannot both be enabled").

6. **Ports 29101 / 29357 follow from `ENGINE_PORT_BASE=29101` and `ENGINE_PORT_STRIDE=256` (`v0521-base.sh:212-213`)**: CONFIRMED.
   - The comment at line 212 says prefill uses block 29334-29344 and decode uses 29590-29600.
   - `run.v2.yihou.sh:38-39` derives the ports as `P_PORT=$ENGINE_PORT_BASE` and `D_PORT=BASE+STRIDE`.

7. **NODE_RW_ROOT logic at `v0521-base.sh:166-167`**: CONFIRMED.

8. **Timings from `runs/q1/status.tsv`**: CONFIRMED, full path `/shared_nfs/yihou_glm52/v0521dbg_20261010/runs/q1/status.tsv`.

   | status.tsv line | event | time (UTC) |
   |---|---|---|
   | 1 | start | 10:54:15 |
   | 5 | prefill `ok:43x20s` | 11:09:38 |
   | 6 | decode-a1 | 11:21:26 |
   | 10-11 | A1 C44 | 11:21:42 → 12:43:20 |
   | 12-13 | A2 C44 | 12:43:20 → 14:03:13 |
   | 14-15 | A3 C48 | 14:04:07 → 15:24:22 |
   | 16-17 | A4 C52 | 15:28:10 → 16:51:02 |

   - Every one of these bench runs ends `AgentX passed`.
   - The GPU-idle gate passed at line 3 with `max=283MiB (2 consecutive)`.
   - The output directories are `agentx-c44`, `-c44-A2`, `-c48-A3`, `-c52-A4`.

9. **Expert8 pack-up claim (synthetic checkpoint, not a knob in this harness)**: CONFIRMED, with a path correction. The config lives at `$PK/yihou-expert8-pd.packup_20260924/payload/scripts/config.yihou.expert8.sh`, not under `scripts/`; `scripts/` holds only `collect_env.sh` and `restore_workspace.yihou.py`.

UNVERIFIED (not re-read): the `run.v2` line references at :23-29, :48-50, :132-148, :226-227, :238 and :334-336; the spur shim line numbers; the chain_* and a3/a4 line references; the REPRODUCE/RUNBOOK quotes; the contents of `$WS/cache/102`; and the per-leg `PREFILL_/DECODE_*` knob line numbers.

Important missing facts, all relevant to the user's 8-expert experiments a and b:

- **The expert8 setup is not the current recipe.** `payload/scripts/config.yihou.expert8.sh` sets:
  - `MODEL="$WORKSPACE/model"` (line 6) with `ORIGINAL_MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (line 5)
  - `SERVED_MODEL=glm5.2-synthetic-expert8-yihou` (line 8)
  - `PREFILL_EP=4` and `DECODE_EP=4` (lines 19 and 23)
  - `PREFILL_HICACHE=0`, `DECODE_HICACHE=0`, `DECODE_MTP=0` (lines 25-27)
  - `CONTROL_NODE=crsuse2-m2m-138`
- **It ran cross-node.** The pack-up's CLAUDE.md says "TP4/DP4/DPA/EP4 on 138+136". So it was never run same-node, never with MTP and never with HiCache.
- **What a decode-only or prefill-only expert8 run would need** [inferred]:
  - Per-role `MODEL` in `engine.sh`, plus relaxing `agentx_env.py:216` and `audit_live.v2:44`.
  - Possibly a different `SERVED_MODEL` per leg. `agentx_env.py:312` requires one served-model name across instances.
  - A router tokenizer path that both legs agree on.