# Notes — what / why / how / context

## 1. PD warm-up hang on v3-026 (ofg2h, ofg2h2) — root cause (first-hand)

**What happened.**
- ofg2h: the warm-up started 07:57:04 and never ended. At 08:27:05 it failed with `Initialization failed. warmup error ... asyncio.TimeoutError` in `_send_disaggregation_warmup_requests`, and prefill exited (`results/rounds/ofg2h.fail1`).
- ofg2h2: the warm-up started 08:52:37 and hung again.

**Evidence** (`results/pyspy/ofg2h2/dump3.sched.091626.txt`):
- The DP1/DP2/DP3 schedulers were blocked in `aiter file_baton.wait ← mp_lock ← build_module` (layernorm `forward_hip` in the dsa_indexer) on `/aiter-jit/build/lock_module_norm`.
- DP0 was in a flydsl gemm compile.
- The lock file was dated 07:47:27 and contained `1417 / crsuse2-m2m-v3-026`. That process was the pf90b prefill, killed during bring-up ~07:48.
- Cache key `026` had 4 aiter `.so` in prefill; cache/102, 020, 252, 288 and 089 each have 16. Key `026` came from the old node m2m-026 and was wrongly treated as seeded.

**Fix.**
- Removing the single stale lock (09:17:02) only produced `ModuleNotFoundError: module_norm` in DP1, because the build had never finished.
- What worked: re-seed with `cp -an cache/102/aiter/{prefill,decode}/<I>/. cache/026/aiter/...` (no clobber) and merge triton.
- After that, ofg2h3's warm-up took 61 s.

**Not verified.** Why ofg2h's first warm-up exceeded 30 min. Missing `.so` files being compiled during warm-up is consistent with it, but this was not measured. Also open: 252's 20 min warm-up (ofg4c, cache/252 had 16 `.so`).

## 2. ofg2h3 A1 client failure

- **What:** uv reported `Python installation is missing a _sysconfigdata_ file` and `client-home: Directory nonexistent`. Both came from client caches under the reused key `026`.
- **How fixed:** `mv` `agentx uv-python client-home client-pycache` to `*.bad1-20261015`, and move the failed out-dir to `agentx-c56-A1.fail1`.
- **Then:** `cont2_v3026.yihou.sh` = cont_bench 56 56, then cont_bench - 64. It ran on the live stack, which had not served any request yet.

## 3. Node 136 driver issues

- `run.v3 stop` (ofg4b) hung 50 min in `sync_back`. That step copies node-local logs to NFS over ssh from the login host. The containers were already gone.
  - Only the stop PID was killed; the driver then proceeded.
  - New 136 drivers cap stop at 600 s.
- The KFD gate of `ofs_seq136l.yihou.sh` counts the crusoe `gpuagent` PID, which is always present on 136. It was replaced by a VRAM gate (< 1 GiB on every GPU).
- After ofg4h's prefill came up (10:54:47), the login-host driver on spur007 (`after_ofg4b136l2` + pipeline) was gone; no process was left. The cause is unknown.
  - spur007 was slow (60 s command timeouts) around then.
  - It was relaunched as `ofg4h2` from spur005.
- After a stack stops on 136, GPU0-3 keep 176-285 GB of VRAM for several minutes (11:23). The new driver waits up to 90 min.

## 4. Experimental-design caveats

- Every score is n = 1 on a different node. No replicates were run.
- No like-for-like no-offload baseline exists at C56/C64 (README).
- ofg4b and ofg3b ran without decode tok/detok 4. ofg2h3 and ofg4h2 run with it, so two variables differ between those groups.
- pf90b (same-partition no-offload reference) was stopped at the user's request ("baseline早就有了").

## 5. Operations log

- 05:00: oai 214829/215369 released at the user's request.
  - Procedure: kill our driver PIDs by number, `run.v3 stop`, then `docker rm` of our own exited containers by name.
  - ofg4 A1 and ofg2 A2 were lost.
- 05:4x: burst holds 217125 (v3-020) and 217126 (v3-026) started on `amd-spur-v2`.
  - Foreign GPU containers were `docker stop`-ped (not removed), per user authorization 07:1x: chuschen's `glm52x_*` on v3-020, guanchen's `crusoe-sgl-q36` on v3-026.
- `analysis/ledger/poll.md` has the minute-level record.

## 6. Login-host limits (update 13:30)

- **What:** login hosts police memory per user (user: about 4 GB with kill/evict). Our sync_back held a whole tar archive, base64-encoded, in one bash variable, and stalled under memory-cgroup throttling (D state, `mem_cgroup_handle_over_high`).
  - This happened twice: the ofg4b stop (50 min, spur007) and the ofg4h2 census (50 min, spur005).
- **How to avoid:**
  - Use the streaming copies (`run.v3s` / `pipeline.v3s` / `chain_cap2ls`).
  - Keep login-host drivers few, and spread them over spur005/006/007.
  - Prefer drivers ON a held compute node, where the NFS is writable and no sync_back is needed.
- spur003/004 still answered "Remote commands are not allowed on this SSH-only host" to `ssh <host> '<cmd>'` at 13:2x. The user says they are usable, presumably interactively.
