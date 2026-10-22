# Gotchas (what / why / how / context)

## 1. A2 after a lost A1 is not the standard protocol
- **What:** wts48's A2 ran after a ~50-min partial A1 + flush, not after a full 3600-s A1.
- **Why it matters:** the scored number is defined as the run after a full fresh A1 (warm host/JIT state, HiCache flushed). A shorter A1 is a different warm-up.
- **How:** recorded as a protocol deviation; compare wts48 A2 to q1 A3 with that caveat.
- **Context:** login-host driver terminated at 04:47, cause unknown (notes/incidents.md).

## 2. Two benches in one round write the same metrics file
- **What:** `cont_benchl` names the sampler file by tag (`$N/runs/<round>/metrics.<tag>.tsv`) and the sampler opens it in append mode.
  wts64 was launched with tag A2 on round wts48 → `results/metrics/wts48.metrics.A2.tsv.gz` holds wts48 A2 (from epoch 1791955908 = 05:31:48)
  **and** wts64 (from 07:17:10) samples, and was still growing when copied (~07:58).
- **How to split:** filter column 1 (epoch seconds) at 1791962230 (07:17:10 UTC).
- **Fix for next time:** use a distinct round name or tag (e.g. `A2c64`) for a second bench on a live stack.

## 3. Same tag A2 for wts64's output dir
- The C64 bench on the wts48 stack writes to `$R/wts48/agentx-c64-A2/`. It is the **3rd** use of that stack (partial A1, A2 C48, then C64), not a fresh-then-flushed A2.

## 4. Node 136 is plain ssh with read-only NFS
- **What:** on 136 the harness writes caches, tmpfs and full server logs under the node-local `NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg`;
  drivers run on the login host (`chain_cap2l`, `cont_benchl*`, `after_wts64_pf94`).
- **Effect on this packup:** the NFS copies `$R/wts48/server-logs/*.log` stop at 03:57 (startup only). `/mnt` was not reachable for packing →
  **no wts48/wts64 runtime server logs are in this kit** (gap).

## 5. The 0.92 job died at ~24 h RunTime ([inferred]: 24 h QOS MaxWall)
- **What:** `scontrol update TimeLimit=36:00:00` was accepted but the job ended at 1-00:00:24 with NODE_FAIL (scontrol, poll.md 05:11).
  sacct (read 10-14 ~08:10) shows 211169 `COMPLETE -1:0`, Elapsed 1-00:00:23, TimeLimit UNLIMITED; the NODE_FAIL / NodeDown state is from scontrol as logged in poll.md. The MaxWall cause is [inferred] from the 24 h coincidence, not from a scheduler message.
- **How to avoid:** plan rounds so that A2 ends before the hold's 24 h mark; check `RunTime` vs 24 h before arming a handover.

## 6. Self-matching pgrep kills the caller
- **What:** a stop helper that `pgrep -f`/`pkill -f`s a pattern contained in its own command line kills itself first.
- **How:** use `run.v3 stop` (container-based) or pgrep with a pattern that cannot match the helper (e.g. `[c]ont_bench`).

## 7. Decode `/flush_cache` returns 400 while a bench is running
- 400 means "pending requests" — expected if a bench is already in flight. A loop of 400s is evidence of a concurrent bench, not a stuck server
  (07:22 relaunch). `cont_benchl2` tolerates decode-flush 400 after 6 tries; prefill flush stays mandatory.

## 8. write_through_selective semantics (code read)
- sglang v0.5.21: `write_through_selective` sets `write_through_threshold` to 2 instead of 1 → a radix node is backed up to the host tier only
  after its 2nd hit (`unified_radix_cache.py:516-517`, `unified_tree_core.py:1078-1084`; source copy
  `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang`). Live server-info of wts48 confirms `hicache_write_policy=write_through_selective`.

## 9. Correctness is waived in every scored row
- All score rows use `SGLANG_SIMULATE_ACC_LEN=3.61` (+ `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`). cx90 is the only round with simulated
  acceptance OFF (dryrun diff vs pf90 = only the 3 SIMULATE envs); it is a correctness check, not a throughput run.
