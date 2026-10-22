# Notes — gotchas, wrong turns, caveats (what / why / how / context)

1. **The A/B has n = 1 per cell.**
   What: each score is one 3600 s AgentX run. Why it matters: the deltas (-2.2 % C56, -0.3 % C64) are of the same size
   as the earlier warm-replicate spread on another node (21,681-22,114 at C40, 2.0 %). How to resolve: replicate A2 on
   one node, alternating patch/no-patch stacks. Context: the user's question was "less regression than the old sweep";
   neither this A/B nor t2f answers it cleanly (next note).

2. **No C40 baseline at this config.**
   What: the user's expectation is about regression (C40 -> C56/C64). Only C56/C64 were run at HiCache ratio 1.5 +
   q1 config on 085. Why: the chain was C56 A1 -> C56 A2 -> C64 A3 to keep the round at ~5 h. How: add a C40 A2 at the
   same config on the same node to measure the regression of THIS config. The t2f sweep differs in 5+ variables
   (sglang 0.5.19, cross-node, decode mem 0.85, max_running 128, tilelang DSA), so it is context, not a control.

3. **"Original C40 HiCache capacity" = `--hicache-ratio 1.5` with no `--hicache-size`.**
   What: `PREFILL_HICACHE_SIZE=0 PREFILL_HICACHE_RATIO=1.5`. Why: E2-target defaults to 450 GB (`:=`), so the size
   must be overridden to 0 or the ratio is ignored (size > 0 wins in engine.sh). Result: 3,649,664 host tokens/rank
   (= 1.5 × the 2,433,088-token prefill device pool at mem 0.88), vs 3,423,488 in t2f — same ratio, ~6.6 % more tokens.

4. **Warm-stack protocol: A2 is the scored run; A3 is a 3rd run on the same stack at a different CONC.**
   CONC is client-side only (server args identical), so A3 C64 reuses the A2 stack after flushing decode (:29357)
   then prefill (:29101). Fresh (A1) numbers are not comparable to warm ones.

5. **Drivers run ON the compute node, not on the login host.**
   What: `spur exec <job> bash -c "setsid nohup bash chain_cap2 ..."` with `$N/shimbin/ssh` first on PATH. Why: on
   2026-10-12 the login-host session was lost at ~17:18 and killed the drivers (pc40d A2 never ran); the spur
   controller was also intermittently refusing connections (`failed to connect to spurctld`) on 2026-10-13/14. The shim
   runs `ssh <thisnode> cmd` locally and refuses any other host.

6. **`stop` exits non-zero on a clean stop.** `tb56n.handover.log`: "cleanup completed with errors" / "stop rc!=0",
   yet 0 containers left and 0 KFD procs; `stop_then_chain` therefore checks containers + KFD itself instead of rc.
   VRAM showed 49-95 % allocated right after stop with no KFD process — drain lag, gate on KFD count.

7. **Foreign containers.** 085 carried another user's orphan `crusoe-sgl-q36` (loop that can relaunch sglang on
   GPU0-5). It was docker-stopped (not removed) with the user's approval. `chain_cap2` refuses to start if any KFD
   process exists. Never stop a foreign container without asking.

8. **Patch log volume.** At C56/C64 the cap blocks continuously: 266,980 log lines (rate-limited to every 100th block
   per DP) made the tb56 decode log 89 MB vs 38 MB without the patch. Heavy greps on it should run on a compute node,
   not the memory-policed login host.

9. **`retract` grep is misleading.** `grep -ci retract` on the decode log matches ~23k lines in BOTH rounds (server_args
   and periodic stats lines), not retractions. Use the /metrics `num_retracted_reqs` gauge (0 in all six runs).

10. **Correctness is waived by construction** (`SGLANG_SIMULATE_ACC_LEN=3.61`, match-expected). `spec_accept.after-bench.txt`
    reports the forced acceptance; it is not a quality result.

11. **Container naming / node key.** Containers are `glm52-v0521-yihou-<node suffix>-<round>-{etcd,prefill-0,decode-0,router}`;
    JIT caches are per node and role under `$WS/cache/<node suffix>/`. A new node needs its own cache dir (seed from 102)
    and an image-ID line in a mc4340n* config.
