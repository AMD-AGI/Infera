# Notes — what / why / how / context

## 1. Preemption decided almost everything
- **What:** 34 rounds 10-16 → 10-19, 26 lost. Scored A2 points came only from v3-010 (ofg3h2) and v3-006 (5 rounds). Every other round was lost; see `results/rounds/<round>/` and `analysis/ledger/poll.md`.
- **amd-spur:** of the 11 holds that started (219097, 220567, 220644, 221260, 221521, 221627, 221725, 221768, 222214, 222352, 222575), 10 ended at 20:00–20:03 and 221725 at 31:30. Partition config read first-hand: `PreemptMode=CANCEL PreemptExemptTime=1200`. Bring-up alone takes 25–35 min, so no amd-spur round ever reached a bench. Several amd-spur submits also looped in `JobHoldMaxRequeue` / `JobLaunchFailure ("gpu/resource allocation mismatch", "agent rejected launch")`.
- **amd-spur-v2:** lifetimes 20 min → 19.5 h, no visible pattern. One round needs ~3.5 h.
- **After a preemption the containers stay "Up" but the engines are dead** (detokenizer heartbeat stops at the preemption second; /health no reply). Reuse of the surviving stack was tried once (ofg2h9) and failed → always fresh bring-up.
- sacct for 221726 (ofg2h19) printed `COMPLETED 15:59:27–16:43:45` while our logs on that node kept growing to 17:43:40. Time base of that sacct output not verified.

## 2. /shared_nfs quota (EDQUOT) — ofg4h4 lost, rerun as ofg4h5
- **What:** 10-18 18:18–~19:50 the 420 T volume was full ("Disk quota exceeded"). The ofg4h4 chain driver and the `docker logs -f` writers vanished from the node; the bench clients kept running.
- **Effect (measured):** `profile_export.jsonl` has exactly one NUL-filled line in A1 (line 2473, 398,830 B) and A2 (line 558, 51,158 B). `process_agentic_result` fails with `JSONDecodeError`, so no `agentx_conc64.json`.
- **Unofficial rescoring:** copies with the NUL line dropped (originals untouched), processed by the same `process_agentic_result` with the run's `runtime.env` (exported verbatim; `source` strips the JSON quotes of `KV_OFFLOAD_BACKEND_METADATA`): A1 16,905 / A2 17,693 (`results/unofficial/`). The NUL block replaced an unknown number of records, and A2 ran via `cont_bench` → not comparable to official scores.
- The earlier fill (10-17 morning) only paused work.

## 3. Driver / controller hiccups
- spur controller unreachable 10-17 16:53–18:12 and briefly several times on 10-18: `spur exec` fails with `failed to connect to controller`; sbatch "not the Raft leader". Retry from another login host (spur005/007) worked.
- `run.v3s stop` returned rc=1 ("cleanup completed with errors") after each v3-006 round; every time the node then showed 0 containers and 0 KFD processes before the next launch.

## 4. PD warm-up watcher
- `warmup_pyspy.yihou.sh` dumped once: ofg2h12 warm-up 01:48:59 → 01:54:01 (~302 s, just over the 300 s limit; dump in `results/pyspy/ofg2h12/`).

## 5. Node effect (open)
- G2 C56 A1: v3-008 21,766 vs v3-006 19,746 and 19,650 (two allocations). Same argv (tokdiff = node/IP). Not explained; an A/B on two nodes in the same window would answer it.

## 6. Comparisons that are NOT like-for-like
- No-offload references at C56/C64 (tb56n, tb56) used HiCache ~154 GB and prefill mem 0.88, vs 450 GB / 0.90 here.
- C48 rows and ofg4b A3 C56 ran decode tok/detok 1/1; all rows from this kit ran 4/4.
- ofg4b ran on 136 with its own image build (`9872bbda…`).
