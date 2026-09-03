# Scripts — copied verbatim from the run

These are the scratch scripts that actually produced the result. Their headers
carry the reasoning; nothing has been paraphrased or tidied. The productised
form of the same recipe now lives in the repo at `examples/sglang_mix_glm5.3/`
(commit `37f2a8f`) — reach for that if you are a new user rather than a
reproducer.

| script | role | note |
|---|---|---|
| `mix_up.sh` | the whole bring-up: teardown → GPU reset gate → container → etcd → infera worker → kv-aware router | `ETCD_PORT` defaults to **12379** (2379 was held by a foreign host etcd). Binds the model dir explicitly. |
| `mix_worker.sh` | the infera worker itself, run **inside** the container | This is where the recipe lives. Its header explains every non-default flag; `SHARED_EXPERT_FUSION=0` (the default) is what emits `--disable-shared-experts-fusion`. |
| `reset_gpus.sh` | pre-bring-up gate: our GPUs are back to VRAM baseline | Resolves ownership **per PID** from `/proc/<pid>/cgroup`. Kills only our own containers' processes; **aborts** rather than touching a foreign one. |
| `mix_smoke.sh` | health checks against the **router** (`:8100`) | Reads the blocks, not the exit code: workers, models, a coherent answer, `17*23`, both memory pools, fault scan, AITER/mHC dispatch. Set `SERVED=glm5.3-flash-mxfp4`. |
| `fixlen_sweep.sh` | the fixed-length sweep that produced `results/fixlen_p*.csv` | Targets the **worker** (`:30000`), not the router. Three non-default flags are load-bearing — see its header. |
| `rung.sh` | one rung of the Track-F ladder: bare sglang, no infera, no etcd, no router | `RUNG=0` vendor image + bind-mounted source, `RUNG=1` our base + bind-mounted, `RUNG=2` our built image. Exactly one variable differs between adjacent rungs. |
| `collect_env.sh` | read-only host snapshot (GPU, driver, RDMA rails, docker digests, CPU/RAM) | Produced `../env_n0433.txt` and `../env_n0133.txt`. Not part of the experiment; included so the environment can be re-captured the same way. |
| `decode_probe.sh` | cheap decode probe used during the CUDA-graph debug loop | **Its numbers are not trustworthy above low concurrency** — at conc 32 it reported 350 tok/s while the engine's own log showed ~2398 tok/s with an empty queue. See `../notes.md` §2(b). Kept because it is part of the record, not because it should be used. |

Every script takes its configuration from environment variables and has no
hard-coded host. The invocations that produced the packaged numbers are in
`../REPRODUCE.md`.
