# Notes: decisions, the readings behind them, and what not to repeat

All times are UTC on 2026-10-15, on `smc300x-ccs-aus-a16-19` with `transport: local`.

## Decisions

**Card 6, because the operator named it.** `E2E_GPU_DEVICES=6` was a list, not
`none`, so the choice was not mine. At the 06:50:12Z preflight
(`results/preflight.json`), card 6 read **284 MiB used, 191.7 GiB free of
196,592 MiB, 0 % use**. All eight cards read 284 MiB at that moment.
`rocm-smi --json` listed eight `card` keys (card0 to card7). So the record says
`fixed.gpu_count: 8` and `fixed.gpu_devices: [6]`. Both bring-ups in this task,
including the self-test, bound card 6 and nothing else (`HIP_VISIBLE_DEVICES=6`
in the worker, `results/selftest/seams_and_mounts.json`).

**CUDA graph ceiling `E2E_KIT_CUDA_GRAPH_MAX_BS=32`, sized for the heaviest
downstream consumer: stage 2's Mooncake trace replay.** That replay is
closed-loop, and `shared.yaml` caps its concurrency at `E2E_MAX_CONC`
(`max_conc:-32`). The brief records a decode concurrency of 25.42 for it on rung
2e, above this stage's own load of 16. A closed-loop load cannot exceed the
concurrency it offers, so a ceiling equal to the cap covers it with no margin.
The value 32 also covers `check_deploy_serves`'s concurrency-16 load. I read the
ceiling back rather than assuming it:

- `get_server_info` reports `cuda_graph_max_bs_decode: 32`;
- the worker log shows `bs=[1, 2, 4, 8, 12, 16, 24, 32]` captured in 34.60 s
  using 0.31 GB;
- under the concurrency-16 load, all 193 logged decode batches say
  `cuda graph: True` and none says `False`;
- `graph_ceiling.check` on the live engine argv gives `ok: true`, ceiling 32
  against effective decode concurrency 14.593 (max 16).

**If a consumer drives more than 32 concurrent decodes, it must raise this
parameter.** If a caller sets the parameter, the kit uses that value as given.

**MIX is selected by omission.** No `--disaggregation-mode` flag is passed
(`logs/engine_argv.txt`). Two components reported the mode independently: the
router said `"disagg_mode": "mixed"` and the worker said `disagg=DisaggMode.MIXED`.

**Served name `Qwen3.6-27B`.** `E2E_SERVED_NAME` was empty, so it resolves to
`E2E_MODEL_NAME`, which was given as `Qwen3.6-27B` with no org prefix. The kit
passes `--served-model-name` explicitly. infera forwards it to sglang's parser,
which is why the flag is absent from infera's `--help`.

**No image build.** `infera:sglang-gfx942-glm52` was present at
`sha256:c05d2bebe442…0818` (created 2026-08-20). Despite `glm52` in the tag, it
carries sglang 0.5.16, which serves `Qwen3_5ForConditionalGeneration`. So
`fixed.dockerfile` is null.

**Model flags.** `--reasoning-parser qwen3` comes from `E2E_PARSER_ARGS`. No DSA
flags are passed (`E2E_DSA_ARGS=none`; the model has no DSA path). No MoE or EP
flags are passed, because the model is dense. Compared with the repository
example `examples/sglang_mix_glm5.3`, I added `--enable-metrics`, so later stages
can read `sglang:*` series. The `metrics_exposed` probe passes.

**Loopback.** The operator gave `E2E_NODE_IP=127.0.0.1`, so the router, worker,
etcd and KV sockets bind and advertise 127.0.0.1. On a shared host that is safe,
because etcd has no auth. Every consumer in this flow is on this node.

## Measured, worth knowing

- **Cold start was 276 s to READY through the router, the same on both
  bring-ups, each on the first attempt.** For the first bring-up:
  - weights took **29.59 s** for 51.06 GB from local NVMe ext4;
  - then the mamba state pool (288 slots, ssm 40.64 GB, conv 0.79 GB);
  - then the KV pool (751,733 tokens, 22.94 GB each for K and V);
  - then decode graph capture (34.60 s);
  - `Application startup complete` came at 06:55:01, but the worker's
    `/health` returned 200 only at 06:57:19. The ~2.3 minutes between are
    AITER JIT builds during warm-up (`[aiter] start build …` in the log), while
    the log repeats `GET /health 503`. The JIT cache lives in the container
    layer, so every fresh container pays it again.

  `E2E_KIT_READY_TIMEOUT=1800` is about 6.5x the measured time, as slack for a
  cold page cache. Weights read over NFS need a larger budget.
- **The AITER JIT prints a scary line that is harmless:**
  `clang (LLVM option parsing): Unknown command line argument '-amdgpu-coerce-illegal-types=1'`
  followed by `failed to execute: …clang++`. AITER is probing a flag and falls
  back. The build continued, and the engine served.
- **`mem_fraction_static` reads back as 0.7225, not the 0.85 passed.** sglang
  rescales it for the hybrid (mamba) model. Trust `get_server_info`, not the
  argv. The resolved `max_running_requests` was 57.
- **Reasoning is separated correctly.** Of 160 completion tokens, 157 were
  reasoning, and `content` is `"\n\nParis"`. With `max_tokens: 2048` the request
  finished with `stop`.
- **Load smoke, not a benchmark.** 1k in, 1k out, concurrency 16, 180 s,
  through the router, using the validator's own `aiperf_synthetic.sh` and
  `summarise.py`. Results:
  - 112 requests, `error_summary: []`, mean OSL 1024.04;
  - effective concurrency 15.99 on average, 16.0 at p99;
  - ITL 23.92 ms on average, 25.4 ms at p99;
  - TTFT 2347 ms on average, 4501 ms at p99;
  - 610.7 output tok/s.

  The validator's `judge_load` returned no faults; its floor was 96 completions.
- **One port sits outside the band.** sglang's scheduler opens its own
  kv-event publisher on an OS-chosen port and the router registers it as
  `kv_events_endpoint` (`tcp://127.0.0.1:42841` in this run). The OS assigns it,
  so two copies cannot collide on it. infera's own KV plane uses the band
  (`--kv-events-bind` at base+4, snapshot at base+5). The repository example
  hard-codes 5557 and 8801 for those two, which would collide as soon as a second
  copy ran, so the kit puts them in the band.
- infera logs `KV host IP pinned to RDMA rail 192.168.50.6 (via private NIC)` at
  start. With a single TP1 MIX worker nothing uses that rail, and serving
  through 127.0.0.1 was unaffected.

## Things that went wrong, and what was done

- **A co-tenant took all eight cards between my two bring-ups, so the first
  self-test attempt was refused.** At 07:02:09Z the container `lingxun-dsv4`
  started `torchrun --nproc_per_node 8` (a Primus Qwen3-30B-A3B pretrain). My
  first teardown finished 13 s later, at 07:02:22Z, and its drain check still
  read card 6 at 284 MiB, because the co-tenant had not allocated yet. At 07:02:45Z
  `deploy.sh` read **card 6 at 5778 MiB** and aborted before creating anything,
  with `card 6 already holds 5778 MiB, so it is not free. Not substituting
  another card` (`logs/selftest_deploy.attempt1-card6-cotenant-refused.log`).
  The co-tenant then rose to about **115 GB on every card** (card 6 read 115,024
  MiB at 07:03:22Z). After that its usage fell and moved around, and its
  `torchrun` process was gone by 07:08:30Z. Card 6 read **284–285 MiB at every
  15-second sample from 07:04:16Z to 07:08:07Z**. I did not touch the co-tenant
  and did not substitute a card. The self-test started at 07:08:42Z, read card 6
  at 285 MiB at that moment, and succeeded. The brief warns that "a card free
  this second is not a card that is yours". Here that happened within seconds of
  a clean teardown, and `deploy.sh`'s free-card check at bind time caught it
  where the teardown's reading could not.
- **A `pgrep -fc 'torchrun …'` count of 1 misled me for a moment.** It matched
  its own shell command line. `ps -p PID` is the reliable check.
- **Another agent's containers appeared during this run:**
  `yihou_selftest_engine` and `yihou_selftest_etcd`, label
  `e2e_kit=qwen36_mix_deploy_kit`, created 07:12:22Z, `HIP_VISIBLE_DEVICES=7`.
  They are not mine and were left alone. They show why teardown in this kit goes
  by the `infera_e2e_run=TAG` label and never by name pattern.
- No `HIP failure: 'invalid argument'` or NCCL failure occurred. At TP1 there is
  no collective, so that intermittent multi-rank failure has nothing to act on.
- Throwaway containers I started (help check, version check, aiperf) were all
  named with `yihou_l2a_r16aea3_…` and ran with `--rm`.

## Shape choices a reader might question

- **The callability self-test ran after the first deployment was torn down, on
  the same card, not beside it.** The brief requires that the device set stay
  within `E2E_GPU_DEVICES=6` for the whole task. Two engines at memory fraction
  0.85 do not fit on one card. Isolation was shown against the running
  self-test copy instead (`results/selftest/`):
  - a `teardown.sh` under the other tag removed nothing (`"removed": []`);
  - a second `deploy.sh` under the same tag refused on port 18121;
  - a fresh tag on the same band refused on the port;
  - a fresh tag and band reusing the container name refused with
    `container name … is taken; not removing it`;
  - after all four, the self-test was still `Up` with `/health` OK.

  Serialising the two bring-ups cost about 6 minutes for the self-test, plus
  about 6 minutes waiting for the co-tenant to clear.
- **`deploy.sh` blocks until the router is ready.** It calls `wait_ready.sh`
  itself, because `check_deploy_serves` calls only `deploy.sh` and reads the
  handshake straight away.
- **On failure, `deploy.sh` removes the containers it started in that
  invocation, by ID.** It hands the work root back to the user first. The logs
  stay on the host. `E2E_KIT_KEEP_ON_FAILURE=1` keeps the containers for
  inspection instead. It never removes a container it found already existing.
- **`teardown.sh` hands root-written files back.** It runs `chown` on the work
  root from inside the container before removing it, so the work root is
  deletable by the calling user. That was verified: `work_root_reclaimed: true`,
  and the logs were owned by `yihou` afterwards.
- **Measurement mounts.** Later stages `docker exec` in and cannot add mounts.
  So the engine container mounts three roots read-write at the same path:
  - the run root, derived from the kit's own path up to `runs/RUN_ID`;
  - the scratch root `/var/tmp/yihou/demoladder/l2a/scratch`;
  - the work root.

  Writes from inside appeared on the host for all three. This host's docker has
  **no authorization plugin** (`docker info` → `Plugins.Authorization: null`).
  So `lib_mounts.sh` keeps the cluster's measured forms (`/shared_nfs`,
  `/home/USER`, `/mnt/m2m_nobackup/...`) and adds same-path binds of other
  non-top-level directories only when no plugin is present. With a plugin it
  refuses and names the accepted forms.
