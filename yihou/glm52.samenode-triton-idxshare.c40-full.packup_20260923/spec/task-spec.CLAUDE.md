# Task: same-node 1P1D P4D4 — triton DSA + index_share ON, AgentX CONC=40 full

Re-run the same-node CONC=40 point on `crsuse2-m2m-137`, this time with
**triton DSA backends** and **`index_share_for_mtp_iteration = true`**, in
**full mode** (3600 s, warmup 10/lane) with prefill HiCache on. The previous
attempt died at 89 % of the profiling window and produced no aggregate, so
**finishing the window is part of the deliverable.**

Task of record: `bench/glm5p2_pd/results/yihou-triton-idxshare/spec/mission.md`.
That file, not this one, is the authority on scope.

## Background — what is already settled

Same-node 1P1D P4D4 works. Getting there took three fixes, all invisible
cross-node and all in the **vendored** harness (the tracked repo is untouched):

1. **`tools/topology.py`** rejected two rows on one node/IP.
2. **SGLang's internal port block** derives from `--port`; at the harness's
   stride of 1 the two legs overlapped and prefill died at
   `zmq.error.ZMQError: Address already in use`. Fixed with
   `ENGINE_PORT_STRIDE` (default 1, so cross-node is unchanged).
3. **The two legs' RCCL inits race** on one host. A same-node start gate in
   `launch.sh` addresses it — **but see the open item below.**

Also settled, by transfer rather than inference: **same-host KV does not use
RDMA.** Mooncake advertises the GPU segment as `"rdma,hip"` regardless of
`MOONCAKE_PROTOCOL`, matches the port-stripped host, and prefers `hip` (4) over
`rdma` (2), so KV moves over **GPU-IPC on XGMI** — measured at **~57 GB/s**
between any GPU pair, cross-NUMA included, against ~40-42 GB/s for the ionic
RDMA path.

Three hard rules follow, each a way to break a working deployment:
**never** `MC_DISABLE_HIP=1`; **never** `MC_USE_HIP_IPC=0` (fabric mode
segfaults `register_memory` at every size); and `MC_DISABLE_HIP_TRANSPORT=1` in
`config.sh:89` is a **dead name** — do not "fix" it into the real spelling.

## Open problems — do not treat these as solved

- **The start gate is insufficient.** It fired and decode still died with
  `NCCL error` / `hipIpcGetMemHandle failed : invalid argument`; a manual retry
  ~17 min later worked, with HiCache identical. So HiCache is exonerated and the
  gate's release point (`/health` 200) is too early. **UNSETTLED** whether a
  longer wait is a deterministic fix or the retry won a race. N=1.
- **Why `crsuse2-m2m-276` segfaults a solo leg.** 137 and 276 are
  software-identical; 276 fails, 137 serves. Localised to `scheduler.py:1894`
  (a native HIP stream-handle read — the signature of prior context corruption).
  Cause unidentified. The spur job on 276 is held as a placeholder.
- **Why the last benchmark client died at 89 %.** Container was `--rm`, so no
  exit code. Not OOM (846 G of 2751 G used).

## Context — the risk this run carries on purpose

`glm52.p8d8.agentx-sweep.packup_20260920` records this shape with **IndexShare
ON faulting at 1 h 26 m**, versus **13 h 33 m fault-free with it off**. The user
directed IndexShare on. **Do not silently turn it off.** A fault is a finding.

## Key references

| path | what it gives |
|---|---|
| `yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh:87-89` | the triton DSA settings this run copies |
| `yihou/glm52.samenode-p4d4.agentx-c40.packup_20260922/` | the Phase-A kit: harness patches, the three same-node defects, the mooncake/XGMI findings |
| `yihou/glm52-agentx-t2sweep-simacc.packup_20260920/results/t2-summary.csv` | the cross-node `t2f` c40 reference: 158,390 tok/s, 19,799/chip |
| `bench/glm5p2_pd/results/yihou-samenode-p4d4/working_process.md` | 20 rounds of how the same-node shape was made to work |

## Core principles

1. **Verify first-hand.** Code read, command run, or official source.
2. **Suspend, don't conclude.** Leave a question open rather than guessing.
3. **One variable at a time.**
4. **Docker, not host.**
5. **Deletion rule.** Never delete a file whose path lacks `yihou`.
6. **Nothing outside the workspace.** The harness is vendored and patched there.

## Notable details

- **`engine.sh` has no triton cache handling.** The triton JIT cache is
  container-private, so there is no same-node race but there **is** a cold
  compile every bring-up. Slow first use is not a hang.
- **The model's `config.json` already has `index_share_for_mtp_iteration: true`**
  — "on" is the model default. Set it explicitly anyway so it is checkable in
  `/get_server_info`.
- `--dsa-topk-backend` accepts only `{sgl-kernel,torch,flashinfer}`; `flydsl` is
  accepted nowhere. Verified against this image.
- **`agentx_bench.sh` validates the live deployment against the config** and
  refuses to start on a mismatch (e.g. `live hicache=True, config expects
  False`). Pass the same overrides to `agentx_bench.sh` as to `launch.sh`.
- **The idle-gauge trap.** A rank that served no decode tokens reports
  `spec_accept_length 0.0`. Trust only ranks whose counters moved.
- **Never `pkill -f <pattern>`** from a shell whose own command line contains the
  pattern — it kills the shell. This has bitten twice.
- **zsh eats `-v $M:$M:ro`** — `:r` is a history modifier that strips the
  extension. Always quote: `-v "${M}:${M}:ro"`.
- Settle the GPUs after `stop.sh` and verify the idle baseline (~284 MiB/card,
  no KFD processes) before relaunching. HiCache host-pool teardown lags.

## Repository conventions

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer. Commit
with `-s`, always, signed off **as yourself** — never a bot, assistant, or
colleague identity. Cherry-picks do not inherit it; use `git cherry-pick -s`.

## Language

Work in English (code, comments, commits, notes). Report to the user in Chinese.
