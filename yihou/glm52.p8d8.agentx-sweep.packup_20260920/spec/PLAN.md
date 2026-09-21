# PLAN — P8D8 vs P4D4, then sweep the winner

Plan of record. Mission: `mission.md` in this directory.

## Decisions taken, with the evidence behind them

| decision | evidence |
|---|---|
| Nodes **137 (P) + 136 (D)** | 135/138 are mid-run by another session; 135's GPU[1] is held by a root k8s pod, so 135 cannot serve an 8-GPU prefill leg at all. 137/136 have all 8 free once phase 1 is torn down. |
| Image **`v0519-yihou-0917-nextnfix-hicache`**, transferred not rebuilt | Read inside the image on 135: NextN fusion fix at `glm4_moe.py:1490`; #37152 markers present in all three files (`hicache.cuh` 8, `hicache.py` 2, `mha.py:110` widened to `_is_cuda or _is_hip`); sglang `0.5.19.dev20260917+ga9fb1c3238`. Router patch baked in — the live router runs `INFERA_PD_DP_RANK_AFFINITY=true`. |
| Private worktree `infera.yihou.glm52.p8p4` @ `ad3b85d3` + copied patched bench scripts | `infera.glm52.pd` drives the live 135/138 run and is dirty with the patch set; running or editing there would collide. |
| `SGLANG_OPT_USE_TOPK_V2=false` needs no change | already the default at `config.sh:93` / `config.full.sh:101`, forwarded at `engine.sh:123`. **Still to be confirmed in the live container env.** |
| #37152 needs no build step | already inside the chosen image; markers verified. |

## Open question to settle **before** the timed runs

**Does TP8 need `--disable-custom-all-reduce`?** The five-round A/B that
established it was P4D4/TP4 only; phase 1 measured coherent output at real accept
2.57 on TP8/DP8. Settle it with a 16-request `temperature=0` probe, **simulation
off**, on each shape. Cost ~2 min per shape. Then use the same answer for that
shape's timed runs, and say which arm each number was taken in.

Rationale for probing rather than copying: the flag is an optimisation being
given up, its cost is unmeasured, and the P8D8-vs-P4D4 verdict must not be
decided by a flag that only one of the two shapes needs.

## Phases

### A — bring-up (parallel)
1. **Image**: `docker save | load` the image 135 → 137, 136; verify the same image
   id on both and re-read the three #37152 markers **on the running nodes**.
2. **Hardware**: tear down phase 1's deployment on 137/136; confirm all 8 GPUs
   idle on both; survey rails/GID/NUMA for GPUs 0-7; emit `topology.yihou.tsv` and
   the 8-entry `RDMA_DEVICE` JSON map; run `preflight.sh` and the pinned-NIC probe.
3. **Configs**: author `config.yihou.p8d8.sh` and `config.yihou.p4d4.sh`.

### A2 — scheduling decision: where `--disable-custom-all-reduce` belongs

Taken by the leader 2026-09-18 12:55 UTC, recorded because it affects fairness.

Both `SGLANG_SIMULATE_ACC_LEN` and `--disable-custom-all-reduce` are **launch-time**
settings — the first becomes a module constant in `spec_utils`, the second an argv
entry — so changing either costs a full engine restart (model load + CUDA-graph
capture). That makes the ordering a real cost decision, not a formality.

**The two `fast` points both run WITHOUT `--disable-custom-all-reduce`.** Reasons:

1. **Fairness dominates.** The flag gives up an optimisation, so it moves throughput.
   Running one shape with it and the other without would confound the single number
   the entire branch rule depends on.
2. **Simulation already waives correctness.** With `DECODE_SIMULATE_ACC_LEN=3.61` the
   acceptance is forced after verify, so garbled draft content cannot affect the
   timing being measured. A simulated run is not correctness evidence either way —
   which is exactly the disclosure the reference packup makes about its own C32/C40.
3. It reproduces the reference's configuration, keeping our numbers comparable to it.

**The probe still runs, per shape, before its timed run** — a separate launch with
simulation OFF and no flag. It is not there to change the `fast` runs; it is there to
answer the mission's open question on *this* image and *these* nodes, and to feed the
sweep, where the P4D4 branch turns simulation off and correctness stops being waived.
Cost: one extra launch per shape. Accepted deliberately, because "use the correct MTP
configuration" is an explicit instruction and phase 1's TP8 evidence came from a
different base image and a different DSA patch set, so it does not transfer.

**Consequence to state next to any number:** the two `fast` points are timing under
forced acceptance with custom all-reduce ON. If a shape's probe says it garbles, that
shape's `fast` number describes a stack that does not decode correctly — the same
caveat the reference carries, and it must be printed next to the number, not buried.

### B — correctness probe (serial, per shape)
Deploy, run the 16-request `temperature=0` probe with simulation **off**, record
coherence and per-rank `spec_accept_length`, decide the all-reduce flag.

### C — the two `fast` points
1. P8D8, CONC 80, simulation **on**.
2. P4D4, CONC 40, simulation **on**.
Rail health counted before and after each.

### D — compare and branch
Per-GPU total/input/output throughput, ITL, TTFT, errors. Then the mission's
branch rule — including **stop and ask** if the difference is within noise.

### E — sweep the winner, then pack up
`full` mode, 5 points, OOM at the top point is an acceptable end state.

## Risks

| risk | handling |
|---|---|
| 66 GB image transfer fills node disk | check free space on 137/136 before `docker load` |
| HiCache teardown is slow | **expected** — the user warned; do not call it a leak or a hang |
| `config.full.sh` requests `flydsl` / `aiter` DSA backends the nightly rejects | substitute `tilelang`, clear `DSA_TOPK_BACKEND` **after** sourcing |
| P4D4 on 8-GPU nodes: which 4 GPUs | pick a contiguous rail-consistent set and record it; the reference used 2,3,4,5 for a different reason (k8s pod on GPU[1]) |
| The other session finishes and frees 135/138 | irrelevant — do not migrate mid-experiment |
| **`*_MAX_RUNNING=128` caps the sweep's top points** | `config.full.sh` sets `PREFILL_MAX_RUNNING` and `DECODE_MAX_RUNNING` to 128 and `*_GRAPH_MAX_BS` to 128. A P8D8 sweep reaching CONC 192/256 would be **server-side queue-limited, not hardware-limited**, and a P4D4 sweep at 128 sits exactly at the cap. Decide before the sweep whether to raise these with concurrency — and if they stay at 128, say so next to the number, because then the top points measure the cap rather than the shape. Raising `GRAPH_MAX_BS` also lengthens CUDA-graph capture and costs VRAM, which is the same budget the OOM end-state is about. **Not yet decided; raise with the user when the branch is known.** |
