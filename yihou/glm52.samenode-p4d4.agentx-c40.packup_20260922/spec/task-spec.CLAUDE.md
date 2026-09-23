# Task: GLM-5.2 same-node 1P1D P4D4 on crsuse2-m2m-276 — AgentX CONC=40 fast

Get a 1P1D P4D4 deployment running with **both legs on one node**, and take one
AgentX **fast** point at **CONC=40**. HiCache OFF for the deliverable; only
after that is clean, turn prefill HiCache on and debug it. Everything else
matches the 20260920 packups.

Task of record: `bench/glm5p2_pd/results/yihou-samenode-p4d4/spec/mission.md`.
That file, not this one, is the authority on scope.

## Background

All prior GLM-5.2 P/D work on this cluster was **cross-node** — 135/138,
136/137, 137/136. Machines are now scarce, so the question is whether the same
P4D4 shape can be folded onto a single 8-GPU node: prefill on GPUs 0-3, decode
on GPUs 4-7, KV moving between two containers on the same host.

Nothing about the engine configuration is under investigation. The MTP recipe,
the `--disable-custom-all-reduce` decode fix, the DSA backend substitutions and
the image are all settled by the packups in `yihou/`. **The single new variable
is co-location.**

## Context — what makes 276 different

`crsuse2-m2m-276` is a **spur** compute node, not one of the dedicated
135-140 boxes.

- **No ssh.** Spur nodes carry `AllowUsers ubuntu root`; `ssh` fails with a
  misleading `Permission denied (publickey)`. Use `spur exec <jobid> <cmd>`,
  which goes through spurd. It runs as `yihou`, `pwd=/`, and docker works.
  Job `165913` holds the node (1-day limit).
- **Shared node.** Other users' containers run here. As of 2026-09-22 the user
  authorised clearing the load; only `hy4-nomtp` held GPU memory.
- **Docker images are node-local.** `/mnt/m2m_nobackup/docker`, 26 TB free. The
  run image must be transferred from 137.
- **One physical IPv4**: `10.245.152.249` on `ens3` (mlx5_0). The eight
  `ionic_0..7` RDMA devices have no IPv4 of their own. GPUs 0-3 pair with
  `ionic_0-3` (NUMA0), GPUs 4-7 with `ionic_4-7` (NUMA1). All 8 ports ACTIVE.

## Key references

| path | what it gives |
|---|---|
| `yihou/glm52.p8d8.agentx-sweep.packup_20260920/scripts/config.yihou.p4d4.sh` | the P4D4 config to inherit from, with per-setting rationale |
| `yihou/glm52.p8d8.agentx-sweep.packup_20260920/analysis/image_transfer.yihou.md` | how the 66 GB image was moved between nodes |
| `yihou/glm52.p8d8.agentx-sweep.packup_20260920/scripts/rdma_map.yihou.md` | GPU↔NIC↔NUMA map methodology |
| `yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/` | why the decode leg needs `--disable-custom-all-reduce` |
| `yihou/glm52-agentx-t2sweep-simacc.packup_20260920/` | the AgentX driver/sweep scripts and fast-vs-full semantics |

## Core principles

1. **Verify first-hand.** Code read, command run, or official source. Rank
   second-hand claims as low confidence until promoted.
2. **Suspend, don't conclude.** Leave a question open rather than guessing.
3. **One variable at a time.**
4. **Docker, not host.**
5. **Deletion rule.** Never delete a file whose path lacks `yihou`.
6. **Nothing outside the workspace.** Never mutate the repo's `topology.tsv`,
   `config.sh`, or `tools/` in place; the harness is **vendored** into the
   workspace and patched there.

## Notable details

- **`tools/topology.py:load()` blocks the shape.** It raises on
  `node in nodes or data_ip in ips`, and 276 offers one node and one IP. This is
  the one harness change the co-location actually forces. Patch the vendored
  copy.
- **Ports are already safe.** `launch.sh:rows()` adds the topology row index to
  each port base, so the two legs never collide.
- **`engine.sh:76` hard-rejects decode HiCache + MTP.** Prefill HiCache only.
- **`config.sh` uses `${DECODE_SIMULATE_ACC_LEN-3.61}`** (single dash): an empty
  but *set* value survives and disables simulation.
- **`launch.sh` / `preflight.sh` / `agentx_bench.sh` refuse a pre-existing
  `OUT_DIR`.**
- **The idle-gauge trap.** A rank that served no decode tokens reports
  `spec_accept_length 0.0`. Trust only ranks whose counters moved.
- **Settle the GPUs between runs.** After `stop.sh`, verify via `rocm-smi` that
  the target GPUs are back at the idle baseline with no KFD processes, then wait
  longer still before the next `launch.sh`.
- CUDA graph capture is ~30 min per bring-up; watch the build directory rather
  than assuming a hang.

## Repository conventions

### DCO sign-off is required on every commit

CI blocks any PR containing a commit without a `Signed-off-by:` trailer. Commit
with `-s`, always, signed off **as yourself** (your own `user.name` /
`user.email`) — never a bot, assistant, or colleague identity. Cherry-picks do
not inherit it; use `git cherry-pick -s`.

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
git rebase --signoff <last-already-signed-commit>
```

## Language

Work in English (code, comments, commits, notes). Report to the user in Chinese.
